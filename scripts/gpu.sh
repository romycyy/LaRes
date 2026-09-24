#!/usr/bin/env bash
# Control the LaRes GPU box and run code on it.
#
# Code is written locally; the server only runs it. `sync` copies the local
# working tree (tracked + untracked-not-ignored files) into ~/LaRes on the
# server, so tests run against uncommitted local edits without a commit/push.
#
# Guard: every sync records a checksum manifest on the server. The next sync
# refuses if any server file changed since, so work done directly on the
# server is never silently overwritten. Use `gpu pull` to bring server
# commits home first.
set -euo pipefail

ID="i-00844bdd35e7d6f96"          # LLM-symbolic-backup, g4dn.2xlarge (Tesla T4)
REGION="us-east-1"
PROFILE="ec2-control"             # IAM user local-ec2-control: start/stop/describe only
KEY="$HOME/.ssh/sybolicRL.pem"
USER_NAME="ubuntu"
REMOTE_DIR="LaRes"                # relative to the remote home
MANIFEST=".git/gpu_sync_manifest" # inside REMOTE_DIR; .git/ keeps it out of git status
REMOTE_ENV='source .venv-metaworld/bin/activate && export MUJOCO_GL=egl'

LOCAL_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new
          -o ConnectTimeout=15 -o ServerAliveInterval=60)

aws_()  { aws --profile "$PROFILE" --region "$REGION" "$@"; }
dns()   { aws_ ec2 describe-instances --instance-ids "$ID" \
            --query 'Reservations[0].Instances[0].PublicDnsName' --output text; }
state() { aws_ ec2 describe-instances --instance-ids "$ID" \
            --query 'Reservations[0].Instances[0].State.Name' --output text; }
host()  { echo "$USER_NAME@$(dns)"; }
rsh()   { ssh "${SSH_OPTS[@]}" "$(host)" "$@"; }
need_running() {
  local s; s="$(state)"
  [ "$s" = running ] || { echo "instance is $s; run: gpu start" >&2; exit 1; }
}
wait_ssh() {
  for _ in $(seq 1 20); do rsh true 2>/dev/null && return 0; sleep 5; done
  echo "sshd did not come up" >&2; exit 1
}

do_sync() {
  need_running
  local h; h="$(host)"
  cd "$LOCAL_ROOT"

  # Refuse if the server holds anything the local side would clobber.
  local remote_head
  remote_head="$(rsh "cd $REMOTE_DIR && git rev-parse HEAD")"
  if ! git cat-file -e "$remote_head^{commit}" 2>/dev/null; then
    echo "server HEAD $remote_head is not in the local repo; run: gpu pull" >&2; exit 1
  fi
  if ! rsh "cd $REMOTE_DIR && if [ -f $MANIFEST ]; then
              md5sum --quiet -c $MANIFEST 2>/dev/null
            else
              test -z \"\$(git status --porcelain)\"
            fi"; then
    echo "server working tree changed since the last sync (edited on the server?)." >&2
    echo "inspect with: gpu connect 'cd $REMOTE_DIR && git status'" >&2; exit 1
  fi

  # Copy existing tracked + untracked-not-ignored files; delete files removed locally.
  git ls-files -co --exclude-standard -z \
    | while IFS= read -r -d '' f; do [ -e "$f" ] && printf '%s\0' "$f"; done \
    | COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --null -T - -czf - \
    | ssh "${SSH_OPTS[@]}" "$h" "tar -xzf - -C $REMOTE_DIR"
  local deleted; deleted="$(git ls-files -d)"
  if [ -n "$deleted" ]; then
    printf '%s\n' "$deleted" | rsh "cd $REMOTE_DIR && xargs -d '\n' rm -f --"
  fi

  # Record what the server now holds, for the next sync's guard.
  rsh "cd $REMOTE_DIR && git ls-files -co --exclude-standard -z | xargs -0 md5sum > $MANIFEST"
  echo "synced $(git rev-parse --short HEAD)$(git diff --quiet HEAD || echo '+dirty') -> $h:$REMOTE_DIR"
}

# Run every test file on the server: unittest-style ones via unittest, the
# self-running ones as scripts. tests/test_generate_policy.py is a harness, not a test.
REMOTE_TESTS='
fail=0
ut=$(grep -l "unittest.TestCase" tests/test_*.py | grep -v test_generate_policy.py | sed "s#/#.#; s#\.py\$##")
[ -n "$ut" ] && { python -m unittest $ut || fail=1; }
for f in tests/test_*.py; do
  case "$f" in tests/test_generate_policy.py) continue;; esac
  grep -q "unittest.TestCase" "$f" && continue
  echo "=== $f"; python "$f" || { echo "FAILED: $f"; fail=1; }
done
exit $fail'

case "${1:-}" in
  start)
    aws_ ec2 start-instances --instance-ids "$ID" >/dev/null
    echo "starting..."; aws_ ec2 wait instance-running --instance-ids "$ID"
    wait_ssh; echo "running: $(dns)" ;;
  stop)
    aws_ ec2 stop-instances --instance-ids "$ID" >/dev/null
    echo "stopping..."; aws_ ec2 wait instance-stopped --instance-ids "$ID"
    echo "stopped" ;;
  status)
    echo "$(state)  $(dns)" ;;
  host)
    host ;;
  connect|ssh)            # gpu connect [remote command...]
    shift; need_running; rsh "$@" ;;
  sync)
    do_sync ;;
  run)                    # gpu run <cmd...>: sync, then run in the repo with the venv active
    shift; [ $# -gt 0 ] || { echo "usage: gpu run <command>" >&2; exit 1; }
    do_sync; rsh "cd $REMOTE_DIR && $REMOTE_ENV && $*" ;;
  test)                   # gpu test [unittest targets...]: sync, then run tests
    shift; do_sync
    if [ $# -gt 0 ]; then rsh "cd $REMOTE_DIR && $REMOTE_ENV && python -m unittest $*"
    else rsh "cd $REMOTE_DIR && $REMOTE_ENV && $REMOTE_TESTS"; fi ;;
  pull)                   # fetch server commits into refs/remotes/gpu/<branch>
    need_running; cd "$LOCAL_ROOT"
    branch="$(rsh "cd $REMOTE_DIR && git branch --show-current")"
    GIT_SSH_COMMAND="ssh ${SSH_OPTS[*]}" \
      git fetch "ssh://$(host)/home/$USER_NAME/$REMOTE_DIR" "+$branch:refs/remotes/gpu/$branch"
    echo "fetched gpu/$branch; merge with: git merge --ff-only gpu/$branch"
    rsh "cd $REMOTE_DIR && git status --short" | sed 's/^/server uncommitted: /' ;;
  get)                    # gpu get <remote path under LaRes> [local dest]: copy results back
    shift; need_running; cd "$LOCAL_ROOT"
    src="$1"; dest="${2:-.}"; mkdir -p "$dest"
    rsh "cd $REMOTE_DIR && tar -czf - $(printf '%q' "$src")" | tar -xzf - -C "$dest" ;;
  *)
    echo "usage: gpu {start|stop|status|host|connect [cmd]|sync|run <cmd>|test [targets]|pull|get <path> [dest]}"
    exit 1 ;;
esac
