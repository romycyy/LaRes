#!/usr/bin/env python
"""Drive the research-idea loop: propose -> approve -> start -> record -> analyse -> verdict -> decide.

The ledger lives under ``research/`` (ideas as JSON, decisions as JSONL, a generated
``STATE.md``). The advisor model (gpt-6.1-sol by default, same client as the policy
pipeline) proposes ideas and judges results; the implementer (Claude Code) runs the
protocols and records its own verdict; a decision is written only when the two agree,
otherwise the idea is escalated to the user.

Usage (from the project root)::

    python scripts/research.py state                       # regenerate research/STATE.md and print it
    python scripts/research.py status                      # one line per open idea
    python scripts/research.py propose [--n 3] [--focus "..."] [--dry-run]   # advisor call (needs OPENAI_API_KEY)
    python scripts/research.py show R-001
    python scripts/research.py consult R-001 R-002 [--question "..."] [--dry-run]   # advisor advice, changes no idea
    python scripts/research.py approve R-001 [--note "..."]                  # USER only
    python scripts/research.py reject  R-001 --reason "..."                   # USER only
    python scripts/research.py start   R-001 [--log-dir logs/...] [--require-clean]
    python scripts/research.py record  R-001 --results results.json           # {metrics, controls, artifacts, notes}
    python scripts/research.py analyse R-001                                  # advisor verdict (needs OPENAI_API_KEY)
    python scripts/research.py verdict R-001 --verdict supported --confidence 0.8 --reasoning "..." [--next "..."]
    python scripts/research.py decide  R-001 [--summary "..."] [--review]     # agree -> decision; disagree -> escalate
    python scripts/research.py decide  R-001 --ruling refuted --note "..."     # USER rules on an escalated idea
    python scripts/research.py handoff R-001                                  # append the decision to RESEARCH_HANDOFF.md

Every subcommand that changes an idea regenerates ``research/STATE.md``.
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.research.advisor import CONSULTATIONS_DIR, Advisor, openai_call  # noqa: E402
from lares.research.briefing import build_briefing  # noqa: E402
from lares.research.protocol import check_idea, start_run  # noqa: E402
from lares.research.schemas import (  # noqa: E402
    RESEARCH_ROOT,
    SOURCE_IMPLEMENTER,
    SOURCE_USER,
    STATUS_ANALYSED,
    STATUS_APPROVED,
    STATUS_DECIDED,
    STATUS_REJECTED,
    STATUS_RUNNING,
    VERDICTS,
    ResearchSchemaError,
    Verdict,
    decide,
    find_idea,
    load_decisions,
    load_ideas,
    now,
)
from lares.research.state import write_state  # noqa: E402

HANDOFF = os.path.join(_PROJECT_ROOT, "RESEARCH_HANDOFF.md")


def _advisor(args):
    from lares.core.policy_generation import DEFAULT_LLM_MODEL, DEFAULT_REASONING_EFFORT

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set; the advisor cannot be called")
    from openai import OpenAI

    model = args.model or DEFAULT_LLM_MODEL
    effort = args.reasoning_effort or DEFAULT_REASONING_EFFORT
    print(f"advisor: {model} (reasoning_effort={effort})")
    return Advisor(openai_call(OpenAI(api_key=api_key), model, effort), args.root)


def _finish(args, message=""):
    path = write_state(args.root)
    if message:
        print(message)
    print(f"state: {os.path.relpath(path, _PROJECT_ROOT)}")


# -- commands -------------------------------------------------------------------------


def cmd_state(args):
    path = write_state(args.root)
    print(open(path, "r", encoding="utf-8").read())


def cmd_status(args):
    ideas = [i for i in load_ideas(args.root) if i.status not in (STATUS_DECIDED, STATUS_REJECTED)]
    if not ideas:
        print("no open ideas")
    for i in ideas:
        print(f"{i.id}  [{i.status:<9}]  {i.title}")
    write_state(args.root)


def cmd_show(args):
    idea = find_idea(args.id, args.root)
    print(json.dumps(idea.to_dict(), indent=2, ensure_ascii=False))
    problems = check_idea(idea, args.root)
    print("\nprotocol check:", "ok" if not problems else "")
    for p in problems:
        print("  -", p)


def cmd_propose(args):
    briefing = build_briefing(args.root, "propose", extra=f"User focus: {args.focus}" if args.focus else "")
    print(f"briefing: {briefing['path']} ({len(briefing['text'])} chars, sha256 {briefing['sha256'][:12]})")
    if args.dry_run:
        print(briefing["text"])
        return
    out = _advisor(args).propose(briefing, n=args.n, focus=args.focus)
    for idea in out["ideas"]:
        idea.save(args.root)
        problems = check_idea(idea, args.root)
        flag = "ok" if not problems else f"{len(problems)} protocol problem(s)"
        print(f"saved {idea.id}: {idea.title}  [{flag}]")
        for p in problems:
            print("    -", p)
    for r in out["rejected"]:
        print(f"rejected entry {r['index']}: {r['reason']}")
    _finish(args, f"call {out['call_id']}: {len(out['ideas'])} ideas saved, {len(out['rejected'])} rejected")


def cmd_consult(args):
    ideas = [find_idea(i, args.root) for i in args.ids]
    closed = [i.id for i in ideas if i.status in (STATUS_DECIDED, STATUS_REJECTED)]
    if closed:
        raise SystemExit(f"consult is for open ideas; closed: {closed}")
    briefing = build_briefing(args.root, "consult", extra=f"User question: {args.question}" if args.question else "")
    print(f"briefing: {briefing['path']} ({len(briefing['text'])} chars, sha256 {briefing['sha256'][:12]})")
    if args.dry_run:
        print(briefing["text"])
        return
    advice = _advisor(args).consult(ideas, briefing, question=args.question)
    directory = os.path.join(args.root, CONSULTATIONS_DIR)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{now().replace(':', '')}_{advice['call_id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(advice, f, indent=2, ensure_ascii=False)
    for e in advice["ideas"]:
        print(f"{e['id']}: {e['recommendation']} — {e.get('reasoning', '')}")
        for c in e.get("suggested_changes", []):
            print(f"    - {c}")
    print(f"order: {advice.get('order')}\nnext: {advice.get('next_step', '')}")
    _finish(args, f"call {advice['call_id']}: advice saved to {os.path.relpath(path, _PROJECT_ROOT)}")


def cmd_approve(args):
    idea = find_idea(args.id, args.root)
    problems = check_idea(idea, args.root)
    blocking = [p for p in problems if "does not exist" not in p]   # missing code is the implementer's job
    if blocking and not args.force:
        raise SystemExit(f"{idea.id} has protocol problems; fix the idea file or pass --force:\n  - " + "\n  - ".join(blocking))
    idea.approval = {"by": SOURCE_USER, "at": now(), "note": args.note or ""}
    idea.transition(STATUS_APPROVED, args.note or "approved by user")
    idea.save(args.root)
    todo = [p for p in problems if "does not exist" in p]
    msg = f"{idea.id} approved."
    if todo:
        msg += " Implementer must first create: " + "; ".join(todo)
    _finish(args, msg)


def cmd_reject(args):
    idea = find_idea(args.id, args.root)
    idea.transition(STATUS_REJECTED, args.reason)
    idea.save(args.root)
    _finish(args, f"{idea.id} rejected: {args.reason}")


def cmd_start(args):
    idea = find_idea(args.id, args.root)
    start_run(idea, args.root, log_dir=args.log_dir or "", require_clean=args.require_clean)
    _finish(args, f"{idea.id} running at git {idea.run['git_sha'][:10]}. Command:\n  {idea.protocol.command or idea.protocol.script}")


def cmd_record(args):
    idea = find_idea(args.id, args.root)
    if idea.status != STATUS_RUNNING:
        raise SystemExit(f"{idea.id} is {idea.status}; only a running idea takes a result")
    with open(args.results, "r", encoding="utf-8") as f:
        result = json.load(f)
    for key in ("metrics", "artifacts"):
        if not result.get(key):
            raise SystemExit(f"results file needs a non-empty {key!r}")
    missing = [a for a in result["artifacts"] if not os.path.exists(os.path.join(_PROJECT_ROOT, a))]
    if missing and not args.allow_missing_artifacts:
        raise SystemExit("artifact paths not found locally (use --allow-missing-artifacts if they live on the GPU box):\n  "
                         + "\n  ".join(missing))
    idea.result = result
    before = (result.get("controls") or {}).get(idea.prediction.metric)
    after = result["metrics"].get(idea.prediction.metric)
    idea.prediction_outcome = idea.prediction.check(before, after)
    idea.run["finished_at"] = now()
    idea.transition(STATUS_ANALYSED, f"prediction matched={idea.prediction_outcome.get('matched')}")
    idea.save(args.root)
    _finish(args, f"{idea.id} result recorded; prediction outcome {idea.prediction_outcome}")


def cmd_analyse(args):
    idea = find_idea(args.id, args.root)
    briefing = build_briefing(args.root, f"analyse_{idea.id}")
    verdict = _advisor(args).analyse(idea, briefing)
    idea.add_verdict(verdict)
    idea.save(args.root)
    _finish(args, f"{idea.id} advisor verdict: {verdict.verdict} ({verdict.confidence:.2f})\n{verdict.reasoning}")


def cmd_verdict(args):
    idea = find_idea(args.id, args.root)
    if idea.status not in (STATUS_ANALYSED, "escalated"):
        raise SystemExit(f"{idea.id} is {idea.status}; record a result first")
    v = Verdict(source=SOURCE_IMPLEMENTER, verdict=args.verdict, confidence=args.confidence,
                reasoning=args.reasoning, cited_artifacts=list(args.cite or []), next_experiment=args.next or "")
    idea.add_verdict(v)
    idea.save(args.root)
    _finish(args, f"{idea.id} implementer verdict: {args.verdict} ({args.confidence:.2f})")


def cmd_decide(args):
    idea = find_idea(args.id, args.root)
    ruling = None
    if args.ruling:
        if not args.note:
            raise SystemExit("--ruling needs --note explaining the user's reasoning")
        ruling = {"outcome": args.ruling, "note": args.note, "by": SOURCE_USER, "at": now()}
    elif args.review:
        adv, imp = idea.verdict_from("advisor"), idea.verdict_from(SOURCE_IMPLEMENTER)
        if adv and imp and adv.verdict != imp.verdict:
            reply = _advisor(args).review(idea)
            idea.save(args.root)
            print(f"advisor review: agree_with_implementer={reply.get('agree_with_implementer')} "
                  f"revised={reply.get('revised_verdict')} about={reply.get('disagreement_is_about')}")
    decision = decide(idea, args.root, user_ruling=ruling, summary=args.summary or "", next_step=args.next or "")
    if decision is None:
        _finish(args, f"{idea.id} ESCALATED: verdicts disagree. The user rules with:\n"
                      f"  python scripts/research.py decide {idea.id} --ruling <{'|'.join(VERDICTS)}> --note '...'")
        return
    _finish(args, f"{decision.id} recorded for {idea.id}: {decision.outcome}" + ("" if decision.agreed else " (user ruling)"))
    if args.handoff:
        _append_handoff(idea, decision)


def _append_handoff(idea, decision):
    block = [
        "",
        f"## {idea.title} ({decision.id}, {decision.recorded_at[:10]})",
        "",
        f"**Idea {idea.id}.** {idea.hypothesis}",
        f"**Prediction.** {idea.prediction.metric} {idea.prediction.direction} by at least {idea.prediction.threshold} "
        f"against {idea.prediction.against or 'the control'}.",
        f"**Protocol.** `{idea.protocol.command or idea.protocol.script}` at git `{idea.run.get('git_sha', '')[:10]}`.",
        f"**Result.** {json.dumps(idea.result.get('metrics'))}; controls {json.dumps(idea.result.get('controls'))}.",
        f"**Decision: {decision.outcome}**" + ("" if decision.agreed else " (verdicts disagreed; user ruling)") + f". {decision.summary}",
    ]
    if decision.next_step:
        block.append(f"**Next.** {decision.next_step}")
    block.append("")
    with open(HANDOFF, "a", encoding="utf-8") as f:
        f.write("\n".join(block))
    print(f"appended {decision.id} to RESEARCH_HANDOFF.md")


def cmd_handoff(args):
    idea = find_idea(args.id, args.root)
    if not idea.decision_id:
        raise SystemExit(f"{idea.id} has no decision yet")
    decision = next(d for d in load_decisions(args.root) if d["id"] == idea.decision_id)
    from lares.research.schemas import Decision

    _append_handoff(idea, Decision(**decision))


# -- parser ---------------------------------------------------------------------------


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=RESEARCH_ROOT, help="research ledger directory (default research/)")
    p.add_argument("--model", default=None, help="advisor model (default: the pipeline's DEFAULT_LLM_MODEL)")
    p.add_argument("--reasoning-effort", default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("state").set_defaults(fn=cmd_state)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    s = sub.add_parser("show"); s.add_argument("id"); s.set_defaults(fn=cmd_show)
    s = sub.add_parser("propose"); s.add_argument("--n", type=int, default=3); s.add_argument("--focus", default="")
    s.add_argument("--dry-run", action="store_true", help="build and print the briefing, no call"); s.set_defaults(fn=cmd_propose)
    s = sub.add_parser("consult"); s.add_argument("ids", nargs="+"); s.add_argument("--question", default="")
    s.add_argument("--dry-run", action="store_true", help="build and print the briefing, no call"); s.set_defaults(fn=cmd_consult)
    s = sub.add_parser("approve"); s.add_argument("id"); s.add_argument("--note", default="")
    s.add_argument("--force", action="store_true"); s.set_defaults(fn=cmd_approve)
    s = sub.add_parser("reject"); s.add_argument("id"); s.add_argument("--reason", required=True); s.set_defaults(fn=cmd_reject)
    s = sub.add_parser("start"); s.add_argument("id"); s.add_argument("--log-dir", default="")
    s.add_argument("--require-clean", action="store_true"); s.set_defaults(fn=cmd_start)
    s = sub.add_parser("record"); s.add_argument("id"); s.add_argument("--results", required=True)
    s.add_argument("--allow-missing-artifacts", action="store_true"); s.set_defaults(fn=cmd_record)
    s = sub.add_parser("analyse"); s.add_argument("id"); s.set_defaults(fn=cmd_analyse)
    s = sub.add_parser("verdict"); s.add_argument("id"); s.add_argument("--verdict", required=True, choices=VERDICTS)
    s.add_argument("--confidence", type=float, required=True); s.add_argument("--reasoning", required=True)
    s.add_argument("--cite", nargs="*"); s.add_argument("--next", default=""); s.set_defaults(fn=cmd_verdict)
    s = sub.add_parser("decide"); s.add_argument("id"); s.add_argument("--summary", default="")
    s.add_argument("--next", default=""); s.add_argument("--review", action="store_true", help="ask the advisor to reconsider on disagreement")
    s.add_argument("--ruling", choices=VERDICTS, default=None); s.add_argument("--note", default="")
    s.add_argument("--handoff", action="store_true", help="also append the decision to RESEARCH_HANDOFF.md"); s.set_defaults(fn=cmd_decide)
    s = sub.add_parser("handoff"); s.add_argument("id"); s.set_defaults(fn=cmd_handoff)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except ResearchSchemaError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
