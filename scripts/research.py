#!/usr/bin/env python
"""Drive the research-idea loop: direct -> propose -> triage/position -> start -> record -> verdict/analyse -> decide.

The ledger lives under ``research/`` (ideas as JSON, decisions as JSONL, a generated
``STATE.md``). The advisor model (gpt-6.1-sol by default, same client as the policy
pipeline) proposes ideas, triages them and judges results; the implementer (Claude Code)
runs the protocols, states its own triage position and verdict, and an idea moves only
when the two agree. The user's input is a research *direction*; the user rules only on
ideas that exhaust their negotiation rounds.

Usage (from the project root)::

    python scripts/research.py direct "<research direction>"   # the user's input; shown to every advisor call
    python scripts/research.py direct --show
    python scripts/research.py state                            # regenerate research/STATE.md and print it
    python scripts/research.py status                           # one line per open idea
    python scripts/research.py propose [--n 3] [--focus "..."] [--dry-run]     # advisor call (needs OPENAI_API_KEY)
    python scripts/research.py show R-001
    python scripts/research.py triage [R-001 ...] [--dry-run]                   # advisor round over proposed ideas
    python scripts/research.py position R-001 --position approve --reasoning "..." [--changes-applied ...]
    python scripts/research.py revise   R-001 --patch patch.json [--note "..."] # apply agreed changes to a proposal
    python scripts/research.py rule     R-001 --outcome approve --note "..."    # USER override for a stuck triage
    python scripts/research.py consult  R-001 R-002 [--question "..."]          # advisor advice, changes no idea
    python scripts/research.py start    R-001 [--log-dir logs/...] [--require-clean]
    python scripts/research.py record   R-001 --results results.json           # {metrics, controls, artifacts, notes}
    python scripts/research.py verdict  R-001 --verdict supported --confidence 0.8 --reasoning "..." [--next "..."]
    python scripts/research.py analyse  R-001                                  # advisor verdict (needs OPENAI_API_KEY)
    python scripts/research.py decide   R-001 [--summary "..."] [--review] [--handoff]   # agree -> decision
    python scripts/research.py decide   R-001 --ruling refuted --note "..."     # USER rules on an escalated verdict
    python scripts/research.py handoff  R-001                                  # append the decision to RESEARCH_HANDOFF.md

Every subcommand that changes an idea regenerates ``research/STATE.md``. Every proposal
lands in ``research/proposals.jsonl`` and every triage round in ``research/triage.jsonl``.
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
    DEFAULT_MAX_ROUNDS,
    RESEARCH_ROOT,
    SOURCE_IMPLEMENTER,
    SOURCE_USER,
    STATUS_ANALYSED,
    STATUS_DECIDED,
    STATUS_DEFERRED,
    STATUS_ESCALATED,
    STATUS_PROPOSED,
    STATUS_REJECTED,
    STATUS_RUNNING,
    TRIAGE_FINAL_STATUS,
    TRIAGE_POSITIONS,
    VERDICTS,
    ResearchSchemaError,
    Verdict,
    apply_revision,
    decide,
    find_idea,
    load_decisions,
    load_direction,
    load_ideas,
    log_proposal,
    now,
    record_triage_advice,
    record_triage_position,
    rule_triage,
    set_direction,
    verdict_rounds,
)
from lares.research.state import loop_guard, write_state  # noqa: E402

HANDOFF = os.path.join(_PROJECT_ROOT, "RESEARCH_HANDOFF.md")

# Tests replace this with a factory returning an Advisor over a scripted call.
ADVISOR_FACTORY = None


def _advisor(args):
    if ADVISOR_FACTORY is not None:
        return ADVISOR_FACTORY(args)
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


def _direction(args) -> str:
    return load_direction(args.root).get("text", "")


# -- direction and state ---------------------------------------------------------------


def cmd_direct(args):
    if args.show or not args.text:
        d = load_direction(args.root)
        print(d.get("text") or "(no direction set)")
        for h in d.get("history", []):
            print(f"  earlier ({h.get('set_at', '')[:10]} to {h.get('replaced_at', '')[:10]}): {h['text']}")
        return
    d = set_direction(args.text, args.root)
    _finish(args, f"direction set ({d['set_at']}): {d['text']}")


def cmd_state(args):
    path = write_state(args.root)
    print(open(path, "r", encoding="utf-8").read())


def cmd_status(args):
    ideas = [i for i in load_ideas(args.root) if i.status not in (STATUS_DECIDED, STATUS_REJECTED, STATUS_DEFERRED)]
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


# -- propose ---------------------------------------------------------------------------


def cmd_propose(args):
    direction = _direction(args)
    guard = loop_guard(args.root)
    if guard["consecutive_all_rejected"] >= 2 and not args.force:
        raise SystemExit(f"the last {guard['consecutive_all_rejected']} propose calls were rejected in full at triage; "
                         "ask the user for a new direction (`direct \"...\"`) or pass --force")
    extra = (f"User focus for this call: {args.focus}" if args.focus else "")
    briefing = build_briefing(args.root, "propose", extra=extra)
    print(f"briefing: {briefing['path']} ({len(briefing['text'])} chars, sha256 {briefing['sha256'][:12]})")
    if args.dry_run:
        print(briefing["text"])
        return
    out = _advisor(args).propose(briefing, n=args.n, focus=args.focus, direction=direction)
    for idea in out["ideas"]:
        idea.save(args.root)
        log_proposal(args.root, out["call_id"], briefing["sha256"], direction, index=-1, raw=idea.to_dict(), idea_id=idea.id)
        problems = check_idea(idea, args.root)
        flag = "ok" if not problems else f"{len(problems)} protocol problem(s)"
        print(f"saved {idea.id}: {idea.title}  [{flag}]")
        for p in problems:
            print("    -", p)
    for r in out["rejected"]:
        log_proposal(args.root, out["call_id"], briefing["sha256"], direction, index=r["index"], raw=r["raw"],
                     parse_error=r["reason"])
        print(f"unparseable entry {r['index']}: {r['reason']}")
    if out.get("abstained"):
        log_proposal(args.root, out["call_id"], briefing["sha256"], direction, index=-1, raw={},
                     parse_error="advisor abstained: 0 ideas")
        guard = loop_guard(args.root)
        _finish(args, f"call {out['call_id']}: the advisor abstained (0 ideas); {guard['consecutive_all_rejected']} "
                      "consecutive propose call(s) without a proposable idea. Two stop `propose`: ask the user for a "
                      "new direction.")
        return
    _finish(args, f"call {out['call_id']}: {len(out['ideas'])} ideas saved, {len(out['rejected'])} unparseable. "
                  f"Next: `python scripts/research.py triage`")


# -- triage ----------------------------------------------------------------------------


def _triage_targets(args):
    if args.ids:
        ideas = [find_idea(i, args.root) for i in args.ids]
    else:
        ideas = [i for i in load_ideas(args.root) if i.status == STATUS_PROPOSED]
    bad = [i.id for i in ideas if i.status not in (STATUS_PROPOSED, STATUS_DEFERRED)]
    if bad:
        raise SystemExit(f"triage is for proposed or deferred ideas; not: {bad}")
    waiting = [i.id for i in ideas if i.triage.get("rounds") and not i.triage["rounds"][-1].get("implementer")]
    if waiting:
        raise SystemExit(f"these ideas await the implementer's `position` before another round: {waiting}")
    if not ideas:
        raise SystemExit("nothing to triage: no proposed ideas (run `propose` first)")
    return ideas


def cmd_triage(args):
    ideas = _triage_targets(args)
    direction = _direction(args)
    briefing = build_briefing(args.root, "triage")
    print(f"briefing: {briefing['path']} ({len(briefing['text'])} chars, sha256 {briefing['sha256'][:12]})")
    if args.dry_run:
        print(briefing["text"])
        return
    out = _advisor(args).triage(ideas, briefing, direction=direction)
    for idea in ideas:
        entry = out["ideas"][idea.id]
        rnd = record_triage_advice(idea, entry, out["call_id"], briefing["sha256"])
        idea.save(args.root)
        print(f"\n{idea.id} round {rnd['round']}: advisor says **{entry['decision']}** (priority {entry.get('priority')})")
        print(f"  {entry['reasoning']}")
        for c in entry.get("suggested_changes", []):
            print(f"    - change: {c}")
        if entry.get("depends_on"):
            print(f"    depends on: {entry['depends_on']}")
    print(f"\norder: {out['order']}\nadvisor's next step: {out['next_step']}")
    _finish(args, f"call {out['call_id']}: {len(ideas)} idea(s) triaged. The implementer now answers each with "
                  f"`python scripts/research.py position <id> --position <{'|'.join(TRIAGE_POSITIONS)}> --reasoning \"...\"`")


def cmd_position(args):
    idea = find_idea(args.id, args.root)
    res = record_triage_position(idea, args.position, args.reasoning, root=args.root,
                                 changes_applied=args.changes_applied or [], max_rounds=args.max_rounds)
    rnd = idea.triage["rounds"][-1]
    head = f"{idea.id} round {rnd['round']}: advisor {rnd['advisor']['decision']}, implementer {args.position} -> {res['outcome']}"
    if res["outcome"] == "agreed":
        if idea.status in TRIAGE_FINAL_STATUS.values():
            tail = f"status {idea.status}."
            if idea.status == "approved":
                problems = [p for p in check_idea(idea, args.root) if "does not exist" in p]
                tail += " Implementer must first create: " + "; ".join(problems) if problems else " Next: `start`."
        else:
            tail = ("both say revise: apply the changes with `revise` then run `triage` again for the revised idea.")
    elif res["outcome"] == "disagreed":
        tail = (f"still proposed; run `python scripts/research.py triage {idea.id}` for round {rnd['round'] + 1} "
                f"(the advisor sees your objection; {args.max_rounds - rnd['round']} round(s) left before escalation).")
    else:
        tail = (f"ESCALATED after {rnd['round']} rounds. The user rules with "
                f"`python scripts/research.py rule {idea.id} --outcome <approve|reject|defer> --note ...`")
    _finish(args, f"{head}\n  {tail}")


def cmd_revise(args):
    idea = find_idea(args.id, args.root)
    with open(args.patch, "r", encoding="utf-8") as f:
        patch = json.load(f)
    apply_revision(idea, patch, note=args.note or "", root=args.root)
    problems = check_idea(idea, args.root)
    msg = f"{idea.id} revised ({', '.join(patch)}); protocol check: {'ok' if not problems else problems}"
    _finish(args, msg + f"\n  Next: `python scripts/research.py triage {idea.id}`")


def cmd_rule(args):
    idea = find_idea(args.id, args.root)
    rule_triage(idea, args.outcome, args.note, root=args.root)
    _finish(args, f"{idea.id}: user ruling {args.outcome} -> {idea.status}")


# -- consult ---------------------------------------------------------------------------


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


# -- run, record, verdicts, decision ---------------------------------------------------


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
    if idea.status not in (STATUS_ANALYSED, STATUS_ESCALATED):
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
            print(f"advisor review round {verdict_rounds(idea)}: agree_with_implementer={reply.get('agree_with_implementer')} "
                  f"revised={reply.get('revised_verdict')} about={reply.get('disagreement_is_about')}\n  {reply.get('reasoning', '')}")
    decision = decide(idea, args.root, user_ruling=ruling, summary=args.summary or "", next_step=args.next or "",
                      max_rounds=args.max_rounds)
    if decision is None:
        adv, imp = idea.verdict_from("advisor"), idea.verdict_from(SOURCE_IMPLEMENTER)
        if idea.status == STATUS_ESCALATED:
            _finish(args, f"{idea.id} ESCALATED after {verdict_rounds(idea)} advisor verdicts "
                          f"(advisor {adv.verdict}, implementer {imp.verdict}). The user rules with:\n"
                          f"  python scripts/research.py decide {idea.id} --ruling <{'|'.join(VERDICTS)}> --note '...'")
        else:
            _finish(args, f"{idea.id}: verdicts disagree (advisor {adv.verdict}, implementer {imp.verdict}) after "
                          f"{verdict_rounds(idea)} of {args.max_rounds} rounds. Read the advisor's reasoning above; either "
                          f"restate with `verdict {idea.id} ...` and run `decide {idea.id} --review` again, or change your verdict.")
        return
    _finish(args, f"{decision.id} recorded for {idea.id}: {decision.outcome}" + ("" if decision.agreed else " (user ruling)")
                  + "\n  Next: `python scripts/research.py propose` to continue the loop under the current direction.")
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
    if idea.direction:
        block.insert(3, f"**Direction.** {idea.direction}")
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

    s = sub.add_parser("direct"); s.add_argument("text", nargs="?", default=""); s.add_argument("--show", action="store_true")
    s.set_defaults(fn=cmd_direct)
    sub.add_parser("state").set_defaults(fn=cmd_state)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    s = sub.add_parser("show"); s.add_argument("id"); s.set_defaults(fn=cmd_show)
    s = sub.add_parser("propose"); s.add_argument("--n", type=int, default=3); s.add_argument("--focus", default="")
    s.add_argument("--dry-run", action="store_true", help="build and print the briefing, no call")
    s.add_argument("--force", action="store_true", help="propose even after two fully rejected rounds"); s.set_defaults(fn=cmd_propose)
    s = sub.add_parser("triage"); s.add_argument("ids", nargs="*")
    s.add_argument("--dry-run", action="store_true", help="build and print the briefing, no call"); s.set_defaults(fn=cmd_triage)
    s = sub.add_parser("position"); s.add_argument("id"); s.add_argument("--position", required=True, choices=TRIAGE_POSITIONS)
    s.add_argument("--reasoning", required=True); s.add_argument("--changes-applied", nargs="*")
    s.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS); s.set_defaults(fn=cmd_position)
    s = sub.add_parser("revise"); s.add_argument("id"); s.add_argument("--patch", required=True, help="JSON file of fields to merge")
    s.add_argument("--note", default=""); s.set_defaults(fn=cmd_revise)
    s = sub.add_parser("rule"); s.add_argument("id"); s.add_argument("--outcome", required=True, choices=sorted(TRIAGE_FINAL_STATUS))
    s.add_argument("--note", required=True); s.set_defaults(fn=cmd_rule)
    s = sub.add_parser("consult"); s.add_argument("ids", nargs="+"); s.add_argument("--question", default="")
    s.add_argument("--dry-run", action="store_true", help="build and print the briefing, no call"); s.set_defaults(fn=cmd_consult)
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
    s.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    s.add_argument("--handoff", action="store_true", help="also append the decision to RESEARCH_HANDOFF.md"); s.set_defaults(fn=cmd_decide)
    s = sub.add_parser("handoff"); s.add_argument("id"); s.set_defaults(fn=cmd_handoff)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except ResearchSchemaError as e:
        raise SystemExit(f"error: {e}")


if __name__ == "__main__":
    main()
