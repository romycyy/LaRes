"""The advisor model's calls: propose ideas, analyse a result, review a disagreement, consult on open ideas.

The advisor never runs anything and never writes policy code. It receives a
hashed briefing packet and returns strict JSON, which is parsed against the
research schemas; a malformed entry fails on its own and is recorded as such,
never repaired into shape. Every call is one ``LLMCallLedger`` entry under
``research/llm_calls.jsonl`` with hashed prompt and response files.

``Advisor`` takes a ``call`` function ``(messages) -> (text, usage, model, decoding)``
so tests can script it; ``openai_call`` builds the live one from the same
``_call_llm`` the policy pipeline uses, with the configured model and effort.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict

from lares.research.schemas import (
    SOURCE_ADVISOR,
    Prediction,
    Protocol,
    ResearchIdea,
    ResearchSchemaError,
    Verdict,
    allowed_metrics,
    next_idea_id,
)
from lares.search.generation import (
    LABEL_RESEARCH_ANALYSE,
    LABEL_RESEARCH_CONSULT,
    LABEL_RESEARCH_PROPOSE,
    LABEL_RESEARCH_REVIEW,
    LLMCallLedger,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROMPTS_DIR = os.path.join(_PROJECT_ROOT, "lares", "utils", "research_prompts")
CONSULT_RECOMMENDATIONS = ("run_as_is", "revise", "defer", "drop")
CONSULTATIONS_DIR = "consultations"


def read_prompt(name: str) -> str:
    with open(os.path.join(PROMPTS_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def extract_json(text: str):
    """The first JSON object or array in ``text``, fenced or bare."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = fence.group(1) if fence else text
    start = min([i for i in (body.find("{"), body.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ResearchSchemaError("response contains no JSON")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(body[start:])
    return obj


def openai_call(client, model: str, reasoning_effort: str | None, temperature: float = 1.0):
    """Live call function over the pipeline's ``_call_llm``."""
    from lares.core.policy_generation import _call_llm, check_reasoning_effort

    effort = check_reasoning_effort(reasoning_effort)

    def call(messages):
        choices, prompt_tokens, completion_tokens, total = _call_llm(
            client, 1, model, messages, temperature, reasoning_effort=effort,
        )
        text = choices[0].message.content or ""
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total}
        return text, usage, model, {"temperature": temperature, "reasoning_effort": effort}

    return call


class Advisor:
    def __init__(self, call, root, ledger: LLMCallLedger | None = None):
        self.call = call
        self.root = root
        self.ledger = ledger or LLMCallLedger(root)

    # -- plumbing ---------------------------------------------------------------
    def _ask(self, label: str, system: str, user: str, purpose: str, attempt_ids=()):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        t0 = time.time()
        text, usage, model, decoding = self.call(messages)
        record = self.ledger.record(
            label=label, model=model, temperature=float(decoding.get("temperature", 1.0)),
            messages=messages, responses=[text], usage=usage, wall_time_seconds=time.time() - t0,
            decoding=decoding, purpose=purpose, attempt_ids=list(attempt_ids),
        )
        return text, record.call_id

    # -- propose ------------------------------------------------------------------
    def propose(self, briefing: dict, n: int = 3, focus: str = "") -> dict:
        """Ask for ``n`` ideas. Returns ``{ideas: [ResearchIdea], rejected: [{index, reason, raw}], call_id}``."""
        system = read_prompt("advisor_system.txt")
        user = read_prompt("propose_user.txt").format(n=n, briefing=briefing["text"], focus=focus or "(none)")
        text, call_id = self._ask(LABEL_RESEARCH_PROPOSE, system, user, purpose=f"propose {n} research ideas")
        payload = extract_json(text)
        if isinstance(payload, dict):
            payload = payload.get("ideas", [])
        if not isinstance(payload, list):
            raise ResearchSchemaError("propose response must be a JSON list of ideas")
        if len(payload) != n:
            raise ResearchSchemaError(f"asked for {n} ideas, got {len(payload)}")
        ideas, rejected = [], []
        metrics = allowed_metrics()
        for k, raw in enumerate(payload):
            try:
                idea = self._parse_idea(raw, metrics, briefing["sha256"], call_id, n_existing=len(ideas))
                ideas.append(idea)
            except (ResearchSchemaError, KeyError, TypeError, ValueError) as e:
                rejected.append({"index": k, "reason": f"{type(e).__name__}: {e}", "raw": raw})
        return {"ideas": ideas, "rejected": rejected, "call_id": call_id, "raw_text": text}

    def _parse_idea(self, raw: dict, metrics, briefing_sha, call_id, n_existing) -> ResearchIdea:
        pred = raw["prediction"]
        prediction = Prediction(
            metric=str(pred["metric"]), direction=str(pred["direction"]), threshold=float(pred["threshold"]),
            against=str(pred.get("against", "")), rationale=str(pred.get("rationale", "")),
        )
        pr = raw["protocol"]
        protocol = Protocol(
            script=str(pr["script"]), config=str(pr.get("config", "")), command=str(pr.get("command", "")),
            seeds=[int(s) for s in pr.get("seeds", [])], manifests=list(pr.get("manifests", [])),
            episodes=pr.get("episodes"), llm_calls=pr.get("llm_calls"),
            code_changes=list(pr.get("code_changes", [])),
            needs_gpu=bool(pr.get("needs_gpu", True)), needs_llm=bool(pr.get("needs_llm", False)),
        )
        # Ids are allocated after the existing ledger so a batch never collides.
        base = next_idea_id(self.root)
        number = int(base.split("-")[1]) + n_existing
        idea = ResearchIdea(
            id=f"R-{number:03d}",
            title=str(raw["title"]).strip(),
            hypothesis=str(raw["hypothesis"]).strip(),
            prediction=prediction,
            protocol=protocol,
            falsified_by=str(raw.get("falsified_by", "")).strip(),
            cost_estimate=dict(raw.get("cost_estimate", {})),
            open_question_refs=list(raw.get("open_question_refs", [])),
            briefing_sha256=briefing_sha,
            proposal_call_id=call_id,
            source=SOURCE_ADVISOR,
        )
        idea.validate(metrics)
        return idea

    # -- consult ------------------------------------------------------------------
    def consult(self, ideas, briefing: dict, question: str = "") -> dict:
        """Advice on how to continue with open ideas. Changes no idea: approval stays the user's.

        Returns the parsed reply plus ``call_id``. Refuses a reply that skips an
        idea, names one it was not asked about, or uses an unknown recommendation.
        """
        ids = [i.id for i in ideas]
        if not ids:
            raise ResearchSchemaError("consult needs at least one idea")
        system = read_prompt("advisor_system.txt")
        user = read_prompt("consult_user.txt").format(
            question=question or "(none)",
            ideas=json.dumps([_idea_for_prompt(i) for i in ideas], indent=2, ensure_ascii=False),
            briefing=briefing["text"],
        )
        text, call_id = self._ask(LABEL_RESEARCH_CONSULT, system, user, purpose=f"consult on {', '.join(ids)}",
                                  attempt_ids=ids)
        raw = extract_json(text)
        entries = raw.get("ideas") if isinstance(raw, dict) else None
        if not isinstance(entries, list):
            raise ResearchSchemaError("consult response must be an object with an 'ideas' list")
        answered = [str(e.get("id")) for e in entries]
        if sorted(answered) != sorted(ids):
            raise ResearchSchemaError(f"consult answered {answered}, asked about {ids}")
        for e in entries:
            if e.get("recommendation") not in CONSULT_RECOMMENDATIONS:
                raise ResearchSchemaError(f"{e.get('id')}: recommendation must be one of {CONSULT_RECOMMENDATIONS}")
        unknown = [i for i in raw.get("order", []) if i not in ids]
        if unknown:
            raise ResearchSchemaError(f"consult order names ideas it was not asked about: {unknown}")
        raw["call_id"] = call_id
        raw["briefing_sha256"] = briefing["sha256"]
        raw["question"] = question
        return raw

    # -- analyse ------------------------------------------------------------------
    def analyse(self, idea: ResearchIdea, briefing: dict) -> Verdict:
        """Numbers in, verdict out. Every claim must cite an artifact path from the result."""
        if not idea.result.get("metrics"):
            raise ResearchSchemaError(f"{idea.id} has no recorded result to analyse")
        system = read_prompt("advisor_system.txt")
        user = read_prompt("analyse_user.txt").format(
            idea=json.dumps(_idea_for_prompt(idea), indent=2, ensure_ascii=False),
            result=json.dumps(idea.result, indent=2, ensure_ascii=False),
            prediction_outcome=json.dumps(idea.prediction_outcome, indent=2),
            briefing=briefing["text"],
        )
        text, call_id = self._ask(LABEL_RESEARCH_ANALYSE, system, user, purpose=f"analyse {idea.id}", attempt_ids=[idea.id])
        raw = extract_json(text)
        verdict = Verdict(
            source=SOURCE_ADVISOR, verdict=str(raw["verdict"]), confidence=float(raw["confidence"]),
            reasoning=str(raw.get("reasoning", "")), cited_artifacts=list(raw.get("cited_artifacts", [])),
            next_experiment=str(raw.get("next_experiment", "")), call_id=call_id,
        )
        verdict.validate()
        known = set(idea.result.get("artifacts", []))
        unknown = [a for a in verdict.cited_artifacts if a not in known]
        if unknown:
            raise ResearchSchemaError(f"{idea.id}: advisor cited artifacts not in the result: {unknown}")
        return verdict

    # -- review -------------------------------------------------------------------
    def review(self, idea: ResearchIdea) -> dict:
        """Given both verdicts, ask whether the advisor changes its mind. Returns the parsed reply."""
        advisor = idea.verdict_from(SOURCE_ADVISOR)
        implementer = idea.verdict_from("implementer")
        if advisor is None or implementer is None:
            raise ResearchSchemaError(f"{idea.id}: review needs both verdicts")
        system = read_prompt("advisor_system.txt")
        user = read_prompt("review_user.txt").format(
            idea=json.dumps(_idea_for_prompt(idea), indent=2, ensure_ascii=False),
            result=json.dumps(idea.result, indent=2, ensure_ascii=False),
            advisor=json.dumps(asdict(advisor), indent=2, ensure_ascii=False),
            implementer=json.dumps(asdict(implementer), indent=2, ensure_ascii=False),
        )
        text, call_id = self._ask(LABEL_RESEARCH_REVIEW, system, user, purpose=f"review {idea.id}", attempt_ids=[idea.id])
        raw = extract_json(text)
        raw["call_id"] = call_id
        if raw.get("revised_verdict"):
            v = Verdict(
                source=SOURCE_ADVISOR, verdict=str(raw["revised_verdict"]), confidence=float(raw.get("confidence", 0.5)),
                reasoning=f"[revised after review] {raw.get('reasoning', '')}",
                cited_artifacts=advisor.cited_artifacts, next_experiment=str(raw.get("next_experiment", "")), call_id=call_id,
            )
            idea.add_verdict(v)
        return raw


def _idea_for_prompt(idea: ResearchIdea) -> dict:
    d = idea.to_dict()
    for key in ("history", "verdicts", "result", "prediction_outcome", "briefing_sha256", "proposal_call_id"):
        d.pop(key, None)
    return d
