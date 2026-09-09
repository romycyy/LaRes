"""Freezing the method, and locking the final-test manifest (FR-4, AC-7).

FR-4's last requirement is that winner selection and final estimation use
different data. Every other part of that requirement is enforced by
:mod:`lares.eval.promotion`; this one was not enforced at all. Nothing stopped a
search from scoring a candidate on the final-test manifest, looking at the
number, and carrying on. The manifest would then be development data wearing a
different name, and the interval computed from it would mean nothing.

So the final-test split is locked. :func:`evaluate_manifest` refuses it unless a
:class:`FreezeRecord` has been written and a :func:`final_test_session` is open,
and every evaluation inside such a session is appended to a ledger. Re-scoring
the same code is refused unless the caller says so explicitly, and that
admission is written down too. The ledger is the honest record of how many times
the held-out data was actually looked at, which is the number a selection-effect
argument needs and the number nobody writes down.

A freeze record pins what the run may not change afterwards: model identifiers,
prompt text, structure and policy source, manifests, optimizer settings, budgets
and selection rules. :meth:`FreezeRecord.verify` recomputes the hashes and names
what drifted rather than reporting a boolean.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field

FREEZE_FILENAME = "freeze.json"
LEDGER_FILENAME = "final_test_ledger.jsonl"


class FinalTestLocked(RuntimeError):
    """Raised when the final-test manifest is evaluated outside a session."""


def hash_text(text: str) -> str:
    """The digest used for every frozen artifact."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def hash_directory(directory: str, suffix: str = "") -> dict:
    """One digest per file, so drift names the file that moved."""
    out = {}
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if os.path.isfile(path) and (not suffix or name.endswith(suffix)):
            out[name] = hash_file(path)
    return out


@dataclass
class FreezeRecord:
    """What a final run is not allowed to change once it has started."""

    freeze_id: str
    created_at: str
    #: LLM model identifiers, exactly as sent to the API.
    models: dict = field(default_factory=dict)
    #: Prompt template name to digest.
    prompts: dict = field(default_factory=dict)
    #: Structure or policy source file name to digest.
    code: dict = field(default_factory=dict)
    #: Manifest id to digest of the committed YAML.
    manifests: dict = field(default_factory=dict)
    #: Objective, method, budget, batch size, learning rate, checkpoint rule.
    optimizer: dict = field(default_factory=dict)
    #: Screening and promotion rule, declared before any candidate ran.
    selection: dict = field(default_factory=dict)
    #: Episode counts, generation counts, token and call ceilings.
    budgets: dict = field(default_factory=dict)
    notes: str = ""

    def validate(self):
        for name in ("models", "prompts", "code", "manifests", "optimizer", "selection", "budgets"):
            if not getattr(self, name):
                raise ValueError(
                    f"a freeze record must pin {name}; an empty section means the run "
                    f"is free to change it after seeing a final-test number"
                )
        if not any(m.endswith("final_test") or "final_test" in m for m in self.manifests):
            raise ValueError("a freeze record must name the final-test manifest it locks")
        return True

    def verify(self, prompts=None, code=None, manifests=None) -> dict:
        """Recompute the digests and name what moved.

        Returns a report rather than a boolean, because "the run drifted" is not
        useful and "``initial_system.txt`` changed after the freeze" is.
        """
        drift = {}
        for section, current in (
            ("prompts", prompts), ("code", code), ("manifests", manifests),
        ):
            if current is None:
                continue
            frozen = getattr(self, section)
            changed = sorted(k for k in frozen if k in current and current[k] != frozen[k])
            missing = sorted(k for k in frozen if k not in current)
            added = sorted(k for k in current if k not in frozen)
            if changed or missing or added:
                drift[section] = {"changed": changed, "missing": missing, "added": added}
        return {"matches": not drift, "drift": drift}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FreezeRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, FREEZE_FILENAME)
        if os.path.exists(path):
            raise FileExistsError(
                f"{path} already exists. A freeze is written once; rewriting it after a "
                f"final-test run would erase what the run was actually frozen against."
            )
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return path


def load_freeze(directory: str) -> FreezeRecord:
    path = os.path.join(directory, FREEZE_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"no freeze record at {path}. Final testing needs one: write it with "
            f"scripts/freeze_method.py before the first final-test episode runs."
        )
    with open(path, encoding="utf-8") as f:
        return FreezeRecord.from_dict(json.load(f))


def build_freeze(
    freeze_id: str,
    models: dict,
    prompt_dir: str,
    manifest_dir: str,
    code_files: dict,
    optimizer: dict,
    selection: dict,
    budgets: dict,
    notes: str = "",
) -> FreezeRecord:
    """Digest everything a final run must not change, from what is on disk."""
    record = FreezeRecord(
        freeze_id=freeze_id,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        models=dict(models),
        prompts=hash_directory(prompt_dir, ".txt"),
        code={name: hash_file(path) for name, path in code_files.items()},
        manifests=hash_directory(manifest_dir, ".yaml"),
        optimizer=dict(optimizer),
        selection=dict(selection),
        budgets=dict(budgets),
        notes=notes,
    )
    record.validate()
    return record


# ---------------------------------------------------------------------------
#  The lock
# ---------------------------------------------------------------------------

#: Set only inside :func:`final_test_session`. Module state rather than an
#: argument threaded through every call site, because a lock a caller can forget
#: to pass is not a lock.
_SESSION: dict | None = None


def session_is_open() -> bool:
    return _SESSION is not None


def assert_final_test_allowed(manifest_id: str) -> None:
    """Raise unless a final-test session is open for this manifest."""
    if _SESSION is None:
        raise FinalTestLocked(
            f"{manifest_id} is the final-test split and cannot be evaluated here. "
            f"Winner selection and final estimation must use different data (FR-4). "
            f"Open a final_test_session with a freeze record if this really is the "
            f"final evaluation."
        )


def ledger_entries(path: str) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record_final_test(result, code_hash: str = "", allow_rescore: bool = False) -> dict:
    """Append one final-test evaluation to the ledger.

    Refuses a second look at the same code unless the caller says so, and writes
    the admission into the entry when they do.
    """
    if _SESSION is None:
        raise FinalTestLocked("no final-test session is open")
    path = _SESSION["ledger_path"]
    previous = ledger_entries(path)
    key = code_hash or result.actor_name
    seen = [e for e in previous if e.get("key") == key]
    if seen and not allow_rescore:
        raise FinalTestLocked(
            f"{key} has already been scored on the final-test split "
            f"({len(seen)} time(s), first at {seen[0]['at']}). Scoring it again and "
            f"keeping the better number is selection on the held-out data. Pass "
            f"allow_rescore=True to record a deliberate re-score."
        )
    entry = {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "freeze_id": _SESSION["freeze_id"],
        "key": key,
        "actor": result.actor_name,
        "manifest_id": result.manifest_id,
        "action_mode": result.action_mode,
        "episodes": len(result.episodes),
        "success_rate": result.success_rate,
        "mean_return": result.mean_reward,
        "rescore": bool(seen),
        "previous_looks": len(seen),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


@contextmanager
def final_test_session(freeze: FreezeRecord, directory: str):
    """Authorise final-test evaluation for the duration of the block.

    Nested sessions are refused: two overlapping sessions would write to two
    ledgers and neither would be the record of how often the data was seen.
    """
    global _SESSION
    if _SESSION is not None:
        raise FinalTestLocked("a final-test session is already open")
    freeze.validate()
    os.makedirs(directory, exist_ok=True)
    _SESSION = {
        "freeze_id": freeze.freeze_id,
        "ledger_path": os.path.join(directory, LEDGER_FILENAME),
    }
    try:
        yield _SESSION
    finally:
        _SESSION = None
