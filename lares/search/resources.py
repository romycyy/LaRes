"""Resource accounting kept in separate, named counters (``debug_spec.md`` 4.5, D6).

The old record stored behavioural-cloning gradient steps in a field called
``environment_steps`` and never counted a simulator step at all. Optimisation
work and simulator work are different currencies, and every counter here names
exactly one of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Simulator streams whose episodes and steps are counted separately.
EPISODE_STREAMS = ("probe", "screen", "expanded", "repair", "expert_assisted")

COUNTER_KEYS = (
    "bc_gradient_steps",
    "bc_objective_evaluations",
    *(f"{s}_{kind}" for s in EPISODE_STREAMS for kind in ("episodes", "environment_steps")),
    "llm_calls",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_tokens",
)

WALL_TIME_STAGES = ("generation", "fitting", "evaluation", "repair")

#: The name that used to hold gradient steps. Refused in new records.
FORBIDDEN_KEYS = ("environment_steps",)


def episode_steps(result) -> tuple:
    """``(episodes, simulator steps)`` actually run by one manifest evaluation."""
    episodes = list(getattr(result, "episodes", []) or [])
    return len(episodes), int(sum(int(getattr(e, "length", 0) or 0) for e in episodes))


@dataclass
class ResourceLedger:
    counters: dict = field(default_factory=lambda: {k: 0 for k in COUNTER_KEYS})
    wall_time_seconds: dict = field(default_factory=lambda: {k: 0.0 for k in WALL_TIME_STAGES})

    def add(self, key: str, amount) -> None:
        if key not in COUNTER_KEYS:
            raise KeyError(f"unknown resource counter {key!r}; expected one of {COUNTER_KEYS}")
        self.counters[key] = self.counters.get(key, 0) + int(amount)

    def add_episodes(self, stream: str, result) -> None:
        if stream not in EPISODE_STREAMS:
            raise KeyError(f"unknown episode stream {stream!r}; expected one of {EPISODE_STREAMS}")
        episodes, steps = episode_steps(result)
        self.add(f"{stream}_episodes", episodes)
        self.add(f"{stream}_environment_steps", steps)

    def add_wall_time(self, stage: str, seconds: float) -> None:
        if stage not in WALL_TIME_STAGES:
            raise KeyError(f"unknown wall-time stage {stage!r}; expected one of {WALL_TIME_STAGES}")
        self.wall_time_seconds[stage] = self.wall_time_seconds.get(stage, 0.0) + float(seconds)

    def add_llm_totals(self, totals: dict) -> None:
        for key in ("llm_calls", "llm_prompt_tokens", "llm_completion_tokens", "llm_total_tokens"):
            self.add(key, totals.get(key, 0))

    def merge(self, other: "ResourceLedger") -> "ResourceLedger":
        for key, value in other.counters.items():
            self.add(key, value)
        for stage, seconds in other.wall_time_seconds.items():
            self.add_wall_time(stage, seconds)
        return self

    @property
    def simulator_steps(self) -> int:
        return sum(self.counters.get(f"{s}_environment_steps", 0) for s in EPISODE_STREAMS)

    def to_dict(self) -> dict:
        return {**{k: int(self.counters.get(k, 0)) for k in COUNTER_KEYS},
                "wall_time_seconds": {k: float(v) for k, v in self.wall_time_seconds.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "ResourceLedger":
        ledger = cls()
        for key in COUNTER_KEYS:
            ledger.counters[key] = int(d.get(key, 0) or 0)
        for stage, seconds in (d.get("wall_time_seconds") or {}).items():
            if stage in WALL_TIME_STAGES:
                ledger.wall_time_seconds[stage] = float(seconds)
        return ledger


def resource_problems(resources: dict) -> list:
    """Why a record's resource block is not a complete ledger, if it is not."""
    problems = [f"forbidden resource key {k!r}: it once held gradient steps" for k in FORBIDDEN_KEYS if k in resources]
    missing = [k for k in COUNTER_KEYS if k not in resources]
    if missing:
        problems.append(f"resource ledger is missing {missing}")
    if not isinstance(resources.get("wall_time_seconds"), dict):
        problems.append("resource ledger has no wall_time_seconds by stage")
    return problems
