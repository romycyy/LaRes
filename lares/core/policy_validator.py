"""Pre-rollout validation for generated symbolic policies (``spec.md`` FR-2 / AC-1).

Every failure a generated program can have before it is worth a rollout gets its
own category, so repair feedback names one defect rather than a stack trace.
Checks run in order of cost: source inspection first, then a handful of forward
passes, then batch and sensitivity probes.

Use :func:`validate_policy` for the full gate.  ``SymbolicPolicy.validate()``
remains the module blacklist alone.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from lares.core.obs_schema import ObsSchema

# Error categories. Stable strings: repair prompts and experiment records key on them.
RAW_OBS_INDEXING = "raw_obs_indexing"
MISSING_SCHEMA = "missing_schema"
FORBIDDEN_MODULE = "forbidden_module"
UNDECLARED_PARAMETER = "undeclared_parameter"
UNKNOWN_RANGE_ENTRY = "unknown_range_entry"
INVALID_RANGE = "invalid_range"
INIT_OUT_OF_RANGE = "init_out_of_range"
RANGES_NOT_A_DICT = "ranges_not_a_dict"
FORWARD_RAISED = "forward_raised"
WRONG_SHAPE = "wrong_shape"
NON_FINITE = "non_finite"
NON_POSITIVE_STD = "non_positive_std"
BATCH_COUPLING = "batch_coupling"
INSENSITIVE_INPUT = "insensitive_input"
NO_GRADIENT = "no_gradient"
GATE_TELEMETRY = "gate_telemetry"
SERIALIZATION = "serialization"

ALL_CATEGORIES = (
    RAW_OBS_INDEXING,
    MISSING_SCHEMA,
    FORBIDDEN_MODULE,
    UNDECLARED_PARAMETER,
    UNKNOWN_RANGE_ENTRY,
    INVALID_RANGE,
    INIT_OUT_OF_RANGE,
    RANGES_NOT_A_DICT,
    FORWARD_RAISED,
    WRONG_SHAPE,
    NON_FINITE,
    NON_POSITIVE_STD,
    BATCH_COUPLING,
    INSENSITIVE_INPUT,
    NO_GRADIENT,
    GATE_TELEMETRY,
    SERIALIZATION,
)


@dataclass(frozen=True)
class ValidationError:
    category: str
    message: str

    def __str__(self) -> str:
        return f"[{self.category}] {self.message}"


@dataclass
class ValidationReport:
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationError] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def categories(self) -> set[str]:
        return {e.category for e in self.errors}

    def add(self, category: str, message: str) -> None:
        self.errors.append(ValidationError(category, message))

    def warn(self, category: str, message: str) -> None:
        self.warnings.append(ValidationError(category, message))

    def summary(self) -> str:
        if self.ok:
            return f"valid ({len(self.checked)} checks passed)"
        return "\n".join(str(e) for e in self.errors)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checked": list(self.checked),
            "errors": [{"category": e.category, "message": e.message} for e in self.errors],
            "warnings": [{"category": e.category, "message": e.message} for e in self.warnings],
        }


# ---------------------------------------------------------------------------
#  Static source check
# ---------------------------------------------------------------------------


class _ObsSubscriptVisitor(ast.NodeVisitor):
    """Finds raw indexing of the ``forward`` observation argument."""

    def __init__(self):
        self.hits: list[str] = []
        self._obs_names: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name == "forward" and len(node.args.args) >= 2:
            self._obs_names.append(node.args.args[1].arg)
            self.generic_visit(node)
            self._obs_names.pop()
        else:
            self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript):
        target = node.value
        if isinstance(target, ast.Name) and target.id in self._obs_names:
            self.hits.append(
                f"line {node.lineno}: {target.id}[...] indexes the observation directly"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # torch.narrow(obs, ...) / obs.narrow(...) / obs.split(...) slice just as
        # surely as a subscript does.
        sliceish = {"narrow", "split", "chunk", "index_select", "take", "unbind"}
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in sliceish:
            if isinstance(func.value, ast.Name) and func.value.id in self._obs_names:
                self.hits.append(
                    f"line {node.lineno}: {func.value.id}.{func.attr}(...) slices the observation"
                )
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id in self._obs_names:
                    self.hits.append(
                        f"line {node.lineno}: {func.attr}({arg.id}, ...) slices the observation"
                    )
        self.generic_visit(node)


def check_source(code: str, report: ValidationReport | None = None) -> ValidationReport:
    """Reject raw observation indexing in generated source.

    Named accessors are the only sanctioned way to read the observation, so a
    layout mistake becomes a ``KeyError`` naming the field rather than a policy
    that silently reads a quaternion as a position.
    """
    report = report or ValidationReport()
    report.checked.append("source")
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        report.add(FORWARD_RAISED, f"generated source does not parse: {exc}")
        return report
    visitor = _ObsSubscriptVisitor()
    visitor.visit(tree)
    for hit in visitor.hits:
        report.add(
            RAW_OBS_INDEXING,
            f"{hit}. Use self.obs_field(obs, '<name>') instead; the schema owns the layout.",
        )
    return report


# ---------------------------------------------------------------------------
#  Runtime checks
# ---------------------------------------------------------------------------


def _check_parameters(policy, report: ValidationReport) -> None:
    report.checked.append("parameters")
    try:
        ranges = policy.get_param_ranges()
    except Exception as exc:
        report.add(RANGES_NOT_A_DICT, f"get_param_ranges() raised: {exc!r}")
        return
    if not isinstance(ranges, dict):
        report.add(RANGES_NOT_A_DICT, f"get_param_ranges() returned {type(ranges).__name__}, not dict")
        return

    declared = set(ranges)
    registered = {name for name, _ in policy.named_parameters()}

    for name in sorted(registered - declared):
        report.add(
            UNDECLARED_PARAMETER,
            f"parameter {name!r} has no entry in get_param_ranges(); clip_params() would "
            f"leave it unconstrained",
        )
    for name in sorted(declared - registered):
        report.add(
            UNKNOWN_RANGE_ENTRY,
            f"get_param_ranges() declares {name!r}, which is not a registered nn.Parameter",
        )

    for name, bounds in ranges.items():
        if not (isinstance(bounds, (tuple, list)) and len(bounds) == 2):
            report.add(INVALID_RANGE, f"range for {name!r} is not a (lo, hi) pair")
            continue
        lo, hi = bounds
        try:
            lo, hi = float(lo), float(hi)
        except (TypeError, ValueError):
            report.add(INVALID_RANGE, f"range for {name!r} is not numeric: {bounds!r}")
            continue
        if not (math.isfinite(lo) and math.isfinite(hi)):
            report.add(INVALID_RANGE, f"range for {name!r} is not finite: ({lo}, {hi})")
            continue
        if lo >= hi:
            report.add(INVALID_RANGE, f"range for {name!r} has lo={lo} >= hi={hi}")
            continue
        if name in registered:
            value = dict(policy.named_parameters())[name].detach()
            if bool((value < lo).any() or (value > hi).any()):
                report.add(
                    INIT_OUT_OF_RANGE,
                    f"parameter {name!r} initialises outside its declared range "
                    f"[{lo}, {hi}]; clip_params() would move it on the first step",
                )


def _forward(policy, obs, report: ValidationReport, context: str):
    try:
        return policy(obs)
    except Exception as exc:
        report.add(FORWARD_RAISED, f"forward({context}) raised: {type(exc).__name__}: {exc}")
        return None


def _check_outputs(policy, obs_dim, action_dim, report: ValidationReport, batch_sizes) -> None:
    report.checked.append("outputs")
    generator = torch.Generator().manual_seed(0)
    for bs in batch_sizes:
        obs = torch.randn(bs, obs_dim, generator=generator)
        out = _forward(policy, obs, report, f"batch={bs}")
        if out is None:
            return
        if not (isinstance(out, (tuple, list)) and len(out) == 2):
            report.add(WRONG_SHAPE, f"forward() must return (mean, std), got {type(out).__name__}")
            return
        mean, std = out
        for label, t in (("mean", mean), ("std", std)):
            if not torch.is_tensor(t):
                report.add(WRONG_SHAPE, f"{label} is {type(t).__name__}, not a tensor")
                return
            if tuple(t.shape) != (bs, action_dim):
                report.add(
                    WRONG_SHAPE,
                    f"{label} has shape {tuple(t.shape)}, expected ({bs}, {action_dim})",
                )
                return
            if not torch.isfinite(t).all():
                report.add(NON_FINITE, f"{label} contains NaN or Inf at batch size {bs}")
        if torch.is_tensor(std) and bool((std <= 0).any()):
            report.add(
                NON_POSITIVE_STD,
                f"std must be strictly positive; minimum is {float(std.min()):.6g}",
            )


# Probe observations are drawn at several magnitudes. A single standard-normal
# scale drives distance gates to zero: with a 0.06 m threshold and a sharpness of
# 40, unit-scale positions put every sigmoid at exactly 0, so a gated term is
# invisible and both checks below silently measure nothing.
PROBE_SCALES = (0.05, 0.2, 1.0)


def _check_batch_independence(
    policy, obs_dim, report: ValidationReport, batch_size: int = 8
) -> None:
    """Row ``i`` of a batched forward must equal the forward of row ``i`` alone."""
    report.checked.append("batch_independence")
    for scale in PROBE_SCALES:
        generator = torch.Generator().manual_seed(1)
        obs = scale * torch.randn(batch_size, obs_dim, generator=generator)
        batched = _forward(policy, obs, report, "batch independence")
        if batched is None:
            return
        mean_b, std_b = batched
        for i in range(batch_size):
            single = _forward(policy, obs[i : i + 1], report, f"row {i}")
            if single is None:
                return
            mean_s, std_s = single
            if not torch.allclose(mean_b[i : i + 1], mean_s, atol=1e-5, rtol=1e-4):
                report.add(
                    BATCH_COUPLING,
                    f"row {i} of a batched forward differs from that row evaluated alone "
                    f"(observation scale {scale}); the policy mixes information across the "
                    f"batch, usually a reduction missing dim=-1 or a batch-wide normalisation",
                )
                return
            if not torch.allclose(std_b[i : i + 1], std_s, atol=1e-5, rtol=1e-4):
                report.add(
                    BATCH_COUPLING,
                    f"std for row {i} depends on the rest of the batch "
                    f"(observation scale {scale})",
                )
                return


def _check_field_sensitivity(
    policy, schema: ObsSchema, obs_dim, report: ValidationReport, trials: int = 4
) -> None:
    """Required fields must actually influence the action.

    A policy that ignores the goal can still fit much of the expert's motion,
    score plausibly, and be structurally incapable of the task. Sensitivity only
    has to show up at one probed scale: a term behind a contact gate is a real
    dependence even though it is inert far from contact.
    """
    report.checked.append("field_sensitivity")
    for name in schema.required_names:
        f = schema.field(name)
        sensitive = False
        for scale in PROBE_SCALES:
            generator = torch.Generator().manual_seed(2)
            base = scale * torch.randn(trials, obs_dim, generator=generator)
            baseline = _forward(policy, base, report, "sensitivity baseline")
            if baseline is None:
                return
            mean0, _ = baseline
            perturbed = base.clone()
            perturbed[:, f.start : f.stop] += scale * torch.randn(
                trials, f.size, generator=generator
            )
            out = _forward(policy, perturbed, report, f"sensitivity to {name}")
            if out is None:
                return
            mean1, _ = out
            if not torch.allclose(mean0, mean1, atol=1e-7, rtol=0.0):
                sensitive = True
                break
        if not sensitive:
            report.add(
                INSENSITIVE_INPUT,
                f"the action does not change when {name!r} changes, at any probed "
                f"observation scale; the policy ignores a field the task requires",
            )


def _check_gradients(policy, obs_dim, report: ValidationReport) -> None:
    report.checked.append("gradients")
    obs = torch.randn(4, obs_dim, generator=torch.Generator().manual_seed(3))
    policy.zero_grad()
    out = _forward(policy, obs, report, "gradient")
    if out is None:
        return
    mean, std = out
    try:
        (mean.sum() + std.sum()).backward()
    except Exception as exc:
        report.add(FORWARD_RAISED, f"backward() raised: {type(exc).__name__}: {exc}")
        return
    if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in policy.parameters()):
        report.warn(NO_GRADIENT, "no gradient reaches any parameter; the policy cannot be fitted")
    policy.zero_grad()


def _check_gate_telemetry(policy, obs_dim, action_dim, report: ValidationReport) -> None:
    """Telemetry must be detached, batch-shaped and inert."""
    report.checked.append("gate_telemetry")
    if not hasattr(policy, "reset_diagnostics"):
        return
    policy.reset_diagnostics()
    bs = 5
    obs = torch.randn(bs, obs_dim, generator=torch.Generator().manual_seed(4))
    first = _forward(policy, obs, report, "gate telemetry")
    if first is None:
        return
    gates = getattr(policy, "last_gates", {}) or {}
    if not isinstance(gates, dict):
        report.add(GATE_TELEMETRY, f"last_gates is {type(gates).__name__}, not a dict")
        return
    for name, value in gates.items():
        if not torch.is_tensor(value):
            report.add(GATE_TELEMETRY, f"gate {name!r} is {type(value).__name__}, not a tensor")
            continue
        if value.requires_grad:
            report.add(GATE_TELEMETRY, f"gate {name!r} is attached to the graph; detach it")
        if value.shape[0] != bs:
            report.add(
                GATE_TELEMETRY,
                f"gate {name!r} has batch dimension {value.shape[0]}, expected {bs}",
            )
        if not torch.isfinite(value).all():
            report.add(GATE_TELEMETRY, f"gate {name!r} contains NaN or Inf")

    # Recording telemetry must not change the action.
    mean_a, std_a = first
    policy.reset_diagnostics()
    second = _forward(policy, obs, report, "gate telemetry repeat")
    if second is None:
        return
    mean_b, std_b = second
    if not (torch.allclose(mean_a, mean_b) and torch.allclose(std_a, std_b)):
        report.add(
            GATE_TELEMETRY,
            "forward() is not reproducible across calls on the same observation; "
            "diagnostics or hidden state are affecting the action",
        )
    if gates:
        policy.reset_diagnostics()
        if getattr(policy, "last_gates", None):
            report.add(GATE_TELEMETRY, "reset_diagnostics() did not clear last_gates")


def _check_serialization(policy, report: ValidationReport) -> None:
    report.checked.append("serialization")
    try:
        state = policy.state_dict()
        policy.load_state_dict(state)
    except Exception as exc:
        report.add(SERIALIZATION, f"state_dict round-trip failed: {type(exc).__name__}: {exc}")
        return
    registered = {name for name, _ in policy.named_parameters()}
    missing = registered - set(state)
    if missing:
        report.add(
            SERIALIZATION,
            f"parameters absent from state_dict, so a checkpoint would not restore them: "
            f"{sorted(missing)}",
        )


def _check_forbidden_modules(policy, report: ValidationReport) -> None:
    report.checked.append("forbidden_modules")
    try:
        policy.validate()
    except TypeError as exc:
        report.add(FORBIDDEN_MODULE, str(exc))


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------


def validate_policy(
    policy,
    obs_dim: int,
    action_dim: int,
    source: str | None = None,
    schema: ObsSchema | None = None,
    batch_sizes: Sequence[int] = (1, 4, 16),
    require_schema: bool = True,
) -> ValidationReport:
    """Run every pre-rollout gate and return one report.

    Args:
        policy: an instantiated :class:`~lares.core.symbolic_policy.SymbolicPolicy`.
        obs_dim, action_dim: the shapes the policy must honour.
        source: generated source, if available, for the raw-indexing check.
        schema: observation schema; defaults to the one the policy captured.
        batch_sizes: forward-pass batch sizes to exercise.
        require_schema: treat a missing schema as an error rather than a warning.

    Returns:
        :class:`ValidationReport`. Collects every failure rather than stopping at
        the first, so one repair round can address all of them.
    """
    report = ValidationReport()

    if source is not None:
        check_source(source, report)

    schema = schema or getattr(policy, "obs_schema", None)
    if schema is None:
        if require_schema:
            report.add(
                MISSING_SCHEMA,
                "no observation schema is bound; the policy's field names cannot be checked",
            )
    elif schema.obs_dim != obs_dim:
        report.add(
            MISSING_SCHEMA,
            f"schema {schema.env_id} describes {schema.obs_dim} dimensions, "
            f"the environment provides {obs_dim}",
        )

    _check_forbidden_modules(policy, report)
    _check_parameters(policy, report)
    _check_outputs(policy, obs_dim, action_dim, report, batch_sizes)
    if not report.categories & {FORWARD_RAISED, WRONG_SHAPE}:
        _check_batch_independence(policy, obs_dim, report)
        if schema is not None and schema.obs_dim == obs_dim:
            _check_field_sensitivity(policy, schema, obs_dim, report)
        _check_gradients(policy, obs_dim, report)
        _check_gate_telemetry(policy, obs_dim, action_dim, report)
        _check_serialization(policy, report)
    return report


# ---------------------------------------------------------------------------
#  Gate aggregation
# ---------------------------------------------------------------------------


class GateAggregator:
    """Per-episode gate statistics (``spec.md`` 7.3, AC-2).

    Reports mean, minimum, maximum, the fraction of steps above 0.5, the first
    step crossing 0.5, and how many times the gate crossed it.
    """

    THRESHOLD = 0.5

    def __init__(self):
        self._series: dict[str, list[float]] = {}

    def observe(self, gates: dict | None) -> None:
        if not gates:
            return
        for name, value in gates.items():
            tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
            self._series.setdefault(str(name), []).append(float(tensor.detach().mean()))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._series)

    def statistics(self) -> dict:
        out: dict[str, dict] = {}
        for name, series in self._series.items():
            arr = np.asarray(series, dtype=float)
            above = arr > self.THRESHOLD
            first = int(np.argmax(above)) if above.any() else None
            transitions = int(np.sum(above[1:] != above[:-1])) if arr.size > 1 else 0
            out[name] = {
                "mean": float(arr.mean()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "occupancy_above_half": float(above.mean()),
                "first_crossing_step": first,
                "transitions": transitions,
                "num_steps": int(arr.size),
            }
        return out
