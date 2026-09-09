"""A frozen bank of symbolic-policy structures (``spec.md`` Phase 3, FR-6, AC-3).

Comparing optimizers only means something if the thing being optimized is held
fixed. This module owns that fixed set: source files under ``structures/``, each
one a complete ``GeneratedPolicy``, loaded and instantiated on demand.

The bank is seeded from the candidates an earlier search actually produced, so
the comparison runs on structures the pipeline generates rather than on ones
chosen to make an optimizer look good. Those candidates predate the named
observation accessors, so :func:`port_raw_indices` rewrites each raw slice into
the accessor for the field that slice covers. The rewrite is exact by
construction and :func:`check_port_equivalence` proves it on random inputs.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
import torch

from lares.core.obs_schema import ObsSchema, active_schema, get_obs_schema

STRUCTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "structures")

#: A structure with at most this many scalar parameters counts as the "simple"
#: family ``spec.md`` section 2.2 proposes as the starting complexity.
SIMPLE_FAMILY_MAX_PARAMS = 10


# ---------------------------------------------------------------------------
#  Porting raw indices to named accessors
# ---------------------------------------------------------------------------


def _slice_to_field(schema: ObsSchema) -> dict[tuple[int, int], str]:
    return {(f.start, f.stop): f.name for f in schema.fields}


def port_raw_indices(code: str, schema: ObsSchema) -> tuple[str, list[str]]:
    """Rewrite ``obs[:, a:b]`` into ``self.obs_field(obs, "<name>")``.

    Only slices that exactly match a schema field are rewritten. Anything left
    over is returned as an unported list rather than guessed at, because a slice
    that straddles two fields has no correct name and silently picking one is
    the failure this whole interface exists to prevent.
    """
    lookup = _slice_to_field(schema)
    unported: list[str] = []

    def repl(match):
        start, stop = int(match.group(1)), int(match.group(2))
        name = lookup.get((start, stop))
        if name is None:
            unported.append(match.group(0))
            return match.group(0)
        return f'self.obs_field(obs, "{name}")'

    ported = re.sub(r"obs\[:,\s*(\d+)\s*:\s*(\d+)\s*\]", repl, code)
    for m in re.finditer(r"obs\[:,\s*[^\]]*\]", ported):
        unported.append(m.group(0))
    return ported, unported


def check_port_equivalence(
    original_code: str,
    ported_code: str,
    obs_dim: int,
    action_dim: int,
    schema: ObsSchema,
    trials: int = 3,
    scales=(0.05, 0.5),
) -> bool:
    """Confirm the ported structure computes exactly what the original did."""
    original = instantiate_source(original_code, obs_dim, action_dim, schema=None)
    ported = instantiate_source(ported_code, obs_dim, action_dim, schema=schema)
    ported.load_state_dict(original.state_dict())
    for scale in scales:
        obs = scale * torch.randn(
            trials, obs_dim, generator=torch.Generator().manual_seed(11)
        )
        with torch.no_grad():
            m0, s0 = original(obs)
            m1, s1 = ported(obs)
        if not (torch.equal(m0, m1) and torch.equal(s0, s1)):
            return False
    return True


# ---------------------------------------------------------------------------
#  Loading
# ---------------------------------------------------------------------------


def instantiate_source(code: str, obs_dim: int, action_dim: int, schema=None):
    """exec a structure source and construct its ``GeneratedPolicy``."""
    import torch.nn as nn

    from lares.core.symbolic_policy import SymbolicPolicy

    namespace = {"torch": torch, "nn": nn, "np": np, "SymbolicPolicy": SymbolicPolicy}
    exec(code, namespace)
    if "GeneratedPolicy" not in namespace:
        raise ValueError("structure source defines no class named GeneratedPolicy")
    with active_schema(schema):
        return namespace["GeneratedPolicy"](obs_dim, action_dim)


@dataclass
class Structure:
    """One frozen structure: its source, its identity and its complexity."""

    structure_id: str
    source: str
    path: str
    obs_dim: int = 39
    action_dim: int = 4
    provenance: str = ""
    num_parameters: int = 0
    num_parameter_tensors: int = 0

    @property
    def family(self) -> str:
        """``simple`` or ``complex``, for the parameter-count ablation (E3)."""
        return "simple" if self.num_parameters <= SIMPLE_FAMILY_MAX_PARAMS else "complex"

    def build(self, schema=None):
        """A fresh policy at the structure's declared initial values."""
        return instantiate_source(
            self.source, self.obs_dim, self.action_dim, schema=schema
        )

    def to_dict(self) -> dict:
        return {
            "structure_id": self.structure_id,
            "path": self.path,
            "provenance": self.provenance,
            "num_parameters": self.num_parameters,
            "num_parameter_tensors": self.num_parameter_tensors,
            "family": self.family,
        }


_PROVENANCE = re.compile(r"^#\s*provenance:\s*(.+)$", re.MULTILINE)


def load_structure(path: str, obs_dim: int = 39, action_dim: int = 4, schema=None) -> Structure:
    """Read one structure file and record its complexity."""
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    match = _PROVENANCE.search(source)
    policy = instantiate_source(source, obs_dim, action_dim, schema=schema)
    return Structure(
        structure_id=os.path.splitext(os.path.basename(path))[0],
        source=source,
        path=path,
        obs_dim=obs_dim,
        action_dim=action_dim,
        provenance=match.group(1).strip() if match else "",
        num_parameters=policy.count_parameters(),
        num_parameter_tensors=len(list(policy.parameters())),
    )


def load_bank(
    directory: str = STRUCTURES_DIR,
    env_id: str = "push-v2",
    obs_dim: int = 39,
    action_dim: int = 4,
    family: str | None = None,
) -> list[Structure]:
    """Load every structure in ``directory``, sorted by id.

    ``family`` filters to ``simple`` or ``complex`` for the complexity ablation.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"no structure bank at {directory}. Build one with "
            f"scripts/build_structure_bank.py"
        )
    schema = get_obs_schema(env_id)
    paths = sorted(
        os.path.join(directory, n)
        for n in os.listdir(directory)
        if n.endswith(".py") and not n.startswith("_")
    )
    structures = [load_structure(p, obs_dim, action_dim, schema) for p in paths]
    if family is not None:
        structures = [s for s in structures if s.family == family]
    return structures


def summarise_bank(structures) -> str:
    lines = [f"{'structure':<28}{'params':>8}{'tensors':>9}  family   provenance", "-" * 96]
    for s in structures:
        lines.append(
            f"{s.structure_id:<28}{s.num_parameters:>8}{s.num_parameter_tensors:>9}"
            f"  {s.family:<8} {s.provenance}"
        )
    return "\n".join(lines)
