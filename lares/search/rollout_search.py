"""Independent rollout search over a structure's declared parameter ranges (R-017, D-014; R-018's exploration lane).

The sampler is the one the research protocols measured: the original vector, then 23 draws in a fixed order, local
draws a Gaussian of 0.1 x each parameter's range around the original and global draws uniform within the ranges.
Shared and remaining draws come from separate child streams of one ``SeedSequence`` so the order of evaluation cannot
shift them. The selected candidate is the best by the checkpoint tie rule (success, then signed progress), with ties
going to the earlier candidate, so the original wins any tie.
"""

from __future__ import annotations

import copy

import numpy as np

from lares.fitting.checkpoints import is_better_checkpoint

LOCAL_SIGMA_FRACTION = 0.1
#: First eight candidates (as in R-008's search arm), then sixteen more alternating local and global (R-016's control).
FIRST_ORDER = ("original", "local", "global", "local", "global", "local", "global", "global")
REST_ORDER = ("local", "global") * 8
CANDIDATES = len(FIRST_ORDER) + len(REST_ORDER)
SAMPLER = "independent: original + 23 local (0.1 x range) / global (uniform) draws, fixed order"


def perturb(policy, kind: str, rng) -> None:
    """One local or global draw over every ranged parameter, then clip to the ranges."""
    import torch

    ranges = policy.get_param_ranges()
    with torch.no_grad():
        for name, param in policy.named_parameters():
            if name not in ranges:
                continue
            lo, hi = float(ranges[name][0]), float(ranges[name][1])
            if kind == "local":
                param.add_(torch.as_tensor(rng.normal(0.0, LOCAL_SIGMA_FRACTION * (hi - lo), size=tuple(param.shape)),
                                           dtype=param.dtype))
            else:
                param.copy_(torch.as_tensor(rng.uniform(lo, hi, size=tuple(param.shape)), dtype=param.dtype))
    policy.clip_params()


def independent_candidates(original, seed: int) -> list:
    """``[(kind, policy)]``: the 24 candidates, the first being an unchanged copy of ``original``."""
    first_rng, rest_rng, _ = (np.random.default_rng(s) for s in np.random.SeedSequence(int(seed)).spawn(3))
    out = []
    for kinds, rng in ((FIRST_ORDER, first_rng), (REST_ORDER, rest_rng)):
        for kind in kinds:
            candidate = copy.deepcopy(original)
            if kind != "original":
                perturb(candidate, kind, rng)
            out.append((kind, candidate))
    return out


def search(original, score, seed: int) -> dict:
    """Score every candidate with ``score(policy) -> RolloutScore`` and keep the best.

    Returns ``{"policy", "selected_index", "trace"}``; ``trace`` lists each candidate's kind and score.
    """
    best, best_score, trace = None, None, []
    for index, (kind, candidate) in enumerate(independent_candidates(original, seed)):
        s = score(candidate)
        trace.append({"index": index, "kind": kind, **s.to_dict()})
        if is_better_checkpoint(s, best_score):
            best, best_score, best_index = candidate, s, index
    return {"policy": best, "selected_index": best_index, "trace": trace}
