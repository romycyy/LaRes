# --- Symbolic policy validation (appended by the framework) ---
# NOT a test suite. This file is concatenated onto LLM-generated source and run as a
# subprocess by lares/core/policy_generation.py. Changing it changes what generated
# policies must satisfy. Never run it directly.
#
# The code above must define a class named GeneratedPolicy subclassing SymbolicPolicy.
# Running in a subprocess is what makes it safe to exercise untrusted source: a crash,
# a hang or a segfault takes down the child, not the search loop.
import argparse
import sys
import traceback

_parse = argparse.ArgumentParser()
_parse.add_argument("path", type=str, help="Path to pickled env state data")
_parse.add_argument("--obs_dim", type=int, required=True)
_parse.add_argument("--action_dim", type=int, required=True)
_parse.add_argument("--env_name", type=str, default="")
_test_args = _parse.parse_args()

# --- Bind the observation schema this policy is written against ---
_schema = None
if _test_args.env_name:
    try:
        from lares.core.obs_schema import get_obs_schema, set_active_schema

        _schema = get_obs_schema(_test_args.env_name)
        set_active_schema(_schema)
    except Exception as _e:
        print(f"Error resolving observation schema for {_test_args.env_name!r}: {_e}")
        exit(1)

# --- Instantiate ---
try:
    _policy = GeneratedPolicy(_test_args.obs_dim, _test_args.action_dim)
except Exception as _e:
    print(f"Error instantiating GeneratedPolicy: {_e}")
    traceback.print_exc()
    exit(1)

# --- Recover the generated source for the static checks ---
# The framework writes: <head><imports><generated code><marker><this file>.
_MARKER = "# --- Symbolic policy validation (appended by the framework) ---"
_source = None
try:
    with open(__file__, "r", encoding="utf-8") as _f:
        _whole = _f.read()
    _idx = _whole.find(_MARKER)
    if _idx > 0:
        _source = _whole[:_idx]
except Exception:
    _source = None

# --- Full pre-rollout gate ---
try:
    from lares.core.policy_validator import validate_policy
except Exception as _e:
    print(f"Error importing the policy validator: {_e}")
    traceback.print_exc()
    exit(1)

_report = validate_policy(
    _policy,
    _test_args.obs_dim,
    _test_args.action_dim,
    source=_source,
    schema=_schema,
    require_schema=bool(_test_args.env_name),
)

for _w in _report.warnings:
    print(f"Warning: {_w}")

if not _report.ok:
    print("Error: policy failed validation.")
    for _e in _report.errors:
        print(f"  {_e}")
    exit(1)

# --- Exercise the policy on real observations when they are available ---
try:
    import pickle

    import torch

    with open(_test_args.path, "rb") as _f:
        _stored_data = pickle.load(_f)
    if len(_stored_data) > 0:
        _sample = _stored_data[0]
        if isinstance(_sample, dict) and "obs" in _sample:
            for _sd in _stored_data[:10]:
                _obs_t = torch.tensor(_sd["obs"], dtype=torch.float32).unsqueeze(0)
                _m, _s = _policy(_obs_t)
                if torch.isnan(_m).any() or torch.isnan(_s).any():
                    print("Error: NaN on real observation data")
                    exit(1)
except Exception:
    pass

print(f"Checks passed: {', '.join(_report.checked)}")
print("Success!")
