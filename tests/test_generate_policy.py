# --- Symbolic policy validation (appended by the framework) ---
# The LLM-generated code above must define a class named GeneratedPolicy
# that subclasses SymbolicPolicy.
import torch
import torch.nn as nn
import argparse
import pickle
import traceback

_parse = argparse.ArgumentParser()
_parse.add_argument("path", type=str, help="Path to pickled env state data")
_parse.add_argument("--obs_dim", type=int, required=True)
_parse.add_argument("--action_dim", type=int, required=True)
_test_args = _parse.parse_args()

# --- Instantiate ---
try:
    _policy = GeneratedPolicy(_test_args.obs_dim, _test_args.action_dim)
except Exception as _e:
    print(f"Error instantiating GeneratedPolicy: {_e}")
    traceback.print_exc()
    exit(1)

# --- Verify no forbidden NN modules ---
_FORBIDDEN = (
    nn.Linear,
    nn.Bilinear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.LSTM,
    nn.LSTMCell,
    nn.GRU,
    nn.GRUCell,
    nn.RNN,
    nn.RNNCell,
    nn.Transformer,
)
for _name, _mod in _policy.named_modules():
    if _name == "":
        continue
    if isinstance(_mod, _FORBIDDEN):
        print(
            f"Error: Policy contains forbidden module "
            f"{type(_mod).__name__} at '{_name}'"
        )
        exit(1)

# --- Verify get_param_ranges ---
try:
    _ranges = _policy.get_param_ranges()
except Exception as _e:
    print(f"Error in get_param_ranges(): {_e}")
    traceback.print_exc()
    exit(1)

if not isinstance(_ranges, dict):
    print(f"Error: get_param_ranges() must return a dict, got {type(_ranges)}")
    exit(1)

for _pname, _bounds in _ranges.items():
    if not (isinstance(_bounds, (tuple, list)) and len(_bounds) == 2):
        print(f"Error: Range for '{_pname}' must be a (lo, hi) tuple")
        exit(1)
    if _bounds[0] >= _bounds[1]:
        print(
            f"Error: Invalid range for '{_pname}': "
            f"lo={_bounds[0]} >= hi={_bounds[1]}"
        )
        exit(1)

# --- Test forward pass at multiple batch sizes ---
for _bs in [1, 4, 16]:
    try:
        _obs = torch.randn(_bs, _test_args.obs_dim)
        _mean, _std = _policy(_obs)
    except Exception as _e:
        print(f"Error in forward pass (batch_size={_bs}): {_e}")
        traceback.print_exc()
        exit(1)

    if _mean.shape != (_bs, _test_args.action_dim):
        print(
            f"Error: mean shape {_mean.shape}, "
            f"expected ({_bs}, {_test_args.action_dim})"
        )
        exit(1)
    if _std.shape != (_bs, _test_args.action_dim):
        print(
            f"Error: std shape {_std.shape}, "
            f"expected ({_bs}, {_test_args.action_dim})"
        )
        exit(1)
    if (_std <= 0).any():
        print(f"Error: std must be strictly positive, got min={_std.min().item():.6f}")
        exit(1)
    if torch.isnan(_mean).any() or torch.isinf(_mean).any():
        print("Error: NaN or Inf in mean output")
        exit(1)
    if torch.isnan(_std).any() or torch.isinf(_std).any():
        print("Error: NaN or Inf in std output")
        exit(1)

# --- Verify gradient flow ---
_obs = torch.randn(4, _test_args.obs_dim)
_policy.zero_grad()
_mean, _std = _policy(_obs)
_loss = _mean.sum() + _std.sum()
try:
    _loss.backward()
except Exception as _e:
    print(f"Error during backward pass: {_e}")
    traceback.print_exc()
    exit(1)

_has_grad = False
for _pname, _param in _policy.named_parameters():
    if _param.grad is not None and _param.grad.abs().sum() > 0:
        _has_grad = True
        break
if not _has_grad:
    print(
        "Warning: No gradients flowing to any parameter "
        "(policy may not be learnable)"
    )

# --- Test with stored environment data if observations are available ---
try:
    with open(_test_args.path, "rb") as _f:
        _stored_data = pickle.load(_f)
    if len(_stored_data) > 0:
        _sample = _stored_data[0]
        if isinstance(_sample, dict) and "obs" in _sample:
            for _sd in _stored_data[:10]:
                _obs_t = torch.tensor(
                    _sd["obs"], dtype=torch.float32
                ).unsqueeze(0)
                _m, _s = _policy(_obs_t)
                if torch.isnan(_m).any() or torch.isnan(_s).any():
                    print("Error: NaN on real observation data")
                    exit(1)
except Exception:
    pass

print("Success!")
