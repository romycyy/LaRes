"""Smoke-test the R-042 pen-spin pipeline backend on the box (debug seed 4097 only; not protocol data).

Run from the pipeline's Python (``gpu run``), not Isaac's: each evaluation starts its own Isaac child.

    python scripts/isaac/smoke_pen_spin_backend.py --out /tmp/penspin_smoke
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lares.core import policy_generation as pg  # noqa: E402
from lares.envs.isaac_pen_spin import cases  # noqa: E402
from lares.envs.isaac_pen_spin.pipeline_backend import IsaacExpertActor, IsaacPenSpinStream  # noqa: E402
from lares.eval.runner import PolicyActor, evaluate_manifest  # noqa: E402

POLICY = """
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k_flex = nn.Parameter(torch.tensor(-0.8))
        self.k_roll = nn.Parameter(torch.tensor(0.05))

    def forward(self, obs):
        pos = self.obs_field(obs, "obj_pos")
        rot = self.obs_field(obs, "obj_rot")
        w = self.obs_field(obs, "obj_angvel")
        q = self.obs_field(obs, "hand_dof_pos")
        base = torch.tensor([0.555, 0.176, 0, 0, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, 1.0])
        a = base.expand(obs.shape[0], -1).clone()
        sense = 0.01 * torch.tanh(pos.sum(-1) + rot.sum(-1) + q.sum(-1))
        a[:, 7:10] = self.k_flex + self.k_roll * w[:, :1] + sense.unsqueeze(-1)
        return a, self.fixed_std(a)

    def get_param_ranges(self):
        return {"k_flex": (-1.0, 1.0), "k_roll": (-1.0, 1.0)}
"""

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/penspin_smoke")
args = ap.parse_args()
m = cases.build_manifest(3, 4097, 4, label="smoke")
stream = IsaacPenSpinStream(3, args.out, "smoke")
t0 = time.time()
obs = stream.reset_observations(m)
print("reset_obs", obs.shape, round(time.time() - t0, 1), "s", flush=True)
policy, report = pg.instantiate_generated_policy(POLICY.strip(), 157, 20, env_name=cases.STAGES[3].task_id)
assert report.ok, report.errors
t0 = time.time()
res = evaluate_manifest(PolicyActor(policy), m, stream)
print("symbolic", res.success_rate, [(e.length, e.timed_out, round(e.signed_goal_progress, 2), e.failure_label)
                                      for e in res.episodes], round(time.time() - t0, 1), "s", flush=True)
expert = IsaacExpertActor("logs/pen_spin/r040_ppo_expert/train/model_8239.pt",
                          "f81be523a51cc5bb8bd095d5ed208ba8dd227a37ca05d5dc78283a24d89985d7", 4020)
t0 = time.time()
res = evaluate_manifest(expert, m, stream)
print("expert", res.success_rate, round(time.time() - t0, 1), "s", flush=True)
got = stream.collect_traces(expert, m)
print("traces", got["obs"].shape, got["actions"].shape, json.dumps([r["sustained"] for r in got["episodes"]]))
print("calls", stream.calls, "episodes", stream.episodes)
