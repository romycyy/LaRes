"""R-046 pre-launch check under Isaac Lab's Python: build every new child actor and step it on CUDA tensors.

No simulator and no episode: the observation is random, the context is the runner's fields with the documented
neutral action. Checks the gait actor, a scaffolded generated policy (residual and whole) and the shadow wrapper.

    source ~/IsaacLab/source_env.sh && python scripts/isaac/check_gait_scaffold.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch  # noqa: E402

from lares.core import policy_generation as pg  # noqa: E402
from lares.envs.isaac_pen_spin.gait_scaffold import gait_params_sha256  # noqa: E402
from lares.envs.isaac_pen_spin.layout import GAIT_SCAFFOLD_TASK_ID  # noqa: E402
from lares.envs.isaac_pen_spin.pipeline_child import build_controller  # noqa: E402

NEUTRAL = [0.555, 0.176, 0, 0, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, 1]
SOURCE = '''
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k = nn.Parameter(torch.tensor(0.3))

    def forward(self, obs):
        g = self.obs_field(obs, "gait_action")
        w = self.obs_field(obs, "obj_angvel")
        p = self.obs_field(obs, "obj_pos") + self.obs_field(obs, "obj_rot")[:, :3]
        q = self.obs_field(obs, "hand_dof_pos")
        delta = torch.zeros_like(g)
        delta[:, 7] = self.k + 0.01 * torch.tanh(w.sum(-1) + p.sum(-1) + q.sum(-1))
        delta[:, 8] = -self.k
        return delta, self.fixed_std(delta)

    def get_param_ranges(self):
        return {"k": (0.0, 0.6)}
'''.strip()


def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    n = 4
    ctx = SimpleNamespace(device=dev, num_envs=n, control_dt=1 / 60,
                          neutral_action=torch.tensor(NEUTRAL, device=dev).expand(n, -1).clone())
    obs = [torch.randn(n, 157, device=dev) * 0.2 for _ in range(3)]
    sha = gait_params_sha256()
    policy, report = pg.instantiate_generated_policy(SOURCE, 177, 20, env_name=GAIT_SCAFFOLD_TASK_ID)
    assert report.ok, report.errors
    tmp = tempfile.mkdtemp()
    state = os.path.join(tmp, "state.pt")
    torch.save(policy.state_dict(), state)
    base = {"kind": "symbolic", "source": SOURCE, "obs_dim": 177, "action_dim": 20, "fixed_action_scale": 0.1,
            "execution": "check", "state_path": state, "schema_env_id": GAIT_SCAFFOLD_TASK_ID}
    actors = {"gait": {"kind": "gait", "gait_sha256": sha}}
    for mode in ("residual", "whole"):
        actors[mode] = dict(base, scaffold={"mode": mode, "bound": 0.15, "gait_sha256": sha,
                                            "schema_env_id": GAIT_SCAFFOLD_TASK_ID})
    out = {}
    for name, spec in actors.items():
        ctrl = build_controller(spec, "Isaac-ShadowHandSpin-Stage3-Direct-v0")
        ctrl.reset(ctx)
        acts = [ctrl(o, t, ctx) for t, o in enumerate(obs)]
        assert all(a.shape == (n, 20) and a.device.type == dev.split(":")[0] for a in acts), name
        assert all(torch.isfinite(a).all() and a.abs().max() <= 1 for a in acts), name
        out[name] = torch.stack(acts)
    d_res = (out["residual"] - out["gait"]).abs().max().item()
    d_whole = (out["whole"] - out["gait"]).abs().max().item()
    assert d_res <= 0.15 + 1e-6 and d_whole > 0.15, (d_res, d_whole)

    class Inner:
        name = "inner"

        def reset(self, ctx):
            pass

        def __call__(self, obs, t, ctx):
            return ctx.neutral_action.clone()

    from lares.envs.isaac_pen_spin.gait_scaffold import ShadowGait

    shadow = ShadowGait(Inner())
    shadow.reset(ctx)
    for t, o in enumerate(obs):
        shadow(o, t, ctx)
    g = shadow.gait_actions()
    assert g.shape == (n, 3, 20)
    assert abs(g - out["gait"].permute(1, 0, 2).cpu().numpy()).max() < 1e-5
    print(f"OK device={dev} gait_sha256={sha[:12]} residual_max_dev={d_res:.3f} whole_max_dev={d_whole:.3f}")


if __name__ == "__main__":
    main()
