"""Print the pen-spin task's spaces and timing (run with Isaac Lab's python on the box)."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
if "--neutral" in sys.argv:
    # The action that holds each actuated joint at its default target, in the env's (sorted) action order.
    from lares.envs.isaac_pen_spin.runner import ControllerContext, make_env

    env = make_env(3, 1)
    env.reset()
    ctx = ControllerContext(env)
    names = [env.hand.joint_names[i] for i in env.actuated_dof_indices]
    print({"action_order": names, "neutral_action": [round(float(x), 3) for x in ctx.neutral_action[0]],
           "lower": [round(float(x), 3) for x in ctx.lower[0]], "upper": [round(float(x), 3) for x in ctx.upper[0]],
           "act_moving_average": env.cfg.act_moving_average, "hand_joint_names": list(env.hand.joint_names)})
    env.close()
    app.close()
    sys.exit(0)
import isaaclab_tasks  # noqa: F401
import lares.envs.isaac_pen_spin as p
print("added", p.register())
from lares.envs.isaac_pen_spin.shadow_hand_spin_env_cfg import ShadowHandSpinStage3EnvCfg
c = ShadowHandSpinStage3EnvCfg()
print("obs", c.observation_space, "act", c.action_space, "dt", c.sim.dt, "decim", c.decimation,
      "ep_s", c.episode_length_s, "obs_type", c.obs_type, "act_moving_avg", c.act_moving_average)
app.close()
