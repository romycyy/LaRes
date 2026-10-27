"""Print the pen-spin task's spaces and timing (run with Isaac Lab's python on the box)."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import isaaclab_tasks  # noqa: F401
import lares.envs.isaac_pen_spin as p
print("added", p.register())
from lares.envs.isaac_pen_spin.shadow_hand_spin_env_cfg import ShadowHandSpinStage3EnvCfg
c = ShadowHandSpinStage3EnvCfg()
print("obs", c.observation_space, "act", c.action_space, "dt", c.sim.dt, "decim", c.decimation,
      "ep_s", c.episode_length_s, "obs_type", c.obs_type, "act_moving_avg", c.act_moving_average)
app.close()
