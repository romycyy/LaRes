"""Shadow Hand pen-spinning task, vendored from the GPU box's Isaac Lab checkout.

The env and config files were copied unchanged from
``~/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/shadow_hand/`` (Isaac Lab commit
d94504b plus uncommitted local additions) on 2026-10-27, so results can be traced to code in this
repo. Only the config's relative import was made absolute. The box's checkout still registers the
same ids; ``register()`` registers them from here and does nothing for an id already present.

Semantics worth knowing before reading any number from this task:

- ``spin_rate`` is the pen's angular velocity projected on its *own long axis*, i.e. rolling the
  pen about itself, not the end-over-end twirl of human pen spinning.
- ``success`` is evaluated per physics step (centred within tolerance and spin above threshold);
  the env has no episode-level success. Callers must declare their own episode rule.

Importing this package needs Isaac Lab (``source ~/IsaacLab/source_env.sh``) and a launched app.
"""

TASK_IDS = {
    "Isaac-ShadowHandSpin-Stage1-Direct-v0": "ShadowHandSpinStage1EnvCfg",
    "Isaac-ShadowHandSpin-Stage2-Direct-v0": "ShadowHandSpinStage2EnvCfg",
    "Isaac-ShadowHandSpin-Stage3-Direct-v0": "ShadowHandSpinStage3EnvCfg",
}


def register() -> list[str]:
    """Register the three stage ids with gymnasium; return the ids newly registered."""
    import gymnasium as gym

    added = []
    for task_id, cfg_name in TASK_IDS.items():
        if task_id in gym.registry:
            continue
        gym.register(
            id=task_id,
            entry_point=f"{__name__}.shadow_hand_spin_env:ShadowHandSpinEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": f"{__name__}.shadow_hand_spin_env_cfg:{cfg_name}",
                "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_ppo_cfg:ShadowHandSpinPPORunnerCfg",
            },
        )
        added.append(task_id)
    return added
