#!/usr/bin/env python
"""
Full LLM-based symbolic policy evolution pipeline.

Stages
------
1. Expert dataset generation (or load from a saved buffer).
2. For each generation:
   a. LLM proposes ``pop_size`` new symbolic policy structures, conditioned on
      the performance of the previous generation's elite candidates.
   b. Each candidate is trained through BC → RL (the inner loop).
   c. Candidates are evaluated and ranked; elites are passed to the next
      generation as feedback context.
3. The best policy found across all generations is saved.

Configuration is driven entirely by a YAML file; see
``config/run_full_evolution.yaml`` for all available options.

Usage (from the project root)::

    python scripts/run_full_evolution.py
    python scripts/run_full_evolution.py --config path/to/custom.yaml
"""

import os
import sys
import time
import argparse
from types import SimpleNamespace

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import yaml  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from lares.core.training_pipeline import (  # noqa: E402
    DemoBuffer,
    EvolutionOrchestrator,
    TASK_DESCRIPTIONS,
    ensure_mujoco_headless_gl,
    generate_dataset,
)
from lares.core.training_logger import TrainingLogger  # noqa: E402
from lares.eval import (  # noqa: E402
    EvaluationManifest,
    ExpertActor,
    PipelineEnvs,
    PromotionPolicy,
    build_report,
    episode_table,
    evaluate_manifest,
    make_manifest_env,
)
from lares.eval.manifest import SPLIT_DEVELOPMENT, SPLIT_TRAIN  # noqa: E402

DEFAULT_CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "run_full_evolution.yaml")
MANIFEST_DIR = os.path.join(_PROJECT_ROOT, "config", "manifests")

# Required config keys (validated on load)
_REQUIRED_KEYS = (
    "env_name",
    "seed",
    "dataset_episodes",
    "eval_episodes",
    "bc_steps",
    "rl_iterations",
    "rl_episodes",
    "model",
    "num_generations",
    "pop_size",
    "elite_num",
    "log_dir",
    "record_demo_gif",
)


# ---------------------------------------------------------------------------
#  Configuration helpers
# ---------------------------------------------------------------------------


def load_config(path: str) -> SimpleNamespace:
    """Load and validate the YAML config.  Returns a SimpleNamespace."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        raise KeyError(f"Config {path} is missing required keys: {missing}")
    raw["record_demo_gif"] = bool(raw["record_demo_gif"])
    if "episode_length" not in raw:
        raw["episode_length"] = 200
    else:
        raw["episode_length"] = int(raw["episode_length"])
    if "use_mt1" not in raw:
        raw["use_mt1"] = False
    else:
        raw["use_mt1"] = bool(raw["use_mt1"])
    if "policy_gen_two_phase" not in raw:
        raw["policy_gen_two_phase"] = False
    else:
        raw["policy_gen_two_phase"] = bool(raw["policy_gen_two_phase"])
    if "policy_impl_mode" not in raw:
        raw["policy_impl_mode"] = "batched"
    else:
        raw["policy_impl_mode"] = str(raw["policy_impl_mode"]).strip().lower()
    raw.setdefault("eval_max_steps", 150)
    raw["eval_max_steps"] = int(raw["eval_max_steps"])
    # Staged development budget (spec.md section 12). ``eval_episodes`` is kept as
    # the expanded budget so older configs keep their meaning.
    raw.setdefault("screen_episodes", 10)
    raw["screen_episodes"] = int(raw["screen_episodes"])
    raw.setdefault("expanded_episodes", int(raw["eval_episodes"]))
    raw["expanded_episodes"] = int(raw["expanded_episodes"])
    raw.setdefault("manifest_dir", MANIFEST_DIR)
    return SimpleNamespace(**raw)


def manifest_path(cfg, split: str) -> str:
    # Resolve against the project root, not the shell's cwd, so a config that
    # names "./config/manifests" keeps working from anywhere.
    base = cfg.manifest_dir
    if not os.path.isabs(base):
        base = os.path.join(_PROJECT_ROOT, base)
    return os.path.join(base, f"{cfg.env_name}_{split}.yaml")


def load_split(cfg, split: str):
    """Load one committed manifest and rebuild the placements it names."""
    path = manifest_path(cfg, split)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Missing {split} manifest: {path}. Build the evaluation contract first "
            f"with: python scripts/lock_baseline.py --build-manifests"
        )
    manifest = EvaluationManifest.load(path)
    return manifest, manifest.resolve_pool()


def parse_args() -> str:
    """Parse CLI args and return the resolved config file path."""
    p = argparse.ArgumentParser(
        description="Full LLM-based symbolic policy evolution pipeline"
    )
    p.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    ns = p.parse_args()
    cfg_path = os.path.abspath(ns.config)
    if not os.path.isfile(cfg_path):
        p.error(f"Config file not found: {cfg_path}")
    return cfg_path


# ---------------------------------------------------------------------------
#  Environment factory
# ---------------------------------------------------------------------------


def make_env(cfg, pool):
    """Create one wrapped MetaWorld environment bound to ``pool``.

    Call once per stream. Two streams must never share an instance: the whole
    point of the manifest is that an episode depends only on its own case.
    """
    from lares.utils import make_metaworld_env, env_wrapper

    env_cfg = SimpleNamespace(
        env_name=cfg.env_name,
        seed=cfg.seed,
        episode_length=cfg.episode_length,
        use_mt1=getattr(cfg, "use_mt1", False),
    )
    tasks = pool.task_index() if pool is not None else None
    raw_env = make_metaworld_env(env_cfg, cfg.seed, tasks=tasks)
    return env_wrapper(raw_env, env_cfg, tasks=tasks)


def policy_space_dims(env) -> tuple[int, int]:
    """Flat observation size and action size for policy construction (Box spaces)."""
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    return obs_dim, action_dim


# ---------------------------------------------------------------------------
#  Display helpers
# ---------------------------------------------------------------------------


def separator(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def print_eval(label: str, result: dict) -> None:
    print(
        f"  {label:35s}  reward={result['mean_reward']:8.2f}  "
        f"success={result['success_rate']:.2f}"
    )


def evaluate_expert_policy(env, env_name: str, manifest, max_steps: int = 150):
    """Run the MetaWorld built-in expert over a manifest.

    Uses the same runner as every candidate, so the expert reference and the
    policies under test are scored on identical ordered cases.
    """
    return evaluate_manifest(ExpertActor(env_name), manifest, env, max_steps=max_steps)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main() -> None:
    cfg_path = parse_args()
    cfg = load_config(cfg_path)

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    log_dir: str = cfg.log_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = TrainingLogger(log_dir=log_dir, task_name=cfg.env_name)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("\nFull LLM Evolution Pipeline")
    print(
        f"Task    : {cfg.env_name} — {TASK_DESCRIPTIONS.get(cfg.env_name, cfg.env_name)}"
    )
    print(f"Seed    : {cfg.seed}")
    print(f"Device  : {gpu}")
    print(f"Config  : {cfg_path}")
    print(f"Log dir : {log_dir}")

    # -----------------------------------------------------------------------
    #  Environment
    # -----------------------------------------------------------------------
    separator("Environment Setup")
    # Headless SSH: MuJoCo must load EGL before MujocoEnv / first rgb_array render.
    ensure_mujoco_headless_gl()

    train_manifest, train_pool = load_split(cfg, SPLIT_TRAIN)
    dev_manifest_full, dev_pool = load_split(cfg, SPLIT_DEVELOPMENT)
    promotion = PromotionPolicy(
        screen_episodes=cfg.screen_episodes,
        expanded_episodes=cfg.expanded_episodes,
    )
    # The orchestrator carves the screening and expanded subsets out of this, so
    # hand it enough cases for the larger of the two.
    dev_manifest = dev_manifest_full.head(
        min(max(cfg.screen_episodes, cfg.expanded_episodes), len(dev_manifest_full))
    )

    # One env per stream. Collection, evaluation and recording must not be able
    # to disturb each other's placements.
    collect_env = make_env(cfg, train_pool)
    rl_env = make_env(cfg, train_pool)
    dev_env = make_env(cfg, dev_pool)
    gif_env = make_env(cfg, dev_pool)
    envs = PipelineEnvs(development=dev_env, rl=rl_env, gif=gif_env)

    obs_dim, action_dim = policy_space_dims(dev_env)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"  train manifest      : {train_manifest.manifest_id} ({len(train_manifest)} cases)")
    print(f"  development manifest: {dev_manifest.manifest_id} ({len(dev_manifest)} cases)")
    print(f"  development budget  : screen {promotion.screen_episodes}, "
          f"expanded {promotion.expanded_episodes} (nested prefixes)")
    print(f"  metaworld           : {dev_manifest.environment.version_or_commit}")
    print(f"  success rule        : {dev_manifest.environment.success_rule}")

    # -----------------------------------------------------------------------
    #  Expert baseline
    # -----------------------------------------------------------------------
    separator("Expert Baseline")
    expert_record = evaluate_expert_policy(
        dev_env, cfg.env_name, dev_manifest, max_steps=cfg.eval_max_steps
    )
    expert_result = expert_record.fitness_dict()
    print_eval("MetaWorld built-in expert", expert_result)
    expert_report = build_report(
        expert_record, candidate_id="scripted_expert", checkpoint="fitted",
        action_dim=action_dim,
    )
    expert_report.save(os.path.join(log_dir, "expert_report.json"))
    print("\n" + episode_table(expert_report, limit=5))

    # -----------------------------------------------------------------------
    #  Stage 1: Expert dataset generation (or load cached buffer)
    # -----------------------------------------------------------------------
    separator("Stage 1: Expert Dataset Generation")
    demo_buffer_path: str = getattr(cfg, "demo_buffer_path", "")
    if demo_buffer_path and os.path.isfile(demo_buffer_path):
        print(f"  Loading cached demo buffer: {demo_buffer_path}")
        demo_buf = DemoBuffer.load(demo_buffer_path)
        print(f"  Loaded {len(demo_buf)} transitions.")
    else:
        dataset_cases = train_manifest.episodes[: cfg.dataset_episodes]
        if len(dataset_cases) < cfg.dataset_episodes:
            print(
                f"  WARNING: train manifest has {len(train_manifest)} cases, "
                f"dataset_episodes={cfg.dataset_episodes}; collecting "
                f"{len(dataset_cases)}."
            )
        demo_buf, dataset_stats = generate_dataset(
            collect_env,
            cfg.env_name,
            dataset_cases,
            max_steps=150,
        )
        buf_path = os.path.join(log_dir, f"demo_{cfg.env_name}.pkl")
        demo_buf.save(buf_path)
        print(
            f"  Collected {len(demo_buf)} transitions "
            f"(reward={dataset_stats['mean_reward']:.2f}, "
            f"success={dataset_stats['mean_success']:.2f})"
        )
        print(f"  Demo buffer saved: {buf_path}")

    # -----------------------------------------------------------------------
    #  OpenAI client
    # -----------------------------------------------------------------------
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or len(api_key) < 10:
        print("\nERROR: OPENAI_API_KEY environment variable is not set. Aborting.")
        sys.exit(1)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    llm_args = SimpleNamespace(
        model=cfg.model,
        policy_gen_two_phase=getattr(cfg, "policy_gen_two_phase", False),
        policy_impl_mode=getattr(cfg, "policy_impl_mode", "batched"),
    )
    print(f"\n  LLM model      : {cfg.model}")
    print(f"  Generations    : {cfg.num_generations}")
    print(f"  Pop size       : {cfg.pop_size}")
    print(f"  Elite num      : {cfg.elite_num}")
    print(f"  BC steps       : {cfg.bc_steps}")
    print(f"  RL iterations  : {cfg.rl_iterations}  ×  {cfg.rl_episodes} episodes")
    print(f"  Record demo GIF: {cfg.record_demo_gif}")
    print(f"  Policy two-phase: {llm_args.policy_gen_two_phase}")
    print(f"  Policy impl mode: {llm_args.policy_impl_mode}")

    # -----------------------------------------------------------------------
    #  Stages 2-4: LLM-based evolution with BC+RL inner loop
    # -----------------------------------------------------------------------
    separator("LLM-Based Structure Evolution (Stages 2–4)")

    orchestrator = EvolutionOrchestrator(
        env_name=cfg.env_name,
        obs_dim=obs_dim,
        action_dim=action_dim,
        num_generations=cfg.num_generations,
        pop_size=cfg.pop_size,
        elite_num=cfg.elite_num,
        bc_steps=cfg.bc_steps,
        rl_iterations=cfg.rl_iterations,
        rl_episodes_per_iter=cfg.rl_episodes,
        log_dir=log_dir,
        record_demo_gif=cfg.record_demo_gif,
        dev_manifest=dev_manifest,
        train_cases=train_manifest.episodes,
        eval_max_steps=cfg.eval_max_steps,
        promotion_policy=promotion,
    )

    t0 = time.time()
    best = orchestrator.run(
        client=client,
        envs=envs,
        demo_buffer=demo_buf,
        args=llm_args,
        logger=logger,
    )
    elapsed = time.time() - t0

    # -----------------------------------------------------------------------
    #  Results summary
    # -----------------------------------------------------------------------
    separator(f"Evolution Complete  (total time: {elapsed:.1f}s)")

    if best is not None and best.get("policy") is not None:
        print("\n  Best evolved policy:")
        print_eval("LLM-evolved policy", best["eval"])
        print(f"  Score          : {best['score']:.2f}")
        print(f"  Parameters     : {best['policy'].count_parameters()}")

        code_path = os.path.join(log_dir, "best_policy_code.py")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(best["code"])
        print(f"\n  Best policy code    saved: {code_path}")

        model_path = os.path.join(log_dir, "best_policy.pt")
        torch.save(best["policy"].state_dict(), model_path)
        print(f"  Best policy weights saved: {model_path}")

        print("\n  Comparison:")
        print(f"  {'Stage':<35s} {'Reward':>10s} {'Success':>10s}")
        print(f"  {'-' * 57}")
        print_eval("MetaWorld built-in expert", expert_result)
        print_eval("LLM-evolved policy (best)", best["eval"])
    else:
        print("\n  WARNING: Evolution produced no valid policies.")

    print()
    collect_env.close()
    envs.close()


if __name__ == "__main__":
    main()
