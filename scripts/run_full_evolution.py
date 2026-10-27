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

import json
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
    # Every LLM call uses the model and reasoning effort named here (gpt-6.1-sol, high; 2026-10-13).
    from lares.core.policy_generation import DEFAULT_REASONING_EFFORT, check_reasoning_effort

    raw["reasoning_effort"] = check_reasoning_effort(raw.get("reasoning_effort", DEFAULT_REASONING_EFFORT))
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
    # Fitting and provenance (debug_spec.md Stage B). The probe cases are the tail
    # of the development manifest, disjoint from the screen/expanded head.
    raw.setdefault("objective", "deterministic_mse")
    raw.setdefault("fixed_action_scale", 0.1)
    raw["fixed_action_scale"] = float(raw["fixed_action_scale"])
    raw.setdefault("probe_episodes", 10)
    raw["probe_episodes"] = int(raw["probe_episodes"])
    raw.setdefault("rollout_checkpoint_probes", 4)
    raw.setdefault("checkpoint_steps", None)
    raw.setdefault("max_repair_per_candidate", 1)
    raw.setdefault("max_llm_calls_per_generation", 50)
    raw.setdefault("max_llm_tokens_per_generation", None)
    raw.setdefault("promotion_challengers", 1)
    raw.setdefault("param_budget", [6, 10])
    if int(raw["rl_iterations"]) != 0:
        raise ValueError(
            f"{path}: rl_iterations must be 0; RL after checkpoint selection would break the "
            f"link between a record's selected state and the evaluated policy"
        )
    if not raw["policy_gen_two_phase"]:
        raise ValueError(f"{path}: policy_gen_two_phase must be true so candidates carry lineage")
    # Two-lane search (decided 2026-09-14): the plan itself is built and checked in main().
    raw.setdefault("two_lane_search", None)
    # R-037: plateau-triggered trace repair (two structure slots) and task manifests built from pool seeds.
    raw.setdefault("plateau_repair", None)
    raw.setdefault("task_manifests", None)
    raw.setdefault("expert_baseline", True)
    raw["expert_baseline"] = bool(raw["expert_baseline"])
    if raw["plateau_repair"] is not None:
        problems = []
        if raw["two_lane_search"] is not None:
            problems.append("plateau_repair and two_lane_search are separate plans; set one")
        if int(raw["num_generations"]) != 1 or int(raw["pop_size"]) != 2:
            problems.append("a plateau-repair run is one generation of two structure slots (num_generations 1, pop_size 2)")
        if int(raw["bc_steps"]) != 0:
            problems.append("bc_steps must be 0: nothing is fitted in a plateau-repair run")
        if raw["task_manifests"] is None and raw.get("pen_spin") is None:
            problems.append("a plateau-repair run needs task_manifests (development and demonstration pool seeds) "
                            "or a pen_spin block (R-042)")
        if problems:
            raise ValueError(f"{path}: " + "; ".join(problems))
    block = raw["two_lane_search"]
    if block is not None:
        problems = []
        if int(raw["num_generations"]) != int(block.get("generations", 0)) or int(raw["num_generations"]) != 1:
            problems.append(f"num_generations {raw['num_generations']} must equal two_lane_search.generations (1)")
        if int(raw["pop_size"]) != int(block.get("candidates", -1)):
            problems.append(f"pop_size {raw['pop_size']} must equal two_lane_search.candidates {block.get('candidates')}")
        if int(raw["bc_steps"]) != 0:
            problems.append("bc_steps must be 0: explorations are zero-shot; the fittable lane's budget is in the plan")
        if raw["policy_impl_mode"] != "per_idea":
            problems.append("policy_impl_mode must be per_idea for the two-lane search")
        if problems:
            raise ValueError(f"{path}: " + "; ".join(problems))
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


def _build_or_load(env_name, split, pool_seed, episodes, horizon, path, manifest_id, disjoint_from=()):
    """A frozen manifest: loaded (and its id checked) when ``path`` exists, otherwise built, checked and saved."""
    from lares.eval.manifest import assert_disjoint, build_manifest

    if os.path.isfile(path):
        manifest = EvaluationManifest.load(path)
        if manifest.manifest_id != manifest_id:
            raise ValueError(f"{path} holds {manifest.manifest_id}, not {manifest_id}; it was frozen for another run")
        return manifest
    manifest, _ = build_manifest(env_name, split, [int(pool_seed)], horizon=horizon, num_episodes=episodes,
                                 manifest_id=manifest_id)
    assert_disjoint(list(disjoint_from) + [manifest])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest.save(path)
    return manifest


def load_task_manifests(cfg, include_fresh: bool = False) -> dict:
    """R-037: the development manifest (screen, promotion, probe) and the demonstration manifest for ``cfg.env_name``,
    built from the config's pool seeds and frozen under ``task_manifests.manifest_dir``.

    Development cases are ordered screen (first ``screen_episodes``), promotion (next ``expanded_episodes``), probe
    (last ``probe_episodes``). The demonstration manifest must share no placement with the development manifest.
    """
    block = cfg.task_manifests
    base = block.get("manifest_dir", "config/manifests/task_pools")
    base = base if os.path.isabs(base) else os.path.join(_PROJECT_ROOT, base)
    dev_seed, demo_seed = int(block["development_pool_seed"]), int(block["demonstration_pool_seed"])
    n_dev = cfg.screen_episodes + cfg.expanded_episodes + cfg.probe_episodes
    n_demo = int(block.get("demonstration_cases", 10))
    if dev_seed == demo_seed:
        raise ValueError("the development and demonstration pool seeds must differ")
    dev = _build_or_load(cfg.env_name, SPLIT_DEVELOPMENT, dev_seed, n_dev, cfg.eval_max_steps,
                         os.path.join(base, f"{cfg.env_name}_development_s{dev_seed}_n{n_dev}.yaml"),
                         f"{cfg.env_name}:development:s{dev_seed}:n{n_dev}")
    demo = _build_or_load(cfg.env_name, SPLIT_DEVELOPMENT, demo_seed, n_demo, cfg.eval_max_steps,
                          os.path.join(base, f"{cfg.env_name}_demonstration_s{demo_seed}_n{n_demo}.yaml"),
                          f"{cfg.env_name}:development_demonstration:s{demo_seed}:n{n_demo}", disjoint_from=[dev])
    if {c.task_hash for c in dev.episodes} & {c.task_hash for c in demo.episodes}:
        raise ValueError("the demonstration manifest shares placements with the development manifest")
    out = {"development": dev, "demonstration": demo}
    if include_fresh and block.get("fresh_evaluation_pool_seed") is not None:
        # Scored only after both incumbents of a block are fixed (scripts/evaluate_on_fresh_manifest.py --frozen);
        # the id is the one that script builds, so the rebuilt manifest can be checked against this file.
        fresh_seed, n_fresh = int(block["fresh_evaluation_pool_seed"]), int(block.get("fresh_evaluation_cases", 30))
        out["fresh_evaluation"] = _build_or_load(
            cfg.env_name, SPLIT_DEVELOPMENT, fresh_seed, n_fresh, cfg.eval_max_steps,
            os.path.join(base, f"{cfg.env_name}_confirmation_s{fresh_seed}.yaml"),
            f"{cfg.env_name}:development_confirmation:s{fresh_seed}:n{n_fresh}", disjoint_from=[dev, demo])
    return out


def parse_args() -> tuple:
    """Parse CLI args: ``(config path, seed override or None, log-dir override or None)``."""
    p = argparse.ArgumentParser(
        description="Full LLM-based symbolic policy evolution pipeline"
    )
    p.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config file (default: {DEFAULT_CONFIG_PATH})",
    )
    p.add_argument("--seed", type=int, default=None,
                   help="run seed for a replicate (R-018): overrides `seed` and becomes the fitting and search seed")
    p.add_argument("--log-dir", default=None, help="override `log_dir` (a fresh directory per replicate run)")
    p.add_argument("--repair", choices=("enabled", "disabled"), default=None,
                   help="R-037: override plateau_repair.repair (the paired repair-on / repair-off runs of one block)")
    p.add_argument("--dev-pool-seed", type=int, default=None, help="R-037: override task_manifests.development_pool_seed")
    p.add_argument("--demo-pool-seed", type=int, default=None,
                   help="R-037: override task_manifests.demonstration_pool_seed")
    p.add_argument("--fresh-pool-seed", type=int, default=None,
                   help="R-037: override task_manifests.fresh_evaluation_pool_seed")
    p.add_argument("--freeze-manifests", action="store_true",
                   help="R-037: build (or verify) the task manifests, print their ids and hashes, and exit; no LLM call "
                        "and no episode")
    p.add_argument("--study", choices=("keyframe_repair",), default=None,
                   help="R-042: run the whole pen-spin keyframe-repair study (parity gate, blocks, arms, fresh cases)")
    p.add_argument("--block", type=int, default=None, help="R-042 child: the block to run (with --phase or --arm)")
    p.add_argument("--phase", choices=("prefix",), default=None, help="R-042 child: run the block's prefix only")
    p.add_argument("--arm", choices=("no_trace", "uniform", "event"), default=None,
                   help="R-042 child: run one continuation arm in a copy of the block's prefix run")
    ns = p.parse_args()
    cfg_path = os.path.abspath(ns.config)
    if not os.path.isfile(cfg_path):
        p.error(f"Config file not found: {cfg_path}")
    overrides = {"repair": ns.repair, "development_pool_seed": ns.dev_pool_seed,
                 "demonstration_pool_seed": ns.demo_pool_seed, "fresh_evaluation_pool_seed": ns.fresh_pool_seed,
                 "freeze_manifests": ns.freeze_manifests or None, "study": ns.study, "block": ns.block,
                 "phase": ns.phase, "arm": ns.arm}
    return cfg_path, ns.seed, ns.log_dir, {k: v for k, v in overrides.items() if v is not None}


def apply_overrides(cfg, overrides: dict):
    """Apply the R-037 CLI overrides to their config blocks; refuse one whose block is absent."""
    if "repair" in overrides:
        if cfg.plateau_repair is None:
            raise ValueError("--repair needs a plateau_repair block in the config")
        cfg.plateau_repair = {**cfg.plateau_repair, "repair": overrides["repair"]}
    for key in ("development_pool_seed", "demonstration_pool_seed", "fresh_evaluation_pool_seed"):
        if key in overrides:
            if cfg.task_manifests is None:
                raise ValueError(f"--{key} needs a task_manifests block in the config")
            cfg.task_manifests = {**cfg.task_manifests, key: int(overrides[key])}
    return cfg


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


# ---------------------------------------------------------------------------
#  R-042: Shadow Hand pen spinning through the same orchestrator
# ---------------------------------------------------------------------------


def pen_spin_stream(cfg, name: str, work_dir: str):
    from lares.envs.isaac_pen_spin.pipeline_backend import IsaacPenSpinStream

    ps = cfg.pen_spin
    return IsaacPenSpinStream(int(ps["stage"]), work_dir, name, isaac_env=ps["isaac_env"])


def run_pen_spin_block(cfg, cfg_path: str, block: int, phase: str | None, arm: str | None) -> None:
    """One orchestrator run of a block: its prefix (slot 0 and the trigger) or one continuation arm."""
    from lares.core.training_pipeline import PREFIX_STATE
    from lares.envs.isaac_pen_spin import study
    from lares.search.lanes import load_plateau_repair_plan

    ps = cfg.pen_spin
    blk = ps["blocks"][block]
    bdir = os.path.join(ps["study_dir"], f"block{block}")
    log_dir = os.path.join(bdir, "prefix" if phase == "prefix" else arm)
    if phase != "prefix" and not os.path.isfile(os.path.join(log_dir, PREFIX_STATE)):
        raise SystemExit(f"{log_dir} holds no prefix state; the driver copies the prefix run first")
    seed = int(blk["generation_seed"])
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = study.load("development", int(blk["development_seed"]))
    probe = study.load("probe", int(blk["probe_seed"]))
    demo = study.load("demo", int(ps["demonstration_seed"]))
    if len(dev) != cfg.screen_episodes + cfg.expanded_episodes or len(probe) != cfg.probe_episodes:
        raise ValueError("pen-spin development/probe manifests do not match screen + promotion and probe sizes")
    isaac = os.path.join(log_dir, "isaac")
    envs = PipelineEnvs(development=pen_spin_stream(cfg, "development", isaac), rl=None,
                        gif=pen_spin_stream(cfg, "reset", isaac), probe=pen_spin_stream(cfg, "probe", isaac))
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if len(api_key) < 10:
        raise SystemExit("OPENAI_API_KEY is not set")
    from openai import OpenAI

    llm_args = SimpleNamespace(model=cfg.model, reasoning_effort=cfg.reasoning_effort, policy_gen_two_phase=True,
                               policy_impl_mode=cfg.policy_impl_mode)
    fork = {"phase": "prefix"} if phase == "prefix" else {
        "phase": "continuation", "arm": arm, "demonstrations": os.path.join(ps["study_dir"], "demonstrations.json")}
    orchestrator = EvolutionOrchestrator(
        env_name=cfg.env_name, obs_dim=157, action_dim=20, num_generations=1, pop_size=2, elite_num=1, bc_steps=0,
        rl_iterations=0, rl_episodes_per_iter=0, log_dir=log_dir, record_demo_gif=False, dev_manifest=dev,
        train_cases=(), eval_max_steps=cfg.eval_max_steps,
        promotion_policy=PromotionPolicy(screen_episodes=cfg.screen_episodes, expanded_episodes=cfg.expanded_episodes),
        rollout_checkpoint_probes=0, objective=cfg.objective, checkpoint_steps=[],
        fixed_action_scale=cfg.fixed_action_scale, probe_manifest=probe,
        max_repair_per_candidate=cfg.max_repair_per_candidate,
        max_llm_calls_per_generation=cfg.max_llm_calls_per_generation, param_budget=tuple(cfg.param_budget),
        max_llm_tokens_per_generation=cfg.max_llm_tokens_per_generation, promotion_challengers=1,
        lane_plan=load_plateau_repair_plan(cfg.plateau_repair), fit_seed=seed, demonstration_manifest=demo,
        demonstration_env=None, plateau_fork=fork,
    )
    best = orchestrator.run(client=OpenAI(api_key=api_key), envs=envs, demo_buffer=None, args=llm_args,
                            logger=TrainingLogger(log_dir=log_dir, task_name=cfg.env_name))
    inc = {"candidate_id": orchestrator._incumbent_id, "phase": phase or "continuation", "arm": arm,
           "fixed_action_scale": cfg.fixed_action_scale,
           "isaac_processes": sum(e.calls for e in (envs.development, envs.gif, envs.probe)),
           "isaac_episodes": sum(e.episodes for e in (envs.development, envs.gif, envs.probe))}
    if best is not None and best.get("policy") is not None:
        with open(os.path.join(log_dir, "best_policy_code.py"), "w", encoding="utf-8") as f:
            f.write(best["code"])
        torch.save(best["policy"].state_dict(), os.path.join(log_dir, "best_policy.pt"))
        inc.update(code_path="best_policy_code.py", state_path="best_policy.pt")
    with open(os.path.join(log_dir, "final_incumbent.json" if phase != "prefix" else "prefix_incumbent.json"), "w") as f:
        json.dump(inc, f, indent=2)


def run_pen_spin_study(cfg, cfg_path: str) -> None:
    """The R-042 driver: parity gate, then per block the prefix, demonstrations (once), three arms, fresh cases."""
    import shutil
    import subprocess

    from lares.core.training_pipeline import PREFIX_STATE
    from lares.envs.isaac_pen_spin import study
    from lares.search.lanes import load_plateau_repair_plan

    ps = cfg.pen_spin
    sdir = ps["study_dir"]
    os.makedirs(sdir, exist_ok=True)
    study.freeze_manifests(ps)
    factory = lambda name, work: pen_spin_stream(cfg, name, work)  # noqa: E731
    parity = study.parity_gate(ps, sdir, factory, ps["expert"]["protocol_config"])
    print("parity gate:", json.dumps(parity))
    if not parity["passes"]:
        raise SystemExit(f"R-042 parity gate failed; no LLM call is made: {parity}")
    plan = load_plateau_repair_plan(cfg.plateau_repair)
    me = [sys.executable, "-u", os.path.abspath(__file__), "--config", cfg_path]
    blocks = []
    for b, blk in enumerate(ps["blocks"]):
        bdir = os.path.join(sdir, f"block{b}")
        prefix = os.path.join(bdir, "prefix")
        if not os.path.isfile(os.path.join(prefix, PREFIX_STATE)):
            subprocess.run(me + ["--block", str(b), "--phase", "prefix"], check=True)
        state = json.load(open(os.path.join(prefix, PREFIX_STATE)))
        row = {"block": b, "trigger": state["trigger"], "eligible": bool(state["trigger"]["fired"])}
        if not row["eligible"]:
            print(f"block {b}: no plateau parent ({state['trigger']['reason']}); reported as missing")
            blocks.append(row)
            continue
        demos = study.demonstrations(ps, sdir, factory, plan.min_demo_successes, plan.trace_episodes)
        if not demos["qualified"]:
            summary = {"inconclusive": f"demonstrations {demos['successes']}/{demos['episodes']} < "
                                       f"{plan.min_demo_successes}", "parity": parity, "blocks": blocks}
            json.dump(summary, open(os.path.join(sdir, "summary.json"), "w"), indent=2)
            raise SystemExit(summary["inconclusive"])
        fresh_m = study.load("fresh", int(blk["fresh_seed"]))
        row["fresh"] = {}
        for arm in study.ARMS:
            adir = os.path.join(bdir, arm)
            if not os.path.isfile(os.path.join(adir, "final_incumbent.json")):
                if os.path.exists(adir):
                    raise SystemExit(f"{adir} exists without a final incumbent: an interrupted arm is not rerun")
                shutil.copytree(prefix, adir)
                subprocess.run(me + ["--block", str(b), "--arm", arm], check=True)
            row["fresh"][arm] = study.fresh_evaluation(adir, fresh_m, factory, cfg.env_name)
        blocks.append(row)
        json.dump(study.summarise(blocks), open(os.path.join(sdir, "summary.partial.json"), "w"), indent=2)
    summary = study.summarise(blocks)
    summary["parity"] = parity
    with open(os.path.join(sdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "blocks"}, indent=2))


def main() -> None:
    cfg_path, seed_override, log_dir_override, overrides = parse_args()
    cfg = apply_overrides(load_config(cfg_path), overrides)
    if getattr(cfg, "pen_spin", None) is not None:
        from lares.envs.isaac_pen_spin import study

        if overrides.get("freeze_manifests"):
            print(json.dumps(study.freeze_manifests(cfg.pen_spin), indent=2))
        elif overrides.get("study"):
            run_pen_spin_study(cfg, cfg_path)
        elif overrides.get("block") is not None:
            run_pen_spin_block(cfg, cfg_path, int(overrides["block"]), overrides.get("phase"), overrides.get("arm"))
        else:
            raise SystemExit("a pen_spin config runs with --freeze-manifests, --study or --block")
        return
    if overrides.get("freeze_manifests"):
        if cfg.task_manifests is None:
            raise SystemExit("--freeze-manifests needs a task_manifests block")
        ensure_mujoco_headless_gl()
        from lares.utils.hashing import sha256_text

        for role, m in load_task_manifests(cfg, include_fresh=True).items():
            digest = sha256_text("\n".join(f"{c.case_id}:{c.task_hash}:{c.reset_seed}" for c in m.episodes))
            print(f"  {role:17s} {m.manifest_id}  {len(m)} cases  cases_sha256={digest}")
        return
    if seed_override is not None:
        cfg.seed = int(seed_override)
        cfg.fit_seed = int(seed_override)
    if log_dir_override:
        cfg.log_dir = log_dir_override

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

    demo_manifest = demo_pool = None
    # Only a repair-enabled plateau run ever reads the demonstration manifest; the paired control never touches it.
    repair_enabled = cfg.plateau_repair is not None and str(cfg.plateau_repair.get("repair")).lower() == "enabled"
    if cfg.task_manifests is not None:
        # R-037: manifests for any task, from the config's pool seeds; no train split when nothing is fitted.
        task_manifests = load_task_manifests(cfg)
        dev_manifest_full, dev_pool = task_manifests["development"], task_manifests["development"].resolve_pool()
        if repair_enabled:
            demo_manifest = task_manifests["demonstration"]
            demo_pool = demo_manifest.resolve_pool()
        train_manifest = train_pool = None
    else:
        train_manifest, train_pool = load_split(cfg, SPLIT_TRAIN)
        dev_manifest_full, dev_pool = load_split(cfg, SPLIT_DEVELOPMENT)
    promotion = PromotionPolicy(
        screen_episodes=cfg.screen_episodes,
        expanded_episodes=cfg.expanded_episodes,
    )
    # The orchestrator carves the screen (first cases) and the disjoint promotion set
    # (the next `expanded_episodes`) out of this. The checkpoint probe scores the tail,
    # so screening, checkpoint selection and promotion evidence share no case.
    evaluated_cases = cfg.screen_episodes + cfg.expanded_episodes
    if evaluated_cases + cfg.probe_episodes > len(dev_manifest_full):
        raise ValueError(
            f"development manifest has {len(dev_manifest_full)} cases; a {cfg.screen_episodes}-case screen, "
            f"a {cfg.expanded_episodes}-case promotion set and {cfg.probe_episodes} probe cases would overlap"
        )
    dev_manifest = dev_manifest_full.head(evaluated_cases)
    probe_manifest = dev_manifest_full.tail(cfg.probe_episodes)

    # One env per stream. Collection, evaluation, probing and recording must not
    # be able to disturb each other's placements.
    collect_env = make_env(cfg, train_pool) if train_manifest is not None else None
    rl_env = make_env(cfg, train_pool) if train_manifest is not None else None
    demo_env = make_env(cfg, demo_pool) if demo_manifest is not None else None
    dev_env = make_env(cfg, dev_pool)
    gif_env = make_env(cfg, dev_pool)
    probe_env = make_env(cfg, dev_pool)
    envs = PipelineEnvs(development=dev_env, rl=rl_env, gif=gif_env, probe=probe_env)

    obs_dim, action_dim = policy_space_dims(dev_env)
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
    if train_manifest is not None:
        print(f"  train manifest      : {train_manifest.manifest_id} ({len(train_manifest)} cases)")
    if demo_manifest is not None:
        print(f"  demonstration manifest: {demo_manifest.manifest_id} ({len(demo_manifest)} cases, disjoint placements)")
    print(f"  development manifest: {dev_manifest.manifest_id} ({len(dev_manifest)} cases)")
    print(f"  probe manifest      : {probe_manifest.manifest_id} ({len(probe_manifest)} cases, disjoint)")
    print(f"  development budget  : screen {promotion.screen_episodes}, "
          f"promotion set {promotion.expanded_episodes} (disjoint from the screen and the probe)")
    print(f"  metaworld           : {dev_manifest.environment.version_or_commit}")
    print(f"  success rule        : {dev_manifest.environment.success_rule}")

    # -----------------------------------------------------------------------
    #  Expert baseline
    # -----------------------------------------------------------------------
    separator("Expert Baseline")
    expert_result = None
    if cfg.expert_baseline:
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
    else:
        print("  skipped (expert_baseline: false): no expert episode on the development cases")

    # -----------------------------------------------------------------------
    #  Stage 1: Expert dataset generation (or load cached buffer)
    # -----------------------------------------------------------------------
    separator("Stage 1: Expert Dataset Generation")
    demo_buffer_path: str = getattr(cfg, "demo_buffer_path", "")
    if train_manifest is None:
        demo_buf = None
        print("  skipped: nothing is fitted, so no demonstration buffer is collected")
    elif demo_buffer_path and os.path.isfile(demo_buffer_path):
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
        reasoning_effort=cfg.reasoning_effort,
        policy_gen_two_phase=getattr(cfg, "policy_gen_two_phase", False),
        policy_impl_mode=getattr(cfg, "policy_impl_mode", "batched"),
    )
    print(f"\n  LLM model      : {cfg.model} (reasoning effort {cfg.reasoning_effort})")
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

    lane_plan = None
    if cfg.plateau_repair is not None:
        from lares.search.lanes import expected_plateau_budget, load_plateau_repair_plan

        lane_plan = load_plateau_repair_plan(cfg.plateau_repair)
        budget = expected_plateau_budget(lane_plan, cfg.screen_episodes, cfg.expanded_episodes, cfg.probe_episodes,
                                         cfg.eval_max_steps, cfg.max_repair_per_candidate)
        print(f"  Plateau repair: {lane_plan.n} structure slots, repair {lane_plan.to_dict()['repair']}; "
              f"<= {budget['episodes_max']} episodes, <= {budget['llm']['max_calls']} LLM calls")
    elif cfg.two_lane_search is not None:
        from lares.search.lanes import expected_budget, load_two_lane_plan

        lane_plan = load_two_lane_plan(cfg.two_lane_search)
        print(f"  Two-lane search: {len(lane_plan.fittable_slots)} fittable mutations + "
              f"{len(lane_plan.exploration_slots)} unseeded explorations; fittable checkpoints "
              f"{list(lane_plan.checkpoint_steps)} on the probe manifest")
        for spec in lane_plan.fittable_slots:
            print(f"    slot {spec.slot}: {spec.reference.structure_id} (scope {spec.scope.scope_id})")
        budget = expected_budget(lane_plan, cfg.screen_episodes, cfg.expanded_episodes, cfg.probe_episodes,
                                 cfg.eval_max_steps, cfg.max_repair_per_candidate)
        print(f"  Expected budget: <= {budget['total']['episodes_max']} episodes, "
              f"<= {budget['total']['simulator_steps_max']} simulator steps, "
              f"{budget['total']['bc_gradient_steps']} gradient steps, ~{budget['llm']['expected_calls']} LLM calls "
              f"(max {budget['llm']['max_calls']})")

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
        train_cases=train_manifest.episodes if train_manifest is not None else (),
        eval_max_steps=cfg.eval_max_steps,
        promotion_policy=promotion,
        rollout_checkpoint_probes=cfg.rollout_checkpoint_probes,
        objective=cfg.objective,
        checkpoint_steps=cfg.checkpoint_steps,
        fixed_action_scale=cfg.fixed_action_scale,
        probe_manifest=probe_manifest,
        max_repair_per_candidate=cfg.max_repair_per_candidate,
        max_llm_calls_per_generation=cfg.max_llm_calls_per_generation,
        param_budget=tuple(cfg.param_budget),
        max_llm_tokens_per_generation=cfg.max_llm_tokens_per_generation,
        promotion_challengers=cfg.promotion_challengers,
        lane_plan=lane_plan,
        fit_seed=int(getattr(cfg, "fit_seed", 0)),
        demonstration_manifest=demo_manifest,
        demonstration_env=demo_env,
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
        if expert_result is not None:
            print_eval("MetaWorld built-in expert", expert_result)
        print_eval("LLM-evolved policy (best)", best["eval"])
    else:
        print("\n  WARNING: Evolution produced no valid policies.")

    print()
    for env in (collect_env, demo_env):
        if env is not None:
            env.close()
    envs.close()


if __name__ == "__main__":
    main()
