#!/usr/bin/env python
"""Pen-spin studies and research protocols on Isaac Lab ShadowHandSpin (``--study`` / ``--research-protocol``)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.envs.isaac_lab_adapter import _ensure_isaaclab_on_path, close_isaac_app, launch_isaac_app  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402


def load_config(path: str) -> SimpleNamespace:
    return SimpleNamespace(**(load_yaml(path) or {}))


#: ``--study`` registry: study name (the key of its ``research_protocols`` block and its module's ``STUDY``
#: constant) to the ``lares.envs.isaac_pen_spin`` module that implements it. Every module exposes
#: ``build_manifests(protocol_cfg)``, ``make_controller(name, protocol_cfg, out_dir)`` and
#: ``run(protocol_cfg, config_path, out_dir)``; a new study is one module with those three names and one entry here.
#: Modules are resolved lazily, so ``--help`` imports none of them. One module may serve two names
#: (``clamp_qualification``: R-077 and R-080); ``run_study`` tells it which by ``protocol_cfg["study"]``.
PEN_SPIN_STUDIES = {
    "feedback_ablation": "feedback_ablation",                      # R-041
    "feedback_lesions": "feedback_lesions",                        # R-044
    "feedback_delay": "feedback_delay",                            # R-047
    "learner_recovery": "learner_recovery",                        # R-051
    "learner_axial_sign": "axial_sign",                            # R-053
    "r058_scalar_cma_vs_random": "scalar_tuning",                  # R-058
    "r060_archive_rescreen": "archive_rescreen",                   # R-060
    "r062_stroke_lesion": "stroke_lesion",                         # R-062
    "r065_channel_lesions": "channel_lesions",                     # R-065
    "r068_capture_curl_confirmation": "curl_confirmation",         # R-068
    "r073_constant_preload": "constant_preload",                   # R-073
    "r075_curl_bandwidth": "curl_bandwidth",                       # R-075
    "r077_clamp_qualification": "clamp_qualification",             # R-077
    "r080_zero_curl_qualification_90": "clamp_qualification",      # R-080
    "r081_stroke_feedback": "stroke_feedback",                     # R-081
    "r064_lead_feedback": "lead_feedback",                         # R-064
    "r086_phase_warp": "phase_warp",                               # R-086
    "r089_warp_qualification_90": "warp_qualification",            # R-089
    "contact_decoder_calibration": "contact_gate",                 # R-093 (declared calibration, seed 4721)
    "contact_decoder_gate": "contact_gate",                        # R-093 (the gate, seed 4720)
    "contact_settled_calibration": "contact_settled_gate",         # R-095 (declared calibration, seed 4723)
    "contact_settled_gate": "contact_settled_gate",                # R-095 (the gate, seed 4722)
    "contact_settled_gate_80": "contact_settled_gate",             # R-096 (80-case replication, seed 4724)
    "rolling_baseline": "rolling_decoder_gate",                    # R-097 S1 (the incumbent alone, seed 4741)
    "rolling_decoder_calibration": "rolling_decoder_gate",         # R-097 (declared calibration, seed 4740)
    "rolling_decoder_gate": "rolling_decoder_gate",                # R-097 S2 (the gate, seed 4741)
    "incumbent_residual": "incumbent_residual",                    # R-099 S3 (24-draw probe search, seed 4742)
    "incumbent_residual_fresh": "incumbent_residual",              # R-099 S4 (the frozen winner on 90 fresh cases, 4744)
    "transfer_audit": "transfer_audit",                            # R-101 (ff + rf circumferential transfer, seed 4750)
    "incumbent_residual_confirmation": "incumbent_residual",       # R-102 (final 672-pair confirmation, 4751-4758)
    "symbolic_reward_residual": "symbolic_reward",                 # R-103 (draws vs ARS vs PPO on one symbolic residual)
    "symbolic_reward_alignment_audit": "symbolic_reward_audit",    # R-104 (archive-only score/success alignment, no Isaac)
    "symbolic_exact_success_ars": "success_ars",                   # R-106 (ARS on the exact sustained-success count)
    "pen_spin_objective_attribution": "success_ars",               # R-108 (exact-success vs dense-reward ARS, matched)
    "slow_mode_archive_audit": "slow_mode_audit",                  # R-110 (archive-only: does reset geometry predict the slow mode?)
    "compact_memory_matched": "memory_study",                      # R-112 (P3: instant vs filtered vs slow-memory gate)
    "recovery_authority": "recovery_authority",                    # R-114 (scripted re-capture / stroke / residual probes on stuck episodes)
    "symbolic_recovery_ars": "recovery_module",                    # R-116 (learned re-capture module on the R-106 finalist)
    "recovery_retry_vs_continue": "recovery_retry",                # R-118 (second re-capture attempt vs matched continuation)
    "recovery_tail_archive_audit": "recovery_tail_audit",          # R-120 (archive-only: which failures continued training removed)
    "recovery_joint_matched": "recovery_joint",                    # R-121 (joint residual + module adaptation vs module-only, matched)
}


def study_module(name: str):
    """The study module registered under ``name`` (imported on first use)."""
    if name not in PEN_SPIN_STUDIES:
        raise SystemExit(f"unknown study {name!r}; one of {sorted(PEN_SPIN_STUDIES)}")
    return importlib.import_module(f"lares.envs.isaac_pen_spin.{PEN_SPIN_STUDIES[name]}")


@contextmanager
def isaac_app(cfg: SimpleNamespace):
    """Isaac Lab on the path and the app launched headless for the block; closed on exit however it ends."""
    _ensure_isaaclab_on_path(str(cfg.isaaclab_root))
    launch_isaac_app(headless=True, device=str(cfg.device))
    try:
        yield
    finally:
        close_isaac_app()


def run_study(args, cfg: SimpleNamespace) -> None:
    """``--study <name>`` for any name in :data:`PEN_SPIN_STUDIES`: the driver, or one child cell
    (``--cell-manifest``), or ``--build-manifests``."""
    fa = study_module(args.study)
    protocol_cfg = (getattr(cfg, "research_protocols", None) or {}).get(args.study)
    if protocol_cfg is None:
        raise SystemExit(f"{args.config} declares no research protocol {args.study!r}")
    protocol_cfg["study"] = args.study  # the registry name the driver's child cells re-execute under
    out_dir = args.log_dir or str(protocol_cfg["log_dir"])
    if args.manifest_seed is not None and int(args.manifest_seed) != int(protocol_cfg["pool_seed"]):
        raise SystemExit(f"--manifest-seed {args.manifest_seed} != the protocol's pool_seed {protocol_cfg['pool_seed']}")
    if args.build_manifests:
        print("wrote", fa.build_manifests(protocol_cfg) or "nothing (all present and identical)")
        return
    if args.cell_manifest:
        from lares.envs.isaac_pen_spin import protocols

        with isaac_app(cfg):
            ctrl = fa.make_controller(args.controller, protocol_cfg, out_dir)
            print("wrote", protocols.run_cell(args.cell_manifest, ctrl, args.cell_out, str(cfg.device),
                                              record_obs=args.record_obs))
        return
    summary = fa.run(protocol_cfg, args.config, out_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "tape"}, indent=2))


def run_research_protocol(args, cfg: SimpleNamespace) -> None:
    from lares.envs.isaac_pen_spin import protocols

    protocol_cfg = (getattr(cfg, "research_protocols", None) or {}).get(args.research_protocol)
    if protocol_cfg is None:
        raise SystemExit(f"{args.config} declares no research protocol {args.research_protocol!r}")
    if args.research_protocol == protocols.SCRIPTED_EXPERT_QUALIFICATION:
        if args.build_manifests:
            print("wrote", protocols.build_qualification_manifests(protocol_cfg) or "nothing (all present and identical)")
            return
        if args.cell_manifest:
            with isaac_app(cfg):
                print("wrote", protocols.run_cell(args.cell_manifest, args.controller, args.cell_out,
                                                  str(cfg.device), record_obs=args.record_obs))
            return
        out_dir = args.log_dir or str(protocol_cfg["log_dir"])
        summary = protocols.run_scripted_expert_qualification(protocol_cfg, args.config, out_dir)
        print(json.dumps({k: v for k, v in summary.items() if k != "per_stage"}, indent=2))
        return
    if args.research_protocol != protocols.EPISODE_RULE_AUDIT:
        raise SystemExit(f"unknown research protocol {args.research_protocol!r}")
    if args.build_manifests:
        print("wrote", protocols.build_audit_manifests(protocol_cfg) or "nothing (all present and identical)")
        return
    out_dir = args.log_dir or str(protocol_cfg["log_dir"])
    if args.stage is not None:
        with isaac_app(cfg):
            os.makedirs(out_dir, exist_ok=True)
            print("wrote", protocols.run_audit_cell(args.stage, args.controller, protocol_cfg, out_dir, str(cfg.device)))
        return
    summary = protocols.run_episode_rule_audit(protocol_cfg, args.config, out_dir, str(cfg.device))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_stage_controller"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="ShadowHandSpin pen-spin studies and research protocols")
    parser.add_argument("--config", default=os.path.join(_PROJECT_ROOT, "config", "shadowhand_spin_stages.yaml"))
    parser.add_argument("--research-protocol", default=None,
                        help="run a named protocol from the config's research_protocols block")
    parser.add_argument("--study", default=None,
                        help="run a named study (" + ", ".join(PEN_SPIN_STUDIES) + ") from the config's "
                             "research_protocols block")
    parser.add_argument("--build-manifests", action="store_true",
                        help="with --research-protocol: write the protocol's frozen manifests and exit (no Isaac)")
    parser.add_argument("--stage", type=int, default=None, help=argparse.SUPPRESS)  # protocol child process
    parser.add_argument("--controller", default=None, help=argparse.SUPPRESS)  # protocol child process
    parser.add_argument("--cell-manifest", default=None, help=argparse.SUPPRESS)  # protocol child process
    parser.add_argument("--cell-out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--record-obs", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--log-dir", "--output-dir", dest="log_dir", default=None,
                        help="protocol output directory (default: the protocol's log_dir)")
    parser.add_argument("--manifest-seed", type=int, default=None,
                        help="with --study: the pool seed the command expects; refused if the config declares another")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.study:
        run_study(args, cfg)
        return
    if args.research_protocol:
        run_research_protocol(args, cfg)
        return
    parser.error("name a --study or a --research-protocol (the BC + GRPO smoke stages are archived in "
                 "deprecated/scripts/run_shadowhand_spin_stages.py)")


if __name__ == "__main__":
    main()
