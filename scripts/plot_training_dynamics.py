#!/usr/bin/env python
"""
Standalone plotting utility for training pipeline logs.

Reads JSONL logs from the training pipeline and produces figures for BC, RL,
and evolutionary stages. BC and RL metrics are written **one PNG per training
process** (candidate inner loop) so you can inspect stability per run.

Output layout::

    <output_dir>/
      bc_train_loss/
        push-v2__run_20250325_120000__proc0.png
        ...
      rl_total_loss/
        ...

Metric folder names are derived from the canonical metric keys (slashes
replaced with underscores).

Usage (run from LaRes project root):
    # Single log file (recommended for one run)
    python scripts/plot_training_dynamics.py --log-path ./logs/training_dynamics/20250122_123456.jsonl

    # Directory: finds all *.jsonl and merges into one plot
    python scripts/plot_training_dynamics.py --log-path ./logs/training_dynamics --output-dir ./figures

    # Custom output directory
    python scripts/plot_training_dynamics.py --log-path ./logs/demo_evolution/run.jsonl --output-dir ./my_figures
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add project root to path so lares package is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import matplotlib

from lares.core.training_logger import (
    BC_GRAD_NORM_PRE_CLIP,
    BC_MEAN_LOSS,
    BC_STD_LOSS,
    BC_TRAIN_LOSS,
    EVO_FITNESS_BEST,
    EVO_FITNESS_ELITE_MEAN,
    EVO_FITNESS_MEAN,
    EVO_FITNESS_MEDIAN,
    RL_ENTROPY,
    RL_GRAD_NORM_PRE_CLIP,
    RL_KL,
    RL_POLICY_LOSS,
    RL_RETURN_MEAN,
    RL_SUCCESS_RATE,
    RL_TOTAL_LOSS,
)

matplotlib.use("Agg")  # Non-interactive backend for servers
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  Task difficulty ordering (easy -> hard) for multi-task plot legends
# ---------------------------------------------------------------------------

TASK_DIFFICULTY_ORDER = [
    "reach-v2",
    "push-v2",
    "window-close-v2",
    "window-open-v2",
    "button-press-v2",
    "drawer-open-v2",
    "door-close-v2",
    "door-open-v2",
    "lever-pull-v2",
    "faucet-open-v2",
    "faucet-close-v2",
    "handle-press-v2",
    "handle-pull-v2",
    "button-press-topdown-v2",
    "pick-place-v2",
    "assembly-v2",
    "hammer-v2",
    "peg-insert-side-v2",
    "peg-unplug-side-v2",
    "coffee-pull-v2",
    "coffee-push-v2",
    "soccer-v2",
    "basketball-v2",
    "dial-turn-v2",
    "sweep-v2",
    "shelf-place-v2",
    "coffee-button-v2",
]

# Metrics to plot per stage (metric_key, display_label); keys from training_logger
BC_METRICS = [
    (BC_TRAIN_LOSS, "Train loss"),
    (BC_MEAN_LOSS, "MSE (mean)"),
    (BC_STD_LOSS, "Std penalty"),
    (BC_GRAD_NORM_PRE_CLIP, "Grad norm (pre-clip)"),
]

RL_METRICS = [
    (RL_TOTAL_LOSS, "Total loss"),
    (RL_POLICY_LOSS, "Policy loss"),
    (RL_ENTROPY, "Entropy"),
    (RL_KL, "KL (to BC)"),
    (RL_GRAD_NORM_PRE_CLIP, "Grad norm (pre-clip)"),
    (RL_RETURN_MEAN, "Return mean"),
    (RL_SUCCESS_RATE, "Success rate"),
]

EVO_METRICS = [
    (EVO_FITNESS_MEAN, "Fitness mean"),
    (EVO_FITNESS_BEST, "Fitness best"),
    (EVO_FITNESS_MEDIAN, "Fitness median"),
    (EVO_FITNESS_ELITE_MEAN, "Elite mean"),
]


# ---------------------------------------------------------------------------
#  Log loading
# ---------------------------------------------------------------------------


def collect_log_paths(path: str) -> list[Path]:
    """Resolve path to one or more JSONL files. Accepts file or directory."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if p.is_file():
        return [p] if p.suffix == ".jsonl" else []
    return sorted(p.glob("*.jsonl"))


def load_logs(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and parse JSONL log records. Skips malformed lines without raising."""
    records = []
    for p in paths:
        with open(p, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    # TrainingLogger records do not include run_id/candidate_id, so
                    # we tag each JSONL line with its source file for plotting.
                    rec["__run_id__"] = p.stem
                    records.append(rec)
                except json.JSONDecodeError:
                    continue
    return records


def order_tasks(tasks: set[str]) -> list[str]:
    """Order tasks from easy to hard; unknown tasks appended at end."""
    ordered = [t for t in TASK_DIFFICULTY_ORDER if t in tasks]
    remaining = sorted(tasks - set(ordered))
    return ordered + remaining


def group_by_stage_metric_task(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, dict[tuple[str, int], list[tuple[float, float]]]]]]:
    """
    Group records into:
      stage -> metric_name -> task_name -> (run_id, process_id) -> [(x, y), ...]

    Where:
      - x is `update` (BC step / RL iteration / generation), not `global_step`.
      - process_id is inferred from `update` resets within (run_id, stage, task)
        when a new candidate RL/BC inner loop starts.
    """
    grouped: dict[str, dict[str, dict[str, dict[tuple[str, int], list[tuple[float, float]]]]]] = {
        "bc": {},
        "rl": {},
        "evolutionary": {},
    }

    def _norm_stage(stage: str) -> str | None:
        if stage in {"evolutionary", "evo"}:
            return "evolutionary"
        if stage in {"bc", "rl"}:
            return stage
        return None

    # First pass: collect indices by (run_id, norm_stage, task_name) for
    # process_id inference from `update` resets.
    by_process_group: dict[tuple[str, str, str], list[int]] = {}
    for idx, r in enumerate(records):
        norm_stage = _norm_stage(str(r.get("stage", "")))
        if norm_stage is None:
            continue
        metric = r.get("metric_name", "")
        if not metric:
            continue
        task = r.get("task_name") or "unknown"
        run_id = r.get("__run_id__") or "unknown"
        by_process_group.setdefault((run_id, norm_stage, task), []).append(idx)

    # Second pass: infer process_id for each record index.
    process_id_by_idx: dict[int, int] = {}
    for (run_id, norm_stage, task), idxs in by_process_group.items():
        # Within a single JSONL file, `global_step` is monotonic, so sorting
        # by it gives us the order of BC/RL inner-loop executions.
        def _sort_key(i: int) -> float:
            gs = records[i].get("global_step")
            if gs is None:
                gs = records[i].get("update", 0)
            return float(gs)

        idxs_sorted = sorted(idxs, key=_sort_key)
        prev_update: float | None = None
        proc_id = 0
        for i in idxs_sorted:
            upd = records[i].get("update")
            if upd is None:
                upd = records[i].get("global_step", 0)
            upd_f = float(upd)
            if prev_update is not None and upd_f < prev_update:
                proc_id += 1
            process_id_by_idx[i] = proc_id
            prev_update = upd_f

    # Third pass: populate grouped points using:
    #   x = update (unit-correct per stage)
    for idx, r in enumerate(records):
        norm_stage = _norm_stage(str(r.get("stage", "")))
        if norm_stage is None:
            continue
        metric = r.get("metric_name", "")
        if not metric:
            continue
        task = r.get("task_name") or "unknown"
        y = r.get("metric_value")
        if y is None:
            continue

        x = r.get("update")
        if x is None:
            x = r.get("global_step", 0)

        run_id = r.get("__run_id__") or "unknown"
        proc_id = process_id_by_idx.get(idx, 0)

        x_f, y_f = float(x), float(y)
        grouped[norm_stage].setdefault(metric, {})
        grouped[norm_stage][metric].setdefault(task, {})
        grouped[norm_stage][metric][task].setdefault((str(run_id), int(proc_id)), []).append(
            (x_f, y_f)
        )

    # Sort points by x for each series
    for stage in grouped:
        for metric in grouped[stage]:
            for task in grouped[stage][metric]:
                for series_key in grouped[stage][metric][task]:
                    grouped[stage][metric][task][series_key].sort(key=lambda p: p[0])

    return grouped


# ---------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------

# Publication-friendly style
FIGURE_DPI = 150
FIGURE_SIZE = (7, 4)
def _metric_output_dirname(metric_key: str) -> str:
    """Filesystem-safe folder name under output_dir, one folder per metric."""
    return metric_key.replace("/", "_").replace(" ", "_")


def _safe_filename_stem(s: str) -> str:
    """Sanitize a string for use in PNG filenames."""
    out = []
    for ch in str(s):
        if ch.isalnum() or ch in "-_.":
            out.append(ch)
        else:
            out.append("_")
    stem = "".join(out).strip("_")
    return stem or "unknown"


def plot_stage_per_process(
    grouped: dict[str, dict[str, dict[tuple[str, int], list[tuple[float, float]]]]],
    metrics: list[tuple[str, str]],
    stage_title: str,
    output_dir: Path,
    x_label: str,
) -> None:
    """One PNG per (task, run_id, process_id) under output_dir/<metric_key>/."""
    for metric_key, metric_label in metrics:
        if metric_key not in grouped:
            continue
        data = grouped[metric_key]
        tasks = order_tasks(set(data.keys()))
        if not tasks:
            continue

        metric_dir = output_dir / _metric_output_dirname(metric_key)
        metric_dir.mkdir(parents=True, exist_ok=True)

        for task in tasks:
            series_dict = data[task]
            if not series_dict:
                continue
            series_keys_sorted = sorted(series_dict.keys(), key=lambda k: (k[1], k[0]))
            for run_id, proc_id in series_keys_sorted:
                pts = series_dict[(run_id, proc_id)]
                if not pts:
                    continue
                xs, ys = zip(*pts)

                fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
                ax.plot(xs, ys, color="C0", linewidth=1.5)
                ax.set_xlabel(x_label)
                ax.set_ylabel(metric_label)
                ax.set_title(
                    f"{stage_title}: {metric_label}\n"
                    f"{task}  |  run {run_id}  |  process {proc_id}"
                )
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", labelsize=9)
                fig.tight_layout()

                fname = (
                    f"{_safe_filename_stem(task)}__run_{_safe_filename_stem(run_id)}"
                    f"__proc{proc_id}.png"
                )
                fig.savefig(metric_dir / fname, bbox_inches="tight", dpi=FIGURE_DPI)
                plt.close(fig)


def plot_bc(grouped: dict, output_dir: Path) -> None:
    """Behavioral cloning: one figure per process per metric."""
    bc_data = grouped.get("bc", {})
    plot_stage_per_process(
        bc_data, BC_METRICS, "Behavioral Cloning", output_dir, x_label="BC update step"
    )


def plot_rl(grouped: dict, output_dir: Path) -> None:
    """RL (GRPO): one figure per process per metric (stability per candidate)."""
    rl_data = grouped.get("rl", {})
    plot_stage_per_process(
        rl_data,
        RL_METRICS,
        "RL (GRPO)",
        output_dir,
        x_label="RL iteration (outer loop)",
    )


def plot_evolutionary(grouped: dict, output_dir: Path) -> None:
    """Evolutionary: same folder-per-metric layout (usually one process per run)."""
    evo_data = grouped.get("evolutionary", {})
    plot_stage_per_process(
        evo_data,
        EVO_METRICS,
        "Evolutionary",
        output_dir,
        x_label="Generation",
    )


def run_plots(log_paths: list[Path], output_dir: Path) -> None:
    """Load logs and generate BC, RL, and evolutionary figure groups."""
    records = load_logs(log_paths)
    if not records:
        raise ValueError(f"No records loaded from {log_paths}")

    grouped = group_by_stage_metric_task(records)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_bc(grouped, output_dir)
    plot_rl(grouped, output_dir)
    plot_evolutionary(grouped, output_dir)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot training dynamics from JSONL logs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--log-path",
        type=str,
        required=True,
        help="Path to JSONL log file or directory containing *.jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save figures (default: same as log file or cwd)",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point: parse args, load logs, save figures."""
    args = parse_args()
    paths = collect_log_paths(args.log_path)
    if not paths:
        raise SystemExit(f"No JSONL files found at {args.log_path}")

    if args.output_dir is not None:
        out = Path(args.output_dir)
    else:
        out = paths[0].parent / "figures"
    run_plots(paths, out)
    print(f"Figures saved to {out}")


if __name__ == "__main__":
    main()
