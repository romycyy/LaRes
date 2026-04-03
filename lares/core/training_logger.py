"""
Lightweight append-only structured logging for the multi-stage training pipeline.

Logs metrics to JSONL (one JSON object per line) for machine readability.
Optional integration with Weights & Biases or TensorBoard if provided.
"""

import json

# ---------------------------------------------------------------------------
#  Canonical metric names (shared by pipeline and plot_training_dynamics)
# ---------------------------------------------------------------------------

# BC stage
BC_TRAIN_LOSS = "bc/train_loss"
BC_MEAN_LOSS = "bc/mean_loss"
BC_STD_LOSS = "bc/std_loss"
BC_GRAD_NORM_PRE_CLIP = "bc/grad_norm_pre_clip"
BC_GRAD_NORM_POST_CLIP = "bc/grad_norm_post_clip"

# RL stage
RL_TOTAL_LOSS = "rl/total_loss"
RL_POLICY_LOSS = "rl/policy_loss"
RL_ENTROPY = "rl/entropy"
RL_KL = "rl/kl"
RL_ENTROPY_BONUS = "rl/entropy_bonus"
RL_KL_PENALTY = "rl/kl_penalty"
RL_GRAD_NORM_PRE_CLIP = "rl/grad_norm_pre_clip"
RL_GRAD_NORM_POST_CLIP = "rl/grad_norm_post_clip"
RL_LEARNING_RATE = "rl/learning_rate"
RL_RETURN_MEAN = "rl/return_mean"
RL_RETURN_STD = "rl/return_std"
RL_REWARD_MEAN = "rl/reward_mean"
RL_REWARD_STD = "rl/reward_std"
RL_ADVANTAGE_MEAN = "rl/advantage_mean"
RL_ADVANTAGE_STD = "rl/advantage_std"
RL_SUCCESS_RATE = "rl/success_rate"

# Evolutionary stage
EVO_FITNESS_MEAN = "evo/fitness_mean"
EVO_FITNESS_BEST = "evo/fitness_best"
EVO_FITNESS_MEDIAN = "evo/fitness_median"
EVO_FITNESS_WORST = "evo/fitness_worst"
EVO_FITNESS_ELITE_MEAN = "evo/fitness_elite_mean"
import os
import time
from typing import Any, Optional


class TrainingLogger:
    """Unified structured metric logger for BC, RL, and evolution stages.

    Every logged record includes: stage, task_name, global_step, update,
    wall_clock_time, metric_name, metric_value. Logs are append-only JSONL.
    """

    def __init__(
        self,
        log_dir: str = "./logs/training_dynamics",
        run_id: Optional[str] = None,
        task_name: Optional[str] = None,
        wandb_run: Optional[Any] = None,
        tensorboard_writer: Optional[Any] = None,
    ):
        self.log_dir = log_dir
        self.run_id = run_id or time.strftime("%Y%m%d_%H%M%S")
        self.task_name = task_name or ""
        self.global_step = 0
        self.start_time = time.time()
        self.wandb_run = wandb_run
        self.tensorboard_writer = tensorboard_writer

        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{self.run_id}.jsonl")

    def log(
        self,
        stage: str,
        metric_name: str,
        metric_value: float,
        update: Optional[int] = None,
        epoch: Optional[int] = None,
        task_name: Optional[str] = None,
    ) -> None:
        """Append a single metric record. Skips invalid values (None, non-numeric)."""
        wall_time = time.time() - self.start_time
        task = task_name if task_name is not None else self.task_name

        try:
            val = float(metric_value)
        except (TypeError, ValueError):
            return  # Skip invalid values; fail gracefully

        record = {
            "stage": stage,
            "task_name": task,
            "global_step": self.global_step,
            "update": update,
            "epoch": epoch,
            "wall_clock_time": wall_time,
            "metric_name": metric_name,
            "metric_value": val,
        }
        # Omit None for cleaner output
        record = {k: v for k, v in record.items() if v is not None}

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Mirror to wandb / tensorboard if configured
        if self.wandb_run is not None:
            try:
                self.wandb_run.log(
                    {metric_name: metric_value, "global_step": self.global_step}
                )
            except Exception:
                pass
        if self.tensorboard_writer is not None:
            try:
                self.tensorboard_writer.add_scalar(
                    metric_name.replace("/", "_"), metric_value, self.global_step
                )
            except Exception:
                pass

    def log_metrics(
        self,
        stage: str,
        update: int,
        metrics: dict[str, float],
        epoch: Optional[int] = None,
        task_name: Optional[str] = None,
    ) -> None:
        """Log multiple metrics for one update, then advance global_step.

        Invalid or missing values in metrics dict are skipped without raising.
        """
        for name, value in metrics.items():
            self.log(
                stage=stage,
                metric_name=name,
                metric_value=value,
                update=update,
                epoch=epoch,
                task_name=task_name,
            )
        self.global_step += 1

    def advance_step(self, n: int = 1) -> None:
        """Advance global_step without logging (e.g. when skipping log_freq)."""
        self.global_step += n
