"""Render pen-spin rollouts to GIFs, for looking at a controller (not a protocol; nothing here is evidence).

One rollout per process (D-034). The controller runs on a whole frozen manifest in one batched env, exactly as
``protocols.run_cell`` runs it, and the viewport camera follows the environments named by ``--env-index``. Before
each step the camera is moved to each of those environments in turn and a frame is taken, so one process yields one
GIF per named environment. Because the batch and the manifest are the ones a protocol scored, the rollout can be
checked against that protocol's archived cell (``--archived``): the script compares every case's ``sustained``
outcome and per-step trace with the archive and prints whether rendering changed the physics.

Run under Isaac's Python on the box::

    source ~/IsaacLab/source_env.sh && PYTHONPATH=~/np126_overlay python scripts/isaac/record_pen_spin_gif.py \
        --controller incumbent --manifest logs/pen_spin/r081_stroke_feedback/fresh.yaml \
        --archived logs/pen_spin/r081_stroke_feedback/constant_gain_fresh.json \
        --env-index 0 1 2 --out-dir logs/pen_spin/gifs/incumbent

Controllers: ``incumbent`` (the D-058 fixed-stroke incumbent, rebuilt from the R-081 protocol block and checked
against ``config/pen_spin/d058_fixed_stroke_incumbent/incumbent.json``) and ``ppo_expert`` (the D-036 checkpoint).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from lares.utils.cli import load_yaml  # noqa: E402
from lares.utils.hashing import sha256_text  # noqa: E402

CONFIG = os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml")
INCUMBENT = os.path.join(ROOT, "config", "pen_spin", "d058_fixed_stroke_incumbent")


def require_numpy1() -> None:
    """Isaac Sim 5.1's OmniGraph bindings are built against numpy 1; the box's Isaac venv has numpy 2.4.3, under which
    any camera annotator fails ("Unable to write from unknown dtype"). Run with a numpy 1.26 overlay on PYTHONPATH
    (``pip install --target ~/np126_overlay numpy==1.26.4``) rather than changing the shared venv."""
    if int(np.__version__.split(".")[0]) >= 2:
        raise SystemExit(f"numpy {np.__version__}: camera annotators need numpy 1.x; prefix PYTHONPATH=~/np126_overlay")


def build_controller(name: str, cfg: dict):
    protocols = cfg["research_protocols"]
    if name == "incumbent":
        from lares.envs.isaac_pen_spin.stroke_feedback import StrokeFeedbackController, clamp_source, substituted_source

        pcfg = protocols["r081_stroke_feedback"]
        with open(os.path.join(INCUMBENT, "incumbent.json"), encoding="utf-8") as f:
            inc = json.load(f)
        src = substituted_source(clamp_source(pcfg))
        got = sha256_text(src)
        if got != inc["files"]["evaluated_source.py"]:
            raise SystemExit(f"rebuilt incumbent source {got} != frozen evaluated_source.py {inc['files']['evaluated_source.py']}")
        return StrokeFeedbackController(pcfg, "constant", "d058_incumbent", float(inc["constant"]))
    if name == "ppo_expert":
        from lares.envs.isaac_pen_spin.feedback_ablation import check_checkpoint
        from lares.envs.isaac_pen_spin.ppo_expert import PPOController

        pcfg = protocols["feedback_ablation"]
        return PPOController(check_checkpoint(pcfg), int(pcfg["training_seed"]))
    raise SystemExit(f"unknown controller {name!r}")


class FrameGrabber:
    """Wraps a controller: before each action, point the viewport at each watched env and take a frame."""

    def __init__(self, inner, env, indices, eye_offset, target_offset, warmup: int = 30):
        self.inner, self.env, self.indices = inner, env, list(indices)
        self.name = getattr(inner, "name", "controller")
        self.eye_offset = np.asarray(eye_offset, dtype=float)
        self.target_offset = np.asarray(target_offset, dtype=float)
        self.frames = {i: [] for i in self.indices}
        self.warmup = warmup
        self.origins = env.scene.env_origins.detach().cpu().numpy()

    @property
    def params(self):
        return getattr(self.inner, "params", None)

    def attach(self, env):
        if hasattr(self.inner, "attach"):
            self.inner.attach(env)

    def reset(self, ctx):
        if hasattr(self.inner, "reset"):
            self.inner.reset(ctx)

    def _frame(self, i):
        o = self.origins[i]
        self.env.sim.set_camera_view(eye=(o + self.eye_offset).tolist(), target=(o + self.target_offset).tolist())
        frame = None
        for _ in range(4):  # the RTX renderer lags camera moves by a frame or two; re-render until it settles
            self.env.sim.render()
            frame = self.env.render()
        return frame

    def __call__(self, obs, t, ctx):
        if t == 0:
            for _ in range(self.warmup):  # prime the renderer: its first frames are blank
                self._frame(self.indices[0])
        for i in self.indices:
            self.frames[i].append(np.asarray(self._frame(i))[..., :3].copy())
        return self.inner(obs, t, ctx)


def annotate(frames, trace, i, label, stage_spec, stride):
    from PIL import Image, ImageDraw

    out = []
    n = int(trace["length"][i])
    hit = 0
    roll = 0.0  # the pen's roll angle about its own axis, integrated from the recorded axial rate (60 Hz)
    for t, frame in enumerate(frames):
        if t >= n:
            break
        spin = float(trace["spin_rate"][t, i]) if t < trace["spin_rate"].shape[0] else float("nan")
        cd = float(trace["center_dist"][t, i])
        ok = bool(trace["pred"][t, i])
        hit += ok
        if np.isfinite(spin):
            roll += spin / 60.0
        if t % stride:
            continue
        img = Image.fromarray(frame.astype(np.uint8))
        d = ImageDraw.Draw(img)
        lines = [label, f"t={t / 60.0:4.2f}s  step {t:3d}/{trace['max_episode_length']}",
                 f"axial spin {spin:+6.2f} rad/s (need |.|>{stage_spec.success_spin_rate:g})",
                 f"centre dist {cd:.3f} m (need <{stage_spec.success_center_tolerance:g})",
                 f"step success {'YES' if ok else 'no '}   hits so far {hit}"]
        d.rectangle([0, 0, img.width, 16 * len(lines) + 6], fill=(0, 0, 0))
        for k, s in enumerate(lines):
            d.text((6, 4 + 16 * k), s, fill=(120, 255, 120) if (k == 4 and ok) else (255, 255, 255))
        # Roll dial: a capsule rolling about its own axis looks still on camera, so draw the integrated roll angle.
        cx, cy, r = img.width - 44, 44, 34
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=2)
        d.line([cx, cy, cx + r * np.cos(roll), cy - r * np.sin(roll)], fill=(255, 140, 30), width=4)
        d.text((cx - 22, cy + r + 2), "pen roll", fill=(255, 255, 255))
        out.append(np.asarray(img))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controller", required=True, choices=["incumbent", "ppo_expert"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--archived", default=None, help="the protocol cell (.json, with its .npz) this batch reproduces")
    ap.add_argument("--env-index", type=int, nargs="+", default=[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--stride", type=int, default=3, help="keep every n-th 60 Hz step (3 -> 20 fps is real time)")
    ap.add_argument("--resolution", type=int, nargs=2, default=[640, 480])
    ap.add_argument("--eye", type=float, nargs=3, default=[0.30, -0.62, 0.78], help="camera offset from env origin")
    ap.add_argument("--target", type=float, nargs=3, default=[0.0, -0.36, 0.58])
    args = ap.parse_args()

    require_numpy1()
    cfg = load_yaml(CONFIG)
    from lares.envs.isaac_lab_adapter import _ensure_isaaclab_on_path, close_isaac_app, launch_isaac_app

    _ensure_isaaclab_on_path(str(cfg["isaaclab_root"]))
    launch_isaac_app(headless=True, device=str(cfg["device"]), enable_cameras=True)
    try:
        import imageio

        from lares.envs.isaac_pen_spin import cases, runner as rmod
        from lares.eval.manifest import EvaluationManifest

        manifest = EvaluationManifest.load(args.manifest)
        stage = cases.stage_of(manifest)
        ctrl = build_controller(args.controller, cfg)

        orig_make = rmod.make_env

        def make_env_rendered(stage_, num_envs, device="cuda:0", enhanced_determinism=False):
            import importlib

            cfg_mod = importlib.import_module("lares.envs.isaac_pen_spin.shadow_hand_spin_env_cfg")
            ecfg = getattr(cfg_mod, cases.STAGES[stage_].cfg_class)()
            ecfg.scene.num_envs = int(num_envs)
            ecfg.sim.device = device
            ecfg.sim.physx.enable_enhanced_determinism = bool(enhanced_determinism)
            ecfg.viewer.resolution = tuple(args.resolution)
            # Visual only: an orange pen, so it shows against the grey hand. No physics property changes; the
            # archive check below confirms the rollout is unchanged.
            import isaaclab.sim as sim_utils

            ecfg.object_cfg.spawn.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.05))
            return rmod.MeasuredShadowHandSpinEnv(cfg=ecfg, render_mode="rgb_array")

        rmod.make_env = make_env_rendered
        run = rmod.PenSpinBatchRunner(stage, len(manifest), device=str(cfg["device"]))
        rmod.make_env = orig_make
        if hasattr(ctrl, "attach"):
            ctrl.attach(run.env)
        grab = FrameGrabber(ctrl, run.env, args.env_index, args.eye, args.target)
        try:
            trace = run.run(manifest, grab)
        finally:
            run.close()

        episodes = cases.reduce_episodes(trace)
        os.makedirs(args.out_dir, exist_ok=True)
        check = None
        if args.archived:
            with open(args.archived, encoding="utf-8") as f:
                arch = json.load(f)["episodes"]
            arch_npz = np.load(args.archived.replace(".json", ".npz"))
            same_outcome = [bool(a["sustained"]) == bool(b["sustained"]) for a, b in zip(arch, episodes)]
            T = min(arch_npz["spin_rate"].shape[0], trace["spin_rate"].shape[0])
            a_s, b_s = arch_npz["spin_rate"][:T], trace["spin_rate"][:T]
            both = ~np.isnan(a_s) & ~np.isnan(b_s)
            check = {"cases": len(arch), "same_sustained": int(sum(same_outcome)),
                     "archived_successes": int(sum(bool(a["sustained"]) for a in arch)),
                     "rendered_successes": int(sum(bool(b["sustained"]) for b in episodes)),
                     "max_abs_spin_diff": float(np.max(np.abs(a_s[both] - b_s[both]))) if both.any() else None,
                     "bit_identical_spin": bool(np.array_equal(np.nan_to_num(a_s, nan=-999), np.nan_to_num(b_s, nan=-999)))}
            print("archive check:", json.dumps(check), flush=True)

        spec = cases.STAGES[stage]
        written = []
        for i in args.env_index:
            e = episodes[i]
            verdict = ("SUSTAINED SUCCESS" if e["sustained"] else
                       "DROPPED" if not e["completed"] else "FAILED (not sustained)")
            label = f"{args.controller}  {e['case_id']}  -> {verdict}"
            frames = annotate(grab.frames[i], trace, i, label, spec, args.stride)
            path = os.path.join(args.out_dir, f"{args.controller}_{e['case_id']}.gif")
            imageio.mimsave(path, frames, duration=1000.0 / args.fps, loop=0)
            blank = float(np.mean([f.std() < 1.0 for f in grab.frames[i]])) if grab.frames[i] else 1.0
            written.append({"path": path, "env_index": i, "frames": len(frames), "blank_fraction": blank, **e})
            print("wrote", path, f"({len(frames)} frames, blank {blank:.2f})", flush=True)
        with open(os.path.join(args.out_dir, f"{args.controller}_summary.json"), "w", encoding="utf-8") as f:
            json.dump({"controller": args.controller, "manifest": args.manifest, "archive_check": check,
                       "gifs": written}, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    except BaseException:
        import traceback

        traceback.print_exc()  # SimulationApp.close() exits the process before Python would print it
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        close_isaac_app()


if __name__ == "__main__":
    main()
