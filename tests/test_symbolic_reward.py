"""R-103: the symbolic residual actor, the reward, the three learners' arithmetic, selection and inference (no Isaac)."""

import math
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import cases  # noqa: E402
from lares.envs.isaac_pen_spin import incumbent_residual as ir  # noqa: E402
from lares.envs.isaac_pen_spin import symbolic_reward as sr  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SL = cases.obs_slices()


def protocol_cfg() -> dict:
    return load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][sr.STUDY]


class Incumbent:
    """A stateless stand-in for the D-058 incumbent: an action that depends on the observation."""

    params = {"name": "fake"}

    def __call__(self, obs, t, ctx):
        return (0.3 * torch.tanh(obs[:, :20]) + 0.2).clamp(-1.0, 1.0)


def ctx(n: int):
    return SimpleNamespace(lower=torch.full((n, 20), -1.0), upper=torch.full((n, 20), 1.0))


def random_obs(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, cases.OBS_DIM, generator=g)
    obs[:, SL["obj_rot"]] = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1)
    return obs


NORM = {"rms_velocity": [0.5] * 20, "rms_posture": [0.1] * 20}


def controller(cfg, theta=None):
    return sr.build_controller(cfg, "t", theta, NORM, inner=Incumbent())


class TestActor(unittest.TestCase):
    def setUp(self):
        self.cfg = protocol_cfg()

    def test_layout(self):
        self.assertEqual(sr.N_COEF, 32)
        self.assertEqual(len(sr.COEF_NAMES), 32)
        self.assertEqual(self.cfg["actor"]["coefficients"], sr.N_COEF)
        self.assertEqual([sr.JOINTS[i] for i in range(20) if sr.GROUP_OF[i] == sr.GROUPS.index("LF")],
                         ["LFJ4", "LFJ3", "LFJ2", "LFJ1"])
        self.assertEqual(sorted(set(sr.GROUP_OF)), list(range(6)))

    def test_joint_order_is_the_executed_one_not_the_config_list(self):
        sim = ["robot0_" + j for j in ("WRJ1", "WRJ0", "FFJ3", "MFJ3", "RFJ3", "LFJ4", "THJ4", "FFJ2", "MFJ2", "RFJ2", "LFJ3",
                                      "THJ3", "FFJ1", "MFJ1", "RFJ1", "LFJ2", "THJ2", "FFJ0", "MFJ0", "RFJ0", "LFJ1", "THJ1",
                                      "LFJ0", "THJ0")]                               # 24 joints, simulator order
        config_list = ["robot0_" + j for j in ("WRJ1", "WRJ0", "FFJ3", "FFJ2", "FFJ1", "MFJ3", "MFJ2", "MFJ1", "RFJ3", "RFJ2",
                                              "RFJ1", "LFJ4", "LFJ3", "LFJ2", "LFJ1", "THJ4", "THJ3", "THJ2", "THJ1", "THJ0")]
        c = SimpleNamespace(joint_names=sim, actuated_dof_indices=sorted(sim.index(j) for j in config_list),
                            actuated_joint_names=config_list)
        self.assertEqual(c.actuated_dof_indices, list(ir.ACTUATED))
        self.assertEqual(sr.executed_joint_order(c), list(sr.JOINTS))
        self.assertNotEqual([j.split("_")[-1] for j in config_list], list(sr.JOINTS))

    def test_zero_theta_is_the_reference_bit_for_bit(self):
        c = controller(self.cfg)
        plain = ir.JointMotionResidual("plain", sr.reference_params(self.cfg), inner=Incumbent(),
                                       control_dt=float(self.cfg["control_dt"]))
        c.reset(ctx(6))
        plain.reset(ctx(6))
        for t in range(12):
            obs = random_obs(6, t)
            self.assertTrue(torch.equal(c(obs, t, None), plain(obs, t, None)))

    def test_nonzero_theta_changes_the_action_within_cap_and_bounds(self):
        theta = np.full(32, 0.1)                                   # exactly at the bound, as a projected nominee can be
        c = controller(self.cfg, theta)
        c.reset(ctx(5))
        obs = random_obs(5, 3)
        p = c.parts(obs, 0, None)
        mu = c.mean(p, c.theta)
        self.assertLessEqual(float((mu - p["ref"]).abs().max()), 0.10 + 1e-7)
        self.assertGreater(float((mu - p["ref"]).abs().max()), 0.0)
        with self.assertRaises(ValueError):
            controller(self.cfg, np.full(32, 0.2))

    def test_per_environment_theta_matches_single_theta(self):
        thetas = np.random.default_rng(0).uniform(-0.1, 0.1, size=(4, 32)).astype(np.float32)
        obs = random_obs(4, 9)
        c = controller(self.cfg)
        c.reset(ctx(4))
        batched = c.mean(c.parts(obs, 0, None), torch.tensor(thetas))
        for i in range(4):
            ci = controller(self.cfg)
            ci.reset(ctx(4))
            single = ci.mean(ci.parts(obs, 0, None), torch.tensor(thetas[i]))
            self.assertTrue(torch.equal(batched[i], single[i]))

    def test_filter_reset_is_per_environment(self):
        c = controller(self.cfg)
        c.reset(ctx(4))
        for t in range(5):
            c.parts(random_obs(4, t), t, None)
        before = c.ref.v_f.clone()
        mask = torch.tensor([False, True, False, True])
        c.reset_envs(mask)
        obs = random_obs(4, 99)
        c.parts(obs, 5, None)
        v = c.ref.velocity(obs)
        self.assertTrue(torch.equal(c.ref.v_f[mask], v[mask]))                       # a fresh controller's first step
        expected = before + c.ref.alpha * (v - before)
        self.assertTrue(torch.equal(c.ref.v_f[~mask], expected[~mask]))              # the others are untouched
        fresh = controller(self.cfg)
        fresh.reset(ctx(4))
        fresh.parts(obs, 0, None)
        self.assertTrue(torch.equal(fresh.ref.v_f[mask], c.ref.v_f[mask]))

    def test_axial_rate_matches_the_numpy_reducer(self):
        obs = random_obs(7, 4)
        want, _ = cases.spin_components(obs[:, SL["obj_rot"]].numpy(), obs[:, SL["obj_angvel"]].numpy() / 0.2)
        np.testing.assert_allclose(sr.axial_rate(obs)[:, 0].numpy(), want, rtol=1e-4, atol=1e-4)

    def test_theta_hash_is_of_the_float32_values(self):
        t = np.random.default_rng(1).uniform(-0.1, 0.1, 32)
        self.assertEqual(sr.theta_sha256(t), sr.theta_sha256(sr.theta_record(t)))
        self.assertEqual(sr.theta_sha256(t), sr.theta_sha256(torch.tensor(t, dtype=torch.float32)))


class TestReward(unittest.TestCase):
    def test_formula(self):
        cfg = protocol_cfg()
        ref = torch.zeros(3, 20)
        exe = torch.zeros(3, 20)
        exe[2] = 0.1
        r = sr.step_reward(torch.tensor([0.0, 0.08, 0.0]), torch.tensor([-12.0, 3.0, 6.0]),
                           torch.tensor([False, True, False]), exe, ref, cfg["reward"], 0.10)
        self.assertAlmostEqual(float(r[0]), 1.0, places=6)                            # centred, unsigned, capped
        self.assertAlmostEqual(float(r[1]), math.exp(-1.0) * 0.5 - 5.0, places=5)
        self.assertAlmostEqual(float(r[2]), 1.0 - 0.01, places=6)


class TestParameterSearch(unittest.TestCase):
    def setUp(self):
        self.cfg = protocol_cfg()

    def test_draws_are_seeded_bounded_and_scale_mixed(self):
        a = sr.draw_thetas(4760, self.cfg["draws"], 0.10)
        self.assertEqual(a.shape, (80, 32))
        np.testing.assert_array_equal(a, sr.draw_thetas(4760, self.cfg["draws"], 0.10))
        self.assertFalse(np.array_equal(a, sr.draw_thetas(4761, self.cfg["draws"], 0.10)))
        self.assertLessEqual(np.abs(a).max(), 0.10)
        self.assertLess(np.abs(a).max(axis=1).min(), 0.02)

    def test_draw_search_nominates_the_best_so_far(self):
        s = sr.DrawSearch(1, self.cfg["draws"], 0.10)
        np.testing.assert_array_equal(s.nominee(), np.zeros(32))
        s.tell(0, [1.0, 5.0, 5.0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(s.nominee(), s.thetas[1])                       # a tie keeps the earlier draw
        s.tell(1, [0, 0, 0, 0, 0, 0, 0, 4.0])
        np.testing.assert_array_equal(s.nominee(), s.thetas[1])

    def test_ars_climbs_a_quadratic_and_stays_in_bounds(self):
        target = np.full(32, 0.05)
        ars = sr.ARS(3, self.cfg["ars"], 0.10)
        d0 = np.linalg.norm(ars.theta - target)
        for g in range(40):
            cand = ars.ask(g)
            self.assertEqual(cand.shape, (16, 32))
            ars.tell(g, [-np.sum((c - target) ** 2) for c in cand])
        self.assertLess(np.linalg.norm(ars.theta - target), 0.75 * d0)
        self.assertLessEqual(np.abs(ars.theta).max(), 0.10)

    def test_ars_ignores_a_flat_return(self):
        ars = sr.ARS(3, self.cfg["ars"], 0.10)
        ars.ask(0)
        ars.tell(0, [1.0] * 16)
        np.testing.assert_array_equal(ars.theta, np.zeros(32))

    def test_one_common_budget(self):
        s = sr.schedule(self.cfg)
        self.assertEqual(s["transitions"], 460800)
        self.assertEqual(s["checkpoint_transitions"], [92160, 184320, 276480, 368640, 460800])
        bad = {**self.cfg, "ppo": {**self.cfg["ppo"], "updates": 130}}
        with self.assertRaises(SystemExit):
            sr.schedule(bad)
        self.assertEqual(sr.evaluation_allowance(self.cfg), 4608)


class TestPPO(unittest.TestCase):
    def setUp(self):
        self.cfg = protocol_cfg()

    def test_latent_logp_is_the_gaussian_density(self):
        u, mu = torch.tensor([[0.3, 1.7]]), torch.tensor([[0.1, 0.9]])
        want = torch.distributions.Normal(mu, 0.05).log_prob(u).sum(-1)
        torch.testing.assert_close(sr.gaussian_logp(u, mu, 0.05), want)

    def test_gae_handles_falls_and_time_outs(self):
        r = torch.tensor([[1.0], [1.0], [1.0]])
        v = torch.tensor([[2.0], [3.0], [4.0]])
        F, T_ = torch.tensor([[False], [False], [False]]), torch.tensor([[False], [True], [False]])
        adv, ret = sr.gae(r, v, torch.tensor([5.0]), T_, F, 0.9, 1.0)                  # a fall at step 1
        self.assertAlmostEqual(float(adv[1, 0]), 1.0 - 3.0, places=6)
        self.assertAlmostEqual(float(adv[0, 0]), (1.0 + 0.9 * 3.0 - 2.0) + 0.9 * (1.0 - 3.0), places=6)
        self.assertAlmostEqual(float(adv[2, 0]), 1.0 + 0.9 * 5.0 - 4.0, places=6)
        adv, ret = sr.gae(r, v, torch.tensor([5.0]), F, T_, 0.9, 1.0)                  # a time-out at step 1
        self.assertAlmostEqual(float(adv[1, 0]), 1.0 + 0.9 * 3.0 - 3.0, places=6)      # bootstrapped with V(s_1)
        self.assertAlmostEqual(float(ret[1, 0]), 1.0 + 0.9 * 3.0, places=6)

    def _batch(self, psi, n=256, seed=0):
        cfg = self.cfg
        c = controller(cfg)
        c.reset(ctx(n))
        p = c.parts(random_obs(n, seed), 0, None)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            mu = p["ref"] + sr.residual(psi * 0.10, p["s"], p["vhat"], p["ehat"], 0.10)
            u = mu + 0.05 * torch.randn(mu.shape, generator=g)
        return {**p, "u": u, "mu": mu, "logp": sr.gaussian_logp(u, mu, 0.05)}, g

    def test_ratio_is_one_and_every_coefficient_has_a_gradient(self):
        psi = torch.full((32,), 0.2, requires_grad=True)
        batch, g = self._batch(psi.detach())
        obj, kl, ratio = sr.surrogate(psi, batch, torch.randn(256, generator=g), self.cfg["ppo"], 0.10, 0.10)
        self.assertTrue(bool((ratio == 1.0).all()))
        self.assertEqual(float(kl), 0.0)
        obj.backward()
        self.assertTrue(bool(torch.isfinite(psi.grad).all()))
        self.assertEqual(int((psi.grad != 0).sum()), 32)

    def test_update_moves_theta_toward_higher_advantage_and_projects(self):
        cfg = {**self.cfg["ppo"], "critic_hidden": [16]}
        psi = torch.zeros(32, requires_grad=True)
        batch, g = self._batch(psi.detach(), n=512)
        # advantage rewards a positive latent deviation on joint 0, where a_0 * s is the only lever
        adv = (batch["u"][:, 0] - batch["mu"][:, 0]) / 0.05
        critic = sr.Critic(cases.OBS_DIM, cfg["critic_hidden"], cfg["value_scale"])
        roll = {**batch, "obs": random_obs(512, 0), "adv": adv, "ret": torch.zeros(512)}
        opt = torch.optim.Adam([psi], lr=cfg["lr"])
        copt = torch.optim.Adam(critic.parameters(), lr=cfg["critic_lr"])
        frozen = sr.ppo_update(psi, opt, critic, copt, roll, cfg, 0.10, 0.10, cfg["lr"], False, g)
        self.assertTrue(torch.equal(psi.detach(), torch.zeros(32)))                   # critic warm-up: actor untouched
        self.assertIsNone(frozen["kl"])
        info = sr.ppo_update(psi, opt, critic, copt, roll, cfg, 0.10, 0.10, cfg["lr"], True, g)
        self.assertGreater(float(psi[0]), 0.0)
        self.assertLessEqual(float(psi.detach().abs().max()), 1.0)
        self.assertTrue(cfg["lr_min"] <= info["lr"] <= cfg["lr_max"])

    def test_critic_normaliser_tracks_the_data(self):
        critic = sr.Critic(4, [8], 10.0)
        x = torch.randn(1000, 4) * 3.0 + 5.0
        critic.observe(x[:400])
        critic.observe(x[400:])
        torch.testing.assert_close(critic.mean, x.mean(0), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(critic.var, x.var(0, unbiased=False), atol=1e-2, rtol=1e-2)


def rows(success, completed=None, tail=None, seed=0):
    completed = completed or [True] * len(success)
    tail = tail or [1.0 if s else 0.0 for s in success]
    return [{"task_hash": f"h{seed}_{i}", "sustained": bool(s), "completed": bool(c), "tail_hit_fraction": float(t)}
            for i, (s, c, t) in enumerate(zip(success, completed, tail))]


class TestSelectionAndInference(unittest.TestCase):
    def test_selection_rule(self):
        ref = rows([1, 0, 0, 0])
        self.assertEqual(sr.select_checkpoint(ref, {1: rows([1, 0, 0, 0]), 2: rows([1, 0, 0, 0])}), 0)   # ties: the reference
        self.assertEqual(sr.select_checkpoint(ref, {1: rows([1, 1, 0, 0]), 2: rows([1, 1, 0, 0])}), 1)
        better_tail = rows([1, 0, 0, 0], tail=[1.0, 0.5, 0.0, 0.0])
        self.assertEqual(sr.select_checkpoint(ref, {1: rows([1, 0, 0, 0]), 2: better_tail}), 2)
        dropped = rows([1, 0, 0, 0], completed=[True, True, True, False])
        self.assertEqual(sr.select_checkpoint(dropped, {3: rows([1, 0, 0, 0])}), 3)                     # fewer drops

    def test_holm(self):
        h = sr.holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
        self.assertAlmostEqual(h["a"]["p_holm"], 0.004)
        self.assertAlmostEqual(h["b"]["p_holm"], 0.06)
        self.assertAlmostEqual(h["c"]["p_holm"], 0.08)
        self.assertEqual([h[k]["rejects"] for k in "abcd"], [True, False, False, False])

    def test_contrasts(self):
        rng = np.random.default_rng(0)
        data = {}
        for b in range(3):
            base = rng.random(128) < 0.35
            ppo = base | (rng.random(128) < 0.20)
            data[b] = {"reference": rows(base, seed=b), "draws": rows(base, seed=b), "ars": rows(base, seed=b),
                       "ppo": rows(ppo, seed=b)}
        c = sr.contrasts(data, 0.05, 0.05)
        self.assertEqual(set(c), {f"{a}_minus_{b}" for a, b in sr.CONTRASTS})
        head = c[sr.PRIMARY]
        self.assertEqual(head["case_paired"]["n"], 384)
        self.assertEqual(head["block_level"]["n"], 3)
        self.assertAlmostEqual(head["estimate"], head["case_paired"]["mean"])
        self.assertTrue(head["magnitude_supported"] and head["positive_effect_claim"])
        self.assertEqual(c["ars_minus_draws"]["estimate"], 0.0)
        self.assertFalse(c["ars_minus_draws"]["positive_effect_claim"])
        self.assertTrue(c["ppo_minus_reference"]["drops"]["non_inferior"])
        data[0]["ppo"] = data[0]["ppo"][::-1]
        with self.assertRaises(ValueError):
            sr.contrasts(data, 0.05, 0.05)


class TestAmendment(unittest.TestCase):
    def test_amended_blocks_need_complete_runs_and_every_checkpoint(self):
        import json
        import tempfile

        cfg = protocol_cfg()
        sched = sr.schedule(cfg)
        done = {"complete": True, "transitions": sched["transitions"]}
        with tempfile.TemporaryDirectory() as d:
            for b in (0, 1):
                for arm in sr.ARMS:
                    os.makedirs(sr.run_dir(d, b, arm))
                    cks = [{"k": k + 1, "transitions": t} for k, t in enumerate(sched["checkpoint_transitions"])]
                    with open(os.path.join(sr.run_dir(d, b, arm), "checkpoints.json"), "w") as f:
                        json.dump({"checkpoints": cks}, f)
            training = {f"b{b}_{arm}": done for b in (0, 1) for arm in sr.ARMS}
            training["b2_draws"] = {"complete": False, "transitions": 230272}
            cwd = os.getcwd()
            os.chdir(ROOT)
            try:
                self.assertEqual(sr.amended_blocks(cfg, d, training, sched), [0, 1])
                with self.assertRaises(SystemExit):
                    sr.amended_blocks(cfg, d, {**training, "b1_ppo": {"complete": False, "transitions": 3840}}, sched)
                with open(os.path.join(sr.run_dir(d, 1, "ars"), "checkpoints.json"), "w") as f:
                    json.dump({"checkpoints": cks[-1:]}, f)                           # only the terminal checkpoint
                with self.assertRaises(SystemExit):
                    sr.amended_blocks(cfg, d, training, sched)
            finally:
                os.chdir(cwd)

    def test_budget_counts_whole_reset_batches(self):
        from lares.envs.isaac_pen_spin.symbolic_reward_child import BudgetExhausted, Loop

        class Env:
            num_envs = 128

            def reset(self):
                return ({"policy": torch.zeros(128, 4)}, {})

            def step(self, action):
                done = torch.ones(128, dtype=torch.bool)                              # a time-out of every environment
                return {"policy": torch.zeros(128, 4)}, None, ~done, done, {}

        class Ctrl:
            def reset(self, ctx):
                pass

            def reset_envs(self, mask):
                pass

        loop = Loop.__new__(Loop)
        loop.env, loop.ctrl, loop.cap, loop.ctx = Env(), Ctrl(), 300, None
        loop.transitions = loop.episodes = loop.falls = 0
        loop.reset_all()
        loop.step(torch.zeros(128, 20))
        self.assertEqual(loop.episodes, 256)                                          # the placement and its overwritten auto-reset
        with self.assertRaises(BudgetExhausted):
            loop.reset_all()


class TestFrozenFiles(unittest.TestCase):
    def test_reference_descriptor_matches_the_files(self):
        cfg = protocol_cfg()
        d = sr.check_reference(cfg)
        self.assertEqual(d["residual_params_sha256"], "2a509ef5ffc148cab2a513bc2c7339cca9a620b50dec2f7a988fd2b041c58cd1")
        incumbent = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"]["incumbent_residual"]
        self.assertEqual(cfg["reference_residual"], incumbent["selected"])            # the recorded values, not a copy

    def test_manifests_are_frozen_and_named_by_the_config(self):
        cfg = protocol_cfg()
        self.assertEqual(sr.build_manifests(cfg), [])                                 # all present and identical
        self.assertEqual(sr.manifest_for(cfg, "fresh", 2), "config/manifests/pen_spin_stage3_reward_fresh_s4769.yaml")

    def test_registry(self):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[sr.STUDY], "symbolic_reward")
        for name in ("build_manifests", "make_controller", "run"):
            self.assertTrue(callable(getattr(sr, name)))


if __name__ == "__main__":
    unittest.main()
