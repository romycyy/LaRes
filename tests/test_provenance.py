"""Records, ledgers, archive upserts, gate levers and provenance checks.

``debug_spec.md`` Stage C (D5, D7, D8, D11) and section 2.1. The end-to-end
versions of these checks, on records the orchestrator actually wrote, are in
``tests/test_two_generation_mock.py``. Run from the project root::

    python -m unittest tests.test_provenance
"""

import copy
import json
import os
import sys
import tempfile
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from lares.core.obs_schema import active_schema, get_obs_schema  # noqa: E402
from lares.core.policy_validator import (  # noqa: E402
    GATE_NOT_A_LEVER,
    SCALE_PARAMETER,
    validate_policy,
)
from lares.core.symbolic_policy import SymbolicPolicy  # noqa: E402
from lares.eval.manifest import to_v3  # noqa: E402
from lares.search import (  # noqa: E402
    LINEAGE_COMPLETE,
    LINEAGE_FAILED_VALIDATION,
    LINEAGE_LEGACY_MISSING,
    LINEAGE_PARTIAL_RECOVERY,
    SCOPE_CONFIRMED,
    SCOPE_SCREEN,
    SLOT_ROBUST,
    STATUS_INVALID,
    Archive,
    ExperimentRecord,
    LLMCallLedger,
    ResourceLedger,
    SchemaError,
    env_identity,
    load_records,
    source_hash,
    write_feedback_bundle,
)
from lares.search.generation import resolve_registry_env_id, write_artifact  # noqa: E402
from lares.search.provenance import lineage_status, reconstruct_actions  # noqa: E402
from lares.search.resources import resource_problems  # noqa: E402

sys.path.insert(0, _SCRIPT_DIR)
from test_search import make_report  # noqa: E402

SCHEMA = get_obs_schema("push-v2")
OBS_DIM, ACT_DIM = 39, 4


# ---------------------------------------------------------------------------
#  Environment naming
# ---------------------------------------------------------------------------


class TestEnvironmentIdentity(unittest.TestCase):
    def test_the_public_id_is_v2_and_the_registry_id_is_v3(self):
        identity = env_identity("push-v2")
        self.assertEqual(identity["canonical_env_id"], "push-v2")
        self.assertEqual(identity["resolved_registry_env_id"], "push-v3")
        self.assertIn("-v3", identity["env_id_mapping_rule"])
        self.assertTrue(identity["env_id_mapping_version"])

    def test_the_rule_matches_the_manifest_mapping(self):
        for env_id in ("push-v2", "reach-v2", "window-close-v2"):
            self.assertEqual(resolve_registry_env_id(env_id), to_v3(env_id))


# ---------------------------------------------------------------------------
#  Experiment records
# ---------------------------------------------------------------------------


def call_ref(call_id="call_00001"):
    return {"call_id": call_id, "label": "generation", "purpose": "per_idea",
            "prompt_sha256": "p" * 64, "response_sha256": "r" * 64}


def failed_generated_record(generation=0, **overrides):
    lineage = {
        "status": LINEAGE_FAILED_VALIDATION,
        "problems": [],
        "generation": generation,
        "slot": 1,
        "attempt": 0,
        "attempt_id": f"gen{generation}_cand1_a0",
        "previous_attempt_id": None,
        "feedback_id": "fb_gen0_abc" if generation else None,
        "context_parent_ids": ["gen0_cand0", "gen0_cand1"] if generation else [],
        "parent_ids": ["gen0_cand0"] if generation else [],
        "mode": "repair" if generation else "",
        "idea": {"idea_id": f"gen{generation}_idea1"},
        "proposal": ({"proposal_id": "gen1_prop1", "mode": "repair", "parent_ids": ["gen0_cand0"]}
                     if generation else None),
        "mapping": {},
    }
    base = dict(
        experiment_id=f"gen{generation}_cand1_a0",
        candidate_id=f"gen{generation}_cand1",
        generation=generation,
        status=STATUS_INVALID,
        parent_ids=["gen0_cand0"] if generation else [],
        idea_id=f"gen{generation}_idea1",
        proposal_id="gen1_prop1" if generation else "",
        code={"source_hash": source_hash("class GeneratedPolicy: ..."), "source": "class GeneratedPolicy: ..."},
        llm={"calls": [call_ref()]},
        llm_call_ids=["call_00001"],
        validation={"passed": False, "stage": "instantiation", "errors": ["[raw_obs_indexing] ..."]},
        lineage=lineage,
        resources=ResourceLedger().to_dict(),
    )
    base.update(overrides)
    return ExperimentRecord(**base)


class TestGeneratedRecordContract(unittest.TestCase):
    def test_a_complete_failed_attempt_validates(self):
        self.assertTrue(failed_generated_record().validate())
        self.assertTrue(failed_generated_record(generation=1).validate())

    def test_a_rejected_generated_record_cannot_have_empty_lineage(self):
        with self.assertRaises(SchemaError) as ctx:
            failed_generated_record(lineage={}).validate()
        self.assertIn("lineage", str(ctx.exception))

    def test_slot_and_attempt_are_required(self):
        record = failed_generated_record()
        del record.lineage["slot"]
        with self.assertRaises(SchemaError):
            record.validate()

    def test_an_idea_is_required_past_ideation(self):
        with self.assertRaises(SchemaError):
            failed_generated_record(idea_id="").validate()

    def test_a_failure_at_ideation_needs_no_idea(self):
        record = failed_generated_record(idea_id="", validation={"passed": False, "stage": "ideation", "errors": ["not json"]})
        record.lineage["idea"] = None
        self.assertTrue(record.validate())

    def test_call_references_are_required(self):
        with self.assertRaises(SchemaError):
            failed_generated_record(llm_call_ids=[], llm={"calls": []}).validate()
        with self.assertRaises(SchemaError):
            failed_generated_record(llm={"calls": [{"call_id": "call_00001"}]}).validate()

    def test_a_validation_result_is_required(self):
        with self.assertRaises(SchemaError):
            failed_generated_record(validation={"stage": "harness"}).validate()

    def test_a_later_generation_needs_a_proposal_and_feedback(self):
        with self.assertRaises(SchemaError):
            failed_generated_record(generation=1, proposal_id="").validate()
        record = failed_generated_record(generation=1)
        record.lineage["feedback_id"] = None
        with self.assertRaises(SchemaError):
            record.validate()

    def test_a_parent_outside_the_context_is_refused(self):
        record = failed_generated_record(generation=1, parent_ids=["gen0_cand9"])
        record.lineage["proposal"]["parent_ids"] = ["gen0_cand9"]
        with self.assertRaises(SchemaError):
            record.validate()

    def test_gradient_steps_cannot_hide_in_environment_steps(self):
        with self.assertRaises(SchemaError):
            failed_generated_record(resources={"environment_steps": 2000}).validate()

    def test_an_evaluated_generated_record_needs_its_evidence(self):
        record = failed_generated_record(status="evaluated", validation={"passed": True, "stage": "passed", "errors": []})
        record.lineage["status"] = LINEAGE_COMPLETE
        record.evaluation = {"manifest_ids": ["m"], "action_mode": "deterministic"}
        with self.assertRaises(SchemaError):
            record.validate()

    def test_records_written_before_origin_existed_load_as_legacy(self):
        d = failed_generated_record().to_dict()
        del d["origin"]
        self.assertEqual(ExperimentRecord.from_dict(d).origin, "legacy")


class TestVersionedRecords(unittest.TestCase):
    def test_changed_evidence_is_versioned_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = failed_generated_record()
            record.save(tmp)
            self.assertEqual(record.record_version, 1)
            record.save(tmp)
            self.assertFalse(os.path.isdir(os.path.join(tmp, "history")))

            record.final_disposition = "superseded"
            record.save(tmp)
            self.assertEqual(record.record_version, 2)
            with open(os.path.join(tmp, "history", "gen0_cand1_a0.v1.json")) as f:
                self.assertEqual(json.load(f)["final_disposition"], "")
            loaded = load_records(tmp)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].record_version, 2)


# ---------------------------------------------------------------------------
#  Archive
# ---------------------------------------------------------------------------


class TestArchiveUpsert(unittest.TestCase):
    def test_confirmed_metrics_replace_screen_metrics_for_slot_assignment(self):
        archive = Archive()
        archive.add("a", 0, make_report("a", labels={"success": 5, "never_reached_object": 5}), 6, "h1")
        archive.add("b", 0, make_report("b", labels={"success": 3, "never_reached_object": 7}), 6, "h2")
        self.assertEqual(archive.slot(SLOT_ROBUST).candidate_id, "a")
        entry = archive.get("a")
        self.assertEqual(entry.metrics_source, SCOPE_SCREEN)
        self.assertEqual(entry.success_rate, 0.5)

        archive.confirm("a", make_report("a", labels={"success": 1, "never_reached_object": 9}))
        self.assertEqual(len(archive), 2)
        self.assertEqual(entry.metrics_source, SCOPE_CONFIRMED)
        self.assertEqual(entry.success_rate, 0.1)
        self.assertEqual(entry.screen_metrics["success_rate"], 0.5)
        self.assertEqual(entry.confirmed_metrics["success_rate"], 0.1)
        self.assertEqual(archive.slot(SLOT_ROBUST).candidate_id, "b")

    def test_an_upsert_cannot_change_the_code_an_entry_describes(self):
        archive = Archive()
        archive.add("a", 0, make_report("a"), 6, "h1")
        with self.assertRaises(ValueError):
            archive.add("a", 0, make_report("a"), 6, "different")

    def test_confirming_an_unknown_candidate_is_refused(self):
        with self.assertRaises(KeyError):
            Archive().confirm("nobody", make_report("x"))


# ---------------------------------------------------------------------------
#  Gate levers and scale parameters
# ---------------------------------------------------------------------------


class _GatedPolicy(SymbolicPolicy):
    USE_RETURN = True

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w_reach = nn.Parameter(torch.tensor(3.0))
        self.w_push = nn.Parameter(torch.tensor(2.0))
        self.radius = nn.Parameter(torch.tensor(0.05))
        self.grip = nn.Parameter(torch.tensor(0.0))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        to_obj = obj - tcp
        d = torch.norm(to_obj, dim=-1, keepdim=True) + 1e-8
        to_goal = goal - obj
        dg = torch.norm(to_goal, dim=-1, keepdim=True) + 1e-8
        contact = torch.sigmoid(40.0 * (self.radius - d))
        if self.USE_RETURN:
            contact = self.record_gate("contact", contact)
        else:
            self.record_gate("contact", contact)
        move = (1 - contact) * self.w_reach * to_obj / d + contact * self.w_push * to_goal / dg
        grip = self.grip.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        return {"w_reach": (0.1, 10.0), "w_push": (0.1, 10.0), "radius": (0.01, 0.2), "grip": (-1.0, 1.0)}


class _NoOpGatePolicy(_GatedPolicy):
    USE_RETURN = False


class _ScaleParamPolicy(_GatedPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        mean, _ = super().forward(obs)
        return mean, torch.exp(self.log_std) * torch.ones_like(mean)

    def get_param_ranges(self):
        return {**super().get_param_ranges(), "log_std": (-5.0, 0.0)}


def build(cls):
    with active_schema(SCHEMA):
        return cls(OBS_DIM, ACT_DIM)


class TestGateLevers(unittest.TestCase):
    def test_a_gate_whose_return_value_is_discarded_fails_validation(self):
        report = validate_policy(build(_NoOpGatePolicy), OBS_DIM, ACT_DIM, schema=SCHEMA)
        self.assertIn(GATE_NOT_A_LEVER, report.categories)
        self.assertIn("contact = self.record_gate", report.summary())

    def test_a_correctly_assigned_gate_passes(self):
        report = validate_policy(build(_GatedPolicy), OBS_DIM, ACT_DIM, schema=SCHEMA)
        self.assertTrue(report.ok, report.summary())
        self.assertIn("gate_levers", report.checked)

    def test_the_check_leaves_no_override_behind(self):
        policy = build(_GatedPolicy)
        validate_policy(policy, OBS_DIM, ACT_DIM, schema=SCHEMA)
        self.assertEqual(policy.gate_overrides, {})


class TestScaleParameters(unittest.TestCase):
    def test_a_scale_parameter_is_an_error_where_generation_forbids_it(self):
        report = validate_policy(build(_ScaleParamPolicy), OBS_DIM, ACT_DIM, schema=SCHEMA, forbid_scale_parameters=True)
        self.assertIn(SCALE_PARAMETER, report.categories)
        self.assertIn("log_std", report.summary())

    def test_elsewhere_it_is_only_a_warning(self):
        report = validate_policy(build(_ScaleParamPolicy), OBS_DIM, ACT_DIM, schema=SCHEMA)
        self.assertTrue(report.ok, report.summary())
        self.assertIn(SCALE_PARAMETER, {w.category for w in report.warnings})


# ---------------------------------------------------------------------------
#  Ledger, feedback bundle, resources
# ---------------------------------------------------------------------------


class TestLedgerAndBundle(unittest.TestCase):
    def test_a_tampered_prompt_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = LLMCallLedger(tmp)
            record = ledger.record("ideation", "m", 1.0, [{"role": "user", "content": "hi"}], ["ok"],
                                   {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}, 0.1,
                                   attempt_ids=["a", "b"])
            self.assertEqual(ledger.verify(), {})
            self.assertTrue(record.shared)
            with open(os.path.join(tmp, record.prompt_path), "a") as f:
                f.write(" ")
            self.assertIn(record.call_id, ledger.verify())
            self.assertEqual(len(LLMCallLedger(tmp)), 1)

    def test_an_unknown_label_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                LLMCallLedger(tmp).record("chitchat", "m", 1.0, [], [], {}, 0.0)

    def test_totals_count_each_call_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = LLMCallLedger(tmp)
            a = ledger.record("ideation", "m", 1.0, [], ["x"], {"total_tokens": 100}, 0.0)
            b = ledger.record("generation", "m", 1.0, [], ["y"], {"total_tokens": 10}, 0.0)
            totals = ledger.totals([a.call_id, b.call_id, a.call_id, a.call_id])
            self.assertEqual(totals["llm_calls"], 2)
            self.assertEqual(totals["llm_total_tokens"], 110)

    def test_a_feedback_bundle_detects_edited_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = write_feedback_bundle(tmp, 0, "dev#head10", ["gen0_cand0", "gen0_cand0", "gen0_cand1"], "evidence")
            self.assertEqual(bundle.context_parent_ids, ["gen0_cand0", "gen0_cand1"])
            self.assertTrue(bundle.validate(tmp))
            with open(os.path.join(tmp, bundle.text_path), "w") as f:
                f.write("different evidence")
            with self.assertRaises(SchemaError):
                bundle.validate(tmp)

    def test_resource_ledger_keeps_streams_apart(self):
        from lares.eval.runner import EpisodeRecord, ManifestResult

        episodes = [EpisodeRecord("c", "t", 0.0, 0.0, 150, None, None, 0.0, None, 0.0, 0.0, True) for _ in range(3)]
        ledger = ResourceLedger()
        ledger.add_episodes("probe", ManifestResult("a", "m", "deterministic", episodes))
        ledger.add("bc_gradient_steps", 2000)
        d = ledger.to_dict()
        self.assertEqual(d["probe_episodes"], 3)
        self.assertEqual(d["probe_environment_steps"], 450)
        self.assertEqual(d["screen_environment_steps"], 0)
        self.assertEqual(ledger.simulator_steps, 450)
        self.assertNotIn("environment_steps", d)
        self.assertEqual(resource_problems(d), [])
        self.assertTrue(resource_problems({"environment_steps": 2000}))
        with self.assertRaises(KeyError):
            ledger.add("environment_steps", 1)


# ---------------------------------------------------------------------------
#  Lineage status and reconstruction
# ---------------------------------------------------------------------------


class TestLineageStatus(unittest.TestCase):
    def test_statuses_are_distinguished(self):
        self.assertEqual(lineage_status(failed_generated_record()), LINEAGE_FAILED_VALIDATION)
        legacy = ExperimentRecord("gen1_cand0", "gen1_cand0", 1, "rejected", origin="legacy")
        self.assertEqual(lineage_status(legacy), LINEAGE_LEGACY_MISSING)
        sidecar = {"records": {"gen1_cand0": {"lineage_status": LINEAGE_PARTIAL_RECOVERY}}}
        self.assertEqual(lineage_status(legacy, sidecar), LINEAGE_PARTIAL_RECOVERY)


class TestReconstruction(unittest.TestCase):
    def test_a_policy_rebuilt_from_hashed_artifacts_gives_identical_actions(self):
        import io
        import inspect

        source = inspect.getsource(_GatedPolicy).replace("class _GatedPolicy", "class GeneratedPolicy")
        policy = build(_GatedPolicy)
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(0.01)
        probe = 0.1 * torch.randn(8, OBS_DIM, generator=torch.Generator().manual_seed(0))
        with tempfile.TemporaryDirectory() as tmp:
            def tensor_ref(name, value):
                buffer = io.BytesIO()
                torch.save(value, buffer)
                return write_artifact(tmp, name, buffer.getvalue())

            recon = {
                "source": write_artifact(tmp, "policy_source.py", source),
                "selected_state": tensor_ref("selected_state.pt", policy.state_dict()),
                "probe_batch": tensor_ref("probe.pt", probe),
                "env_id": "push-v2", "obs_dim": OBS_DIM, "action_dim": ACT_DIM,
                "fixed_action_scale": 0.1,
            }
            rebuilt = reconstruct_actions(tmp, recon)
            with torch.no_grad():
                self.assertTrue(torch.equal(rebuilt, torch.tanh(policy(probe)[0])))
            tampered = copy.deepcopy(recon)
            tampered["selected_state"]["sha256"] = "0" * 64
            with self.assertRaises(ValueError):
                reconstruct_actions(tmp, tampered)


if __name__ == "__main__":
    unittest.main()
