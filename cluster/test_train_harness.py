import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import oprd_bridge as bridge_module
import train_harness as harness


class LayerMappingTests(unittest.TestCase):
    def test_harness_uses_canonical_shared_bridge_primitives(self):
        self.assertIs(harness.BridgeBank, bridge_module.BridgeBank)
        self.assertIs(harness.resolve_layer_map, bridge_module.resolve_layer_map)
        self.assertIs(
            harness.proportional_teacher_layer,
            bridge_module.proportional_teacher_layer,
        )

    def test_all_layers_uses_paper_28_to_36_mapping(self):
        mapping = harness.resolve_layer_map("all", "", 28, 36)
        self.assertEqual(list(mapping), list(range(1, 29)))
        self.assertEqual(
            list(mapping.values()),
            [1, 2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18,
             19, 20, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 35, 36],
        )

    def test_explicit_subset_maps_after_student_resolution(self):
        self.assertEqual(
            harness.resolve_layer_map("1,14,28", "", 28, 36),
            {1: 1, 14: 18, 28: 36},
        )

    def test_fractional_selection_remains_backward_compatible(self):
        self.assertEqual(
            harness.resolve_layer_map("", "0.25,0.5", 28, 36),
            {7: 9, 14: 18},
        )

    def test_one_layer_student_maps_to_final_teacher_layer(self):
        self.assertEqual(harness.proportional_teacher_layer(1, 1, 36), 36)

    def test_invalid_or_duplicate_layers_are_rejected(self):
        with self.assertRaises(ValueError):
            harness.resolve_layer_map("0,2", "", 28, 36)
        with self.assertRaises(ValueError):
            harness.resolve_layer_map("7,7", "", 28, 36)
        with self.assertRaises(ValueError):
            harness.resolve_layer_map("", "0.01,0.02", 28, 36)


class TeacherPrefillTests(unittest.TestCase):
    def setUp(self):
        self.old_dev = harness.DEV
        harness.DEV = "cpu"

    def tearDown(self):
        harness.DEV = self.old_dev

    def test_topk_targets_can_also_store_diagnostic_hidden_states(self):
        class Output:
            pass

        class Backbone(torch.nn.Module):
            def forward(self, input_ids, output_hidden_states=True, use_cache=False):
                out = Output()
                shape = (*input_ids.shape, 4)
                out.hidden_states = (torch.zeros(shape), torch.ones(shape))
                return out

        class Teacher(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()

            def forward(self, input_ids, output_hidden_states=False, use_cache=False):
                out = Output()
                out.logits = torch.randn(*input_ids.shape, 32)
                return out

        entries = harness.teacher_prefill(
            Teacher(), [{"prompt_ids": [1, 2], "gen_ids": [3, 4, 5]}], [1],
            out=None, max_tok=3, last_k=2, topk_targets=True, store_hidden=True,
        )
        self.assertEqual(entries[0]["h"][1].shape, (2, 4))
        self.assertEqual(entries[0]["tk_ids"].shape, (2, 16))


class ExternalBridgeTests(unittest.TestCase):
    def setUp(self):
        self.old_dev = harness.DEV
        harness.DEV = "cpu"
        torch.manual_seed(3)
        self.identity = {
            "student_model_id": "org/student",
            "student_model_revision": "1" * 40,
            "teacher_model_id": "org/teacher",
            "teacher_model_revision": "2" * 40,
            "tokenizer_id": "org/tokenizer",
            "tokenizer_revision": "3" * 40,
            "corpus_id": "sha256:corpus",
            "stage1_contract_sha256": bridge_module.stage1_contract_sha256(),
            "layer_map": {1: 1},
        }

    def tearDown(self):
        harness.DEV = self.old_dev

    def make_bridge(self, offset=0.0):
        bridge = bridge_module.BridgeBank([1], 4, 6, 2)
        with torch.no_grad():
            bridge.student_projectors["1"].weight.copy_(
                torch.arange(8, dtype=torch.float32).reshape(2, 4) + offset
            )
            bridge.set_teacher_projection(
                1,
                torch.arange(6, dtype=torch.float32),
                torch.arange(12, dtype=torch.float32).reshape(2, 6),
            )
        return bridge.freeze()

    def publish(self, artifact, offset=0.0):
        bridge_module.publish_bridge_artifact(
            artifact,
            self.make_bridge(offset),
            stage1_metadata={
                "selected_epoch": 7,
                "selected_validation_cosine": 0.75,
            },
            **self.identity,
        )

    def args(self, artifact):
        _, manifest_sha256 = bridge_module.read_bridge_manifest(artifact)
        return SimpleNamespace(
            bridge_artifact=str(artifact),
            bridge_manifest_sha256=manifest_sha256,
            bridge_stage1_contract_sha256=bridge_module.stage1_contract_sha256(),
            student="org/student",
            student_revision="1" * 40,
            teacher="org/teacher",
            teacher_revision="2" * 40,
            tokenizer="org/tokenizer",
            tokenizer_revision="3" * 40,
            prompts="org/dataset",
            dataset_revision="4" * 40,
            dataset_split="train",
            prompt_key="prompt",
            n_prompts=8,
            seed=0,
            rollout_tokens=4096,
            max_tokens=512,
            supervise_last_k=2000,
            require_finished=True,
            rollout_refresh=100,
            batch_tokens=1024,
            lr=1e-5,
            bridge_loss_coef=10.0,
            backend="vllm",
            bridge_rank=2,
        )

    def test_external_artifact_loads_and_records_immutable_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = pathlib.Path(temporary) / "bridge"
            self.publish(artifact)
            bridge, stage1, reference = harness._load_external_bridge(
                self.args(artifact),
                SimpleNamespace(hidden_size=4),
                SimpleNamespace(hidden_size=6),
                {1: 1},
            )
            self.assertFalse(bridge.training)
            self.assertTrue(all(not value.requires_grad for value in bridge.parameters()))
            self.assertEqual(stage1["selected_epoch"], 7)
            self.assertEqual(reference["identity"]["corpus_id"], "sha256:corpus")
            self.assertEqual(reference["bridge"]["layer_map"], {"1": 1})
            self.assertEqual(len(reference["manifest_sha256"]), 64)

    def test_manifest_and_revision_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = pathlib.Path(temporary) / "bridge"
            self.publish(artifact)
            args = self.args(artifact)
            args.bridge_manifest_sha256 = "0" * 64
            with self.assertRaisesRegex(SystemExit, "manifest SHA-256 mismatch"):
                harness._load_external_bridge(
                    args,
                    SimpleNamespace(hidden_size=4),
                    SimpleNamespace(hidden_size=6),
                    {1: 1},
                )

            args = self.args(artifact)
            args.student_revision = "f" * 40
            with self.assertRaisesRegex(SystemExit, "student model revision"):
                harness._load_external_bridge(
                    args,
                    SimpleNamespace(hidden_size=4),
                    SimpleNamespace(hidden_size=6),
                    {1: 1},
                )

    def test_resume_accepts_path_relocation_but_rejects_changed_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = root / "first"
            second = root / "second"
            self.publish(first, offset=0.0)
            self.publish(second, offset=1.0)
            _, _, first_ref = harness._load_external_bridge(
                self.args(first),
                SimpleNamespace(hidden_size=4),
                SimpleNamespace(hidden_size=6),
                {1: 1},
            )
            relocated_ref = dict(first_ref)
            relocated_ref["path"] = str(root / "relocated")
            harness._validate_resume_bridge_reference(first_ref, relocated_ref)

            changed_data_ref = dict(first_ref)
            changed_data_ref["stage2_identity"] = {
                **first_ref["stage2_identity"],
                "dataset_revision": "9" * 40,
            }
            with self.assertRaisesRegex(SystemExit, "stage2_identity mismatch"):
                harness._validate_resume_bridge_reference(
                    first_ref, changed_data_ref
                )

            _, _, second_ref = harness._load_external_bridge(
                self.args(second),
                SimpleNamespace(hidden_size=4),
                SimpleNamespace(hidden_size=6),
                {1: 1},
            )
            with self.assertRaisesRegex(SystemExit, "manifest_sha256 mismatch"):
                harness._validate_resume_bridge_reference(first_ref, second_ref)
            with self.assertRaisesRegex(SystemExit, "atomic training state"):
                harness._validate_resume_bridge_reference(None, first_ref)

    def test_oprd_inputs_require_explicit_artifact_and_revisions(self):
        args = SimpleNamespace(
            bridge_artifact="",
            student_revision="",
            teacher_revision="",
            tokenizer_revision="",
            dataset_revision="",
            prompt_key="",
            bridge_manifest_sha256="",
            bridge_stage1_contract_sha256="",
        )
        with self.assertRaisesRegex(SystemExit, "explicit immutable inputs"):
            harness._require_oprd_artifact_inputs(args)

    def test_oprd_inputs_reject_movable_remote_revisions(self):
        args = SimpleNamespace(
            bridge_artifact="artifact",
            bridge_manifest_sha256="a" * 64,
            bridge_stage1_contract_sha256="b" * 64,
            student="org/student",
            student_revision="main",
            teacher="org/teacher",
            teacher_revision="c" * 40,
            tokenizer="org/tokenizer",
            tokenizer_revision="d" * 40,
            dataset_revision="e" * 40,
            prompt_key="prompt",
        )
        with self.assertRaisesRegex(SystemExit, "Hub commit SHA"):
            harness._require_oprd_artifact_inputs(args)

    def test_local_revision_is_recomputed_before_artifact_use(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = pathlib.Path(temporary) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n")
            correct = f"local-tree-sha256:{harness._sha256_tree(model)}"
            harness._require_immutable_revision(
                str(model), correct, "--student-revision"
            )
            (model / "config.json").write_text('{"changed": true}\n')
            with self.assertRaisesRegex(SystemExit, "current local source"):
                harness._require_immutable_revision(
                    str(model), correct, "--student-revision"
                )


class TrainingStateAndLossTests(unittest.TestCase):
    def setUp(self):
        self.old_dev = harness.DEV
        harness.DEV = "cpu"

    def tearDown(self):
        harness.DEV = self.old_dev

    @staticmethod
    def runtime_args(**overrides):
        values = {
            "n_prompts": 8,
            "batch_tokens": 128,
            "bridge_rank": 2,
            "linear_map_init_rows": 16,
            "topk": 4,
            "max_tokens": 32,
            "eval_avg_at": 2,
            "steps": 2,
            "eval_only": False,
            "rollout_tokens": 0,
            "supervise_last_k": 0,
            "rollout_refresh": 0,
            "start_step": 0,
            "lr": 1e-5,
            "tau": 1.0,
            "bridge_loss_coef": 10.0,
            "gpu_mem_util": 0.8,
            "prompts": "org/dataset",
            "dataset_revision": "",
            "dataset_split": "train",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_runtime_validation_rejects_values_that_break_training_progress(self):
        cases = {
            "batch_tokens": 0,
            "supervise_last_k": -1,
            "tau": 0.0,
            "topk": 0,
            "gpu_mem_util": 1.1,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                with self.assertRaises(SystemExit):
                    harness._validate_runtime_args(
                        self.runtime_args(**{field: value})
                    )

    def test_atomic_training_state_keeps_bridge_map_step_and_losses_together(self):
        student = torch.nn.Linear(3, 2)
        wmap = torch.nn.ModuleDict({"1": torch.nn.Linear(2, 4, bias=False)})
        reference = {"manifest_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "train_state_last.pt"
            harness.save_training_state(
                path, student, wmap, 100, [1.0, 0.5], bridge_reference=reference
            )
            state = torch.load(path, map_location="cpu")
            self.assertEqual(state["version"], 2)
            self.assertEqual(state["step"], 100)
            self.assertEqual(state["losses"], [1.0, 0.5])
            self.assertIn("1.weight", state["wmap"])
            self.assertEqual(state["bridge_reference"], reference)

    def test_linear_map_calibration_remains_separate_and_functional(self):
        student_rows = torch.randn(32, 3)
        expected_weight = torch.randn(5, 3)
        teacher_rows = student_rows @ expected_weight.T
        wmap = torch.nn.ModuleDict({"1": torch.nn.Linear(3, 5, bias=False)})
        harness.init_linear_map(
            wmap, {1: student_rows}, {1: teacher_rows}, out=None
        )
        self.assertTrue(
            torch.allclose(wmap["1"].weight, expected_weight, atol=1e-5, rtol=1e-5)
        )

    def test_stage2_is_squared_normalized_mse_and_backpropagates(self):
        bridge = harness.BridgeBank([1], 2, 2, 2)
        with torch.no_grad():
            bridge.student_projectors["1"].weight.copy_(torch.eye(2))
            bridge.set_teacher_projection(1, torch.zeros(2), torch.eye(2))
        bridge.freeze()

        hs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        ht = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        actual = harness.compute_loss(
            "oprd_bridge", {1: hs}, {1: ht}, [1], bridge, None, 1.0, 10,
            bridge_loss_coef=10.0,
        )
        expected = 10.0 * torch.nn.functional.mse_loss(
            torch.nn.functional.normalize(hs, dim=-1),
            torch.nn.functional.normalize(ht, dim=-1),
        )
        self.assertTrue(torch.allclose(actual, expected))
        self.assertNotAlmostEqual(
            actual.item(), (hs.detach() - ht).norm(dim=-1).mean().item()
        )
        actual.backward()
        self.assertIsNotNone(hs.grad)
        self.assertGreater(hs.grad.abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in bridge.parameters()))


if __name__ == "__main__":
    unittest.main()
