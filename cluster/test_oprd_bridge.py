import json
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import oprd_bridge as bridge_module


EXPECTED_28_TO_36 = [
    1, 2, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18,
    19, 20, 22, 23, 24, 26, 27, 28, 30, 31, 32, 33, 35, 36,
]


class LayerMappingTests(unittest.TestCase):
    def test_exact_all_layer_28_to_36_mapping(self):
        mapping = bridge_module.resolve_layer_map("all", "", 28, 36)
        self.assertEqual(list(mapping), list(range(1, 29)))
        self.assertEqual(list(mapping.values()), EXPECTED_28_TO_36)


class BridgeArtifactTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.bridge = bridge_module.BridgeBank([1, 2], 4, 6, 3)
        with torch.no_grad():
            self.bridge.student_projectors["1"].weight.copy_(
                torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10
            )
            self.bridge.student_projectors["2"].weight.copy_(
                torch.arange(12, 24, dtype=torch.float32).reshape(3, 4) / 10
            )
            self.bridge.set_teacher_projection(
                1,
                torch.arange(6, dtype=torch.float32),
                torch.arange(18, dtype=torch.float32).reshape(3, 6) / 20,
            )
            self.bridge.set_teacher_projection(
                2,
                torch.arange(6, dtype=torch.float32) + 1,
                torch.arange(18, 36, dtype=torch.float32).reshape(3, 6) / 20,
            )
        self.identity = {
            "student_model_id": "org/student-1.5b",
            "student_model_revision": "student-revision-abc",
            "teacher_model_id": "org/teacher-4b",
            "teacher_model_revision": "teacher-revision-def",
            "tokenizer_id": "org/shared-tokenizer",
            "tokenizer_revision": "tokenizer-revision-ghi",
            "corpus_id": "org/math-corpus@sha256:abc123",
            "stage1_contract_sha256": bridge_module.stage1_contract_sha256(),
            "layer_map": {1: 1, 2: 3},
        }
        self.expected = {
            "expected_student_model_id": self.identity["student_model_id"],
            "expected_student_model_revision": self.identity[
                "student_model_revision"
            ],
            "expected_teacher_model_id": self.identity["teacher_model_id"],
            "expected_teacher_model_revision": self.identity[
                "teacher_model_revision"
            ],
            "expected_tokenizer_id": self.identity["tokenizer_id"],
            "expected_tokenizer_revision": self.identity["tokenizer_revision"],
            "expected_corpus_id": self.identity["corpus_id"],
            "expected_stage1_contract_sha256": self.identity[
                "stage1_contract_sha256"
            ],
            "expected_layer_map": self.identity["layer_map"],
            "expected_rank": 3,
            "expected_student_dim": 4,
            "expected_teacher_dim": 6,
        }

    def publish(self, path, *, bridge=None, extra_json_files=None):
        return bridge_module.publish_bridge_artifact(
            path,
            self.bridge if bridge is None else bridge,
            stage1_metadata={"epochs": 20, "best_val_cosine": 0.75},
            extra_json_files=extra_json_files,
            **self.identity,
        )

    def test_publish_load_round_trip_has_versioned_checksums_and_is_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = pathlib.Path(temporary) / "bridge-v1"
            published = self.publish(artifact)
            self.assertEqual(published, artifact)
            self.assertTrue((artifact / bridge_module.COMPLETE_FILENAME).is_file())
            self.assertTrue((artifact / bridge_module.MANIFEST_FILENAME).is_file())
            self.assertTrue((artifact / bridge_module.TENSOR_FILENAME).is_file())
            self.assertFalse(list(artifact.parent.glob(f".{artifact.name}.tmp-*")))

            manifest, manifest_sha256 = bridge_module.read_bridge_manifest(artifact)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["artifact_type"], "oprd_bridge")
            self.assertEqual(manifest["storage"]["format"], "safetensors")
            self.assertEqual(len(manifest["storage"]["sha256"]), 64)
            marker = (artifact / bridge_module.COMPLETE_FILENAME).read_text().split()
            self.assertEqual(marker, ["manifest_sha256", manifest_sha256])

            restored, metadata = bridge_module.load_bridge_artifact(
                artifact, device="cpu", **self.expected
            )
            probe = torch.randn(5, 4)
            self.assertTrue(
                torch.equal(
                    self.bridge.project_student(1, probe),
                    restored.project_student(1, probe),
                )
            )
            teacher_probe = torch.randn(5, 6)
            self.assertTrue(
                torch.equal(
                    self.bridge.project_teacher(2, teacher_probe),
                    restored.project_teacher(2, teacher_probe),
                )
            )
            self.assertEqual(metadata, {"epochs": 20, "best_val_cosine": 0.75})
            self.assertFalse(restored.training)
            self.assertTrue(
                all(not parameter.requires_grad for parameter in restored.parameters())
            )

    def test_corrupt_tensor_is_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = pathlib.Path(temporary) / "bridge-v1"
            self.publish(artifact)
            tensor_path = artifact / bridge_module.TENSOR_FILENAME
            content = bytearray(tensor_path.read_bytes())
            content[-1] ^= 0x01
            tensor_path.write_bytes(content)

            with self.assertRaises(bridge_module.ArtifactCorruptError):
                bridge_module.load_bridge_artifact(artifact, **self.expected)

    def test_nonfinite_source_tensor_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = pathlib.Path(temporary) / "bridge-v1"
            with torch.no_grad():
                self.bridge.student_projectors["1"].weight[0, 0] = float("nan")
            with self.assertRaises(bridge_module.ArtifactValidationError):
                self.publish(artifact)
            self.assertFalse(artifact.exists())
            self.assertFalse(list(artifact.parent.glob(f".{artifact.name}.tmp-*")))

    def test_checksummed_nonfinite_tensor_is_rejected_on_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = pathlib.Path(temporary) / "bridge-v1"
            self.publish(artifact)
            tensor_path = artifact / bridge_module.TENSOR_FILENAME
            load_file, save_file = bridge_module._safetensors_functions()
            state = load_file(str(tensor_path), device="cpu")
            name = next(iter(state))
            state[name] = state[name].clone()
            state[name].reshape(-1)[0] = float("inf")
            save_file(state, str(tensor_path))

            manifest_path = artifact / bridge_module.MANIFEST_FILENAME
            manifest = json.loads(manifest_path.read_text())
            manifest["storage"]["size_bytes"] = tensor_path.stat().st_size
            manifest["storage"]["sha256"] = bridge_module._sha256_file(tensor_path)
            manifest_bytes = bridge_module._canonical_json(manifest)
            manifest_path.write_bytes(manifest_bytes)
            manifest_sha256 = bridge_module._sha256_bytes(manifest_bytes)
            (artifact / bridge_module.COMPLETE_FILENAME).write_text(
                f"manifest_sha256 {manifest_sha256}\n", encoding="ascii"
            )

            with self.assertRaises(bridge_module.ArtifactCorruptError):
                bridge_module.load_bridge_artifact(artifact, **self.expected)

    def test_staged_reload_failure_never_commits_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = pathlib.Path(temporary) / "bridge-v1"
            with mock.patch.object(
                bridge_module,
                "_load_bridge_artifact_common",
                side_effect=bridge_module.ArtifactCorruptError("forced staged failure"),
            ):
                with self.assertRaises(bridge_module.ArtifactCorruptError):
                    self.publish(artifact)
            self.assertFalse(artifact.exists())
            self.assertFalse(list(artifact.parent.glob(f".{artifact.name}.tmp-*")))

    def test_corrupt_declared_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = pathlib.Path(temporary) / "bridge-v1"
            self.publish(
                artifact,
                extra_json_files={"provenance.json": {"source": "unit-test"}},
            )
            (artifact / "provenance.json").write_text("{}\n")
            with self.assertRaises(bridge_module.ArtifactCorruptError):
                bridge_module.load_bridge_artifact(artifact, **self.expected)

    def test_partial_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            complete_artifact = root / "complete"
            self.publish(complete_artifact)

            for missing in (
                bridge_module.COMPLETE_FILENAME,
                bridge_module.MANIFEST_FILENAME,
                bridge_module.TENSOR_FILENAME,
            ):
                with self.subTest(missing=missing):
                    partial = root / f"partial-{missing.replace('.', '-')}"
                    shutil.copytree(complete_artifact, partial)
                    (partial / missing).unlink()
                    with self.assertRaises(bridge_module.ArtifactIncompleteError):
                        bridge_module.load_bridge_artifact(partial, **self.expected)
                    if missing == bridge_module.COMPLETE_FILENAME:
                        with self.assertRaises(bridge_module.ArtifactIncompleteError):
                            bridge_module.read_bridge_manifest(partial)

    def test_identity_map_rank_and_dimension_mismatches_are_rejected(self):
        cases = {
            "expected_student_model_id": "wrong-student",
            "expected_student_model_revision": "wrong-student-revision",
            "expected_teacher_model_id": "wrong-teacher",
            "expected_teacher_model_revision": "wrong-teacher-revision",
            "expected_tokenizer_id": "wrong-tokenizer",
            "expected_tokenizer_revision": "wrong-tokenizer-revision",
            "expected_corpus_id": "wrong-corpus",
            "expected_stage1_contract_sha256": "0" * 64,
            "expected_layer_map": {1: 1, 2: 4},
            "expected_rank": 2,
            "expected_student_dim": 5,
            "expected_teacher_dim": 7,
        }
        with tempfile.TemporaryDirectory() as temporary:
            artifact = pathlib.Path(temporary) / "bridge-v1"
            self.publish(artifact)
            for field, wrong_value in cases.items():
                with self.subTest(field=field):
                    expected = dict(self.expected)
                    expected[field] = wrong_value
                    with self.assertRaises(bridge_module.ArtifactMismatchError):
                        bridge_module.load_bridge_artifact(artifact, **expected)

    def test_existing_artifact_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = pathlib.Path(temporary) / "bridge-v1"
            self.publish(artifact)
            manifest_before = (artifact / bridge_module.MANIFEST_FILENAME).read_bytes()
            tensor_before = (artifact / bridge_module.TENSOR_FILENAME).read_bytes()

            with self.assertRaises(bridge_module.ArtifactAlreadyExistsError):
                self.publish(artifact)

            self.assertEqual(
                (artifact / bridge_module.MANIFEST_FILENAME).read_bytes(),
                manifest_before,
            )
            self.assertEqual(
                (artifact / bridge_module.TENSOR_FILENAME).read_bytes(),
                tensor_before,
            )


if __name__ == "__main__":
    unittest.main()
