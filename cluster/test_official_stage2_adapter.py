import copy
import json
import pathlib
import sys
import tempfile
import unittest

import torch

CLUSTER_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CLUSTER_DIR))
sys.path.insert(0, str(CLUSTER_DIR / "official_stage2"))
import contract as task0_contract
import convert_bridge_to_official as adapter
import oprd_bridge


class OfficialBridgeAdapterTests(unittest.TestCase):
    def make_artifact(self, root):
        contract = copy.deepcopy(task0_contract.load_contract())
        layer_map = oprd_bridge.resolve_layer_map("all", "", 28, 36)
        bridge = oprd_bridge.BridgeBank(range(1, 29), 2048, 2560, 8)
        generator = torch.Generator(device="cpu").manual_seed(17)
        with torch.no_grad():
            for layer in bridge.layers:
                bridge.student_projectors[str(layer)].weight.copy_(
                    torch.randn(8, 2048, generator=generator)
                )
                bridge.set_teacher_projection(
                    layer,
                    torch.randn(2560, generator=generator),
                    torch.randn(8, 2560, generator=generator),
                )
        artifact = root / "bridge"
        oprd_bridge.publish_bridge_artifact(
            artifact,
            bridge,
            student_model_id=contract["models"]["student"]["id"],
            student_model_revision=contract["models"]["student"]["revision"],
            teacher_model_id=contract["models"]["teacher"]["id"],
            teacher_model_revision=contract["models"]["teacher"]["revision"],
            tokenizer_id=contract["models"]["tokenizer"]["id"],
            tokenizer_revision=contract["models"]["tokenizer"]["revision"],
            corpus_id="fixture-corpus",
            stage1_contract_sha256=contract["stage1_bridge"][
                "stage1_contract_sha256"
            ],
            layer_map=layer_map,
            stage1_metadata={"fixture": True},
        )
        _, manifest_sha256 = oprd_bridge.read_bridge_manifest(artifact)
        contract["stage1_bridge"]["artifact_manifest_sha256"] = manifest_sha256
        task0_contract.validate_contract(contract)
        return artifact, contract, layer_map

    def test_adapter_publishes_official_schema_and_exact_parity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            artifact, contract, layer_map = self.make_artifact(root)
            destination = root / "official-bank"
            result = adapter.convert_bridge(artifact, destination, contract)
            self.assertEqual(result["destination"], str(destination))
            self.assertTrue((destination / adapter.COMPLETE_FILENAME).is_file())
            self.assertTrue((destination / adapter.OFFICIAL_BANK_FILENAME).is_file())
            self.assertTrue((destination / adapter.AUDIT_TENSOR_FILENAME).is_file())

            manifest = json.loads(
                (destination / adapter.MANIFEST_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["parity_status"], "ok")
            self.assertEqual(len(manifest["official_layer_pairs_0_based"]), 28)
            self.assertEqual(manifest["official_layer_pairs_0_based"][0], [0, 0])
            self.assertEqual(manifest["official_layer_pairs_0_based"][-1], [27, 35])
            self.assertEqual(len(manifest["tensors"]), 84)
            self.assertTrue(all(value["finite"] for value in manifest["tensors"].values()))

            payload = torch.load(
                destination / adapter.OFFICIAL_BANK_FILENAME,
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(payload["rank"], 8)
            self.assertEqual(payload["projector_type"], "linear")
            self.assertEqual(payload["layer_pairs"], [(s - 1, t - 1) for s, t in layer_map.items()])
            self.assertIn("projectors.s0_t0.weight", payload["state_dict"])
            self.assertEqual(payload["state_dict"]["projectors.s0_t0.weight"].shape, (8, 2048))
            self.assertEqual(payload["frozen_pt_weights"]["s27_t35"].shape, (8, 2560))
            self.assertEqual(payload["frozen_pt_means"]["s27_t35"].shape, (2560,))

            parity = json.loads(
                (destination / adapter.PARITY_FILENAME).read_text(encoding="utf-8")
            )
            self.assertEqual(parity["status"], "ok")
            self.assertEqual(parity["maximum_student_projection_abs_difference"], 0.0)
            self.assertEqual(parity["maximum_teacher_projection_abs_difference"], 0.0)
            self.assertEqual(parity["maximum_normalized_mse_abs_difference"], 0.0)

    def test_adapter_rejects_manifest_mismatch_and_output_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            artifact, contract, _ = self.make_artifact(root)
            wrong = copy.deepcopy(contract)
            wrong["stage1_bridge"]["artifact_manifest_sha256"] = "0" * 64
            with self.assertRaisesRegex(adapter.AdapterError, "manifest mismatch"):
                adapter.convert_bridge(artifact, root / "wrong", wrong)

            destination = root / "existing"
            destination.mkdir()
            with self.assertRaisesRegex(adapter.AdapterError, "overwrite"):
                adapter.convert_bridge(artifact, destination, contract)


if __name__ == "__main__":
    unittest.main()
