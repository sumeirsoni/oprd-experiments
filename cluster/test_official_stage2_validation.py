import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

CLUSTER = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CLUSTER / "official_stage2"))
import contract as task0_contract
import validate_task0


class OfficialIdentityPrimitiveTests(unittest.TestCase):
    def test_git_blob_identity_matches_git_hash_object(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "value.txt"
            path.write_bytes(b"pinned model file\n")
            import subprocess
            expected = subprocess.check_output(
                ["git", "hash-object", str(path)], text=True
            ).strip()
            self.assertEqual(
                validate_task0._git_blob_sha1(path, path.stat().st_size), expected
            )


class OfficialTask0AcceptanceTests(unittest.TestCase):
    def accept(self, run, contract):
        fake_yaml = SimpleNamespace(safe_load=json.loads)
        with mock.patch.dict(sys.modules, {"yaml": fake_yaml}), mock.patch(
            "contract.validate_hydra_config"
        ), mock.patch.object(validate_task0, "_validate_project_trust"):
            return validate_task0.accept(run, pathlib.Path.cwd(), contract)

    def make_run(self, root):
        contract = task0_contract.load_contract()
        run = root / "run"
        run.mkdir()
        (run / "preflight.json").write_text(
            json.dumps(
                {
                    "status": "preflight_ok",
                    "project_revision": "a" * 40,
                    "contract_sha256": task0_contract.contract_sha256(contract),
                }
            ),
            encoding="utf-8",
        )
        (run / "actual-hydra.yaml").write_text("{}\n", encoding="utf-8")
        (run / "actual-python-argv.json").write_text(
            '["-m","verl.trainer.main_ppo"]\n', encoding="utf-8"
        )
        checkpoint = run / "checkpoints" / "global_step_2174"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.bin").write_bytes(b"fixture checkpoint")
        merged = run / "merged-hf"
        merged.mkdir()
        (merged / "config.json").write_text("{}\n", encoding="utf-8")
        (merged / "model.safetensors").write_bytes(b"merged checkpoint")
        logs = run / "logs" / "terminal"
        logs.mkdir(parents=True)
        (logs / "task0.log").write_text("training completed\n", encoding="utf-8")
        return contract, run

    def test_acceptance_authenticates_exact_final_step_and_checkpoint_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, run = self.make_run(pathlib.Path(temporary))
            result = self.accept(run, contract)
            self.assertEqual(result["status"], "ACCEPTED_OFFICIAL_TASK0")
            self.assertEqual(result["global_step"], 2174)
            self.assertEqual(len(result["checkpoint_tree_sha256"]), 64)
            self.assertEqual(len(result["checkpoint_files"]), 1)

    def test_acceptance_rejects_later_steps_and_failure_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, run = self.make_run(pathlib.Path(temporary))
            (run / "checkpoints" / "global_step_2175").mkdir()
            with self.assertRaisesRegex(validate_task0.Task0ValidationError, "beyond"):
                self.accept(run, contract)

        with tempfile.TemporaryDirectory() as temporary:
            contract, run = self.make_run(pathlib.Path(temporary))
            (run / "logs" / "terminal" / "task0.log").write_text(
                "CUDA out of memory\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(validate_task0.Task0ValidationError, "failure marker"):
                self.accept(run, contract)


if __name__ == "__main__":
    unittest.main()
