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
            return validate_task0.accept(
                run, pathlib.Path.cwd(), task0_contract.EXECUTION_PATH, contract
            )

    def make_run(self, root):
        contract = task0_contract.load_contract()
        run = root / "official-task0-4h200-prod-r1"
        run.mkdir()
        execution = task0_contract.load_execution_manifest(contract)
        topology_path = run / "resolved-execution-topology.json"
        topology_path.write_text(
            json.dumps(
                task0_contract.effective_execution_topology(execution, contract),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        (run / "preflight.json").write_text(
            json.dumps(
                {
                    "status": "preflight_ok",
                    "project_revision": "a" * 40,
                    "contract_sha256": task0_contract.contract_sha256(contract),
                    "execution_manifest_sha256": task0_contract.execution_manifest_sha256(
                        execution, contract
                    ),
                    "effective_world_size": 4,
                    "resolved_execution_sha256": validate_task0._sha256_file(topology_path),
                }
            ),
            encoding="utf-8",
        )
        (run / "actual-hydra.yaml").write_text("{}\n", encoding="utf-8")
        (run / "actual-python-argv.json").write_text(
            json.dumps([
                "-m",
                "verl.trainer.main_ppo",
                "trainer.n_gpus_per_node=4",
                "trainer.max_actor_ckpt_to_keep=1",
                f"hydra.run.dir={run.resolve()}/hydra-run",
            ]) + "\n",
            encoding="utf-8",
        )
        checkpoint = run / "checkpoints" / "global_step_2239"
        actor = checkpoint / "actor"
        actor.mkdir(parents=True)
        for kind in ("model", "optim", "extra_state"):
            for rank in range(4):
                (actor / f"{kind}_world_size_4_rank_{rank}.pt").write_bytes(b"fixture")
        (actor / "huggingface").mkdir()
        (actor / "huggingface" / "config.json").write_text("{}\n")
        merged = run / "merged-hf"
        merged.mkdir()
        (merged / "config.json").write_text("{}\n", encoding="utf-8")
        (merged / "model.safetensors").write_bytes(b"merged checkpoint")
        logs = run / "logs" / "terminal"
        logs.mkdir(parents=True)
        (logs / "task0.log").write_text("training completed\n", encoding="utf-8")
        (run / "official-runtime.log").write_text(
            "training/global_step:2239\n", encoding="utf-8"
        )
        return contract, run

    def test_acceptance_authenticates_exact_final_step_and_checkpoint_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, run = self.make_run(pathlib.Path(temporary))
            result = self.accept(run, contract)
            self.assertEqual(result["status"], "ACCEPTED_OFFICIAL_TASK0")
            self.assertEqual(result["global_step"], 2239)
            self.assertEqual(len(result["checkpoint_tree_sha256"]), 64)
            self.assertEqual(len(result["checkpoint_files"]), 13)
            self.assertEqual(len(result["fsdp_shards"]), 12)
            self.assertEqual(result["effective_world_size"], 4)

    def test_acceptance_rejects_mutated_topology_symlink_and_ambiguous_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            contract, run = self.make_run(root)
            renamed = root / "unauthorized-copy"
            run.rename(renamed)
            with self.assertRaisesRegex(
                validate_task0.Task0ValidationError, "namespace is not authorized"
            ):
                self.accept(renamed, contract)

        with tempfile.TemporaryDirectory() as temporary:
            contract, run = self.make_run(pathlib.Path(temporary))
            (run / "resolved-execution-topology.json").write_text("{}\n")
            with self.assertRaisesRegex(
                validate_task0.Task0ValidationError, "topology digest"
            ):
                self.accept(run, contract)

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            contract, run = self.make_run(root)
            checkpoint = run / "checkpoints" / "global_step_2239"
            outside = root / "outside-checkpoint"
            checkpoint.rename(outside)
            checkpoint.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(
                validate_task0.Task0ValidationError, "real directory"
            ):
                self.accept(run, contract)

        with tempfile.TemporaryDirectory() as temporary:
            contract, run = self.make_run(pathlib.Path(temporary))
            (run / "official-runtime.log").write_text("training/global_step:22390\n")
            with self.assertRaisesRegex(
                validate_task0.Task0ValidationError, "exact final global step"
            ):
                self.accept(run, contract)

    def test_acceptance_rejects_incomplete_or_mixed_shard_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, run = self.make_run(pathlib.Path(temporary))
            shard = next(run.rglob("model_world_size_4_rank_2.pt"))
            shard.unlink()
            with self.assertRaisesRegex(
                validate_task0.Task0ValidationError, "shard topology"
            ):
                self.accept(run, contract)

        with tempfile.TemporaryDirectory() as temporary:
            contract, run = self.make_run(pathlib.Path(temporary))
            actor = next(run.rglob("global_step_2239")) / "actor"
            (actor / "model_world_size_8_rank_0.pt").write_bytes(b"unexpected")
            with self.assertRaisesRegex(
                validate_task0.Task0ValidationError, "shard topology"
            ):
                self.accept(run, contract)

    def test_acceptance_rejects_later_steps_and_failure_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            contract, run = self.make_run(pathlib.Path(temporary))
            (run / "checkpoints" / "global_step_2240").mkdir()
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
