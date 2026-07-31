import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "official_stage2"))
import contract as task0_contract
import prepare_inputs


EXPECTED_CONTRACT_SHA256 = "8dd4079e2822a7ae7f94c224b27560e3e6fe36fc09d77fdcd0472f71e3e6b253"


class OfficialTask0ContractTests(unittest.TestCase):
    def test_contract_digest_and_internal_counts_are_stable(self):
        contract = task0_contract.load_contract()
        self.assertEqual(task0_contract.contract_sha256(contract), EXPECTED_CONTRACT_SHA256)
        self.assertEqual(contract["training_data"]["expected_updates"], 2239)
        self.assertEqual(contract["training_data"]["expected_consumed_rows"], 17912)
        self.assertEqual(
            contract["training_data"]["expected_dropped_tail_rows"],
            [17912, 17913, 17914, 17915, 17916],
        )
        self.assertEqual(contract["rollout"]["responses_per_update"], 16)

    def test_contract_rejects_project_protocol_regressions(self):
        mutations = [
            (("rollout", "responses_per_prompt"), 1),
            (("rollout", "temperature"), 0.6),
            (("rollout", "retain_cap_hit_responses"), False),
            (("training_data", "shuffle"), True),
            (("objective", "last_k"), 1000),
            (("optimizer", "weight_decay"), 0.0),
            (("distributed", "gpus_per_node"), 1),
        ]
        original = task0_contract.load_contract()
        for path, replacement in mutations:
            with self.subTest(path=path):
                changed = copy.deepcopy(original)
                target = changed
                for component in path[:-1]:
                    target = target[component]
                target[path[-1]] = replacement
                with self.assertRaises(task0_contract.ContractError):
                    task0_contract.validate_contract(changed)

    def test_scientific_environment_is_fail_closed(self):
        task0_contract.reject_scientific_environment({"PATH": "/usr/bin"})
        with self.assertRaisesRegex(
            task0_contract.ContractError, "LR_SCHEDULER, TEMPERATURE"
        ):
            task0_contract.reject_scientific_environment(
                {"TEMPERATURE": "0.6", "LR_SCHEDULER": "cosine"}
            )

    def test_resolved_config_requires_every_scientific_value(self):
        resolved = {
            "data": {
                "rows": 17917,
                "shuffle": False,
                "drop_last": True,
                "prompt_batch_size": 8,
                "expected_updates": 2239,
                "enable_thinking": False,
            },
            "rollout": {
                "responses_per_prompt": 2,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_response_tokens": 16384,
                "retain_cap_hit_responses": True,
            },
            "objective": {
                "representation_only": True,
                "rank": 8,
                "layers": "all",
                "last_k": 2000,
                "coefficient": 10.0,
                "log_prob_top_k": 0,
            },
            "optimizer": {
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "gradient_clip_norm": 1000.0,
                "scheduler": "constant",
                "warmup_steps": 0,
            },
            "distributed": {
                "nodes": 1,
                "gpus_per_node": 8,
                "world_size": 8,
                "rollout_tensor_parallel_size": 1,
            },
        }
        contract = task0_contract.load_contract()
        task0_contract.validate_resolved_config(resolved, contract)
        resolved["rollout"]["responses_per_prompt"] = 1
        with self.assertRaisesRegex(
            task0_contract.ContractError, "rollout.responses_per_prompt"
        ):
            task0_contract.validate_resolved_config(resolved, contract)


class OfficialInputTests(unittest.TestCase):
    def test_source_manifest_requires_clean_pinned_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            source.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=source, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=source,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test User"],
                cwd=source,
                check=True,
            )
            (source / "tracked.txt").write_text("official source\n", encoding="utf-8")
            subprocess.run(["git", "add", "tracked.txt"], cwd=source, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=source, check=True)
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=source,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            contract = copy.deepcopy(task0_contract.load_contract())
            contract["official_source"]["commit"] = commit
            manifest_path = root / "source-manifest.json"
            manifest = prepare_inputs.authenticate_source(
                source, manifest_path, contract
            )
            self.assertEqual(manifest["commit"], commit)
            self.assertEqual(manifest["tracked_files"], 1)
            self.assertEqual(len(manifest["tree_sha256"]), 64)
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["tree_sha256"], manifest["tree_sha256"])

            (source / "untracked.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(prepare_inputs.InputError, "dirty"):
                prepare_inputs.authenticate_source(
                    source, root / "second-manifest.json", contract
                )

    def test_manifest_publication_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = pathlib.Path(temporary) / "manifest.json"
            destination.write_text("existing\n", encoding="utf-8")
            with self.assertRaisesRegex(prepare_inputs.InputError, "overwrite"):
                prepare_inputs._atomic_write_json(destination, {"ok": True})


if __name__ == "__main__":
    unittest.main()
