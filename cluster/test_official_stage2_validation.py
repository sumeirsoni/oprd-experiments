import json
import os
import pathlib
import subprocess
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


class OfficialTask0ResumeValidationTests(unittest.TestCase):
    def test_resume_required_arguments_are_exact_and_continuation_local(self):
        contract = task0_contract.load_contract()
        execution = task0_contract.load_execution_manifest(contract)
        resume = task0_contract.load_resume_manifest(contract, execution)
        source = pathlib.Path("/official/run/checkpoints/global_step_1600")
        continuation = pathlib.Path("/official/run/continuations/resume-global-step-1600-r1")
        self.assertEqual(
            validate_task0._required_resume_arguments(resume, source, continuation),
            {
                "trainer.resume_mode": "resume_path",
                "trainer.resume_from_path": str(source),
                "trainer.default_local_dir": str(continuation / "checkpoints"),
                "trainer.output_log_path": str(continuation / "logs" / "terminal" / "resume.log"),
                "trainer.total_training_steps": "2239",
                "trainer.n_gpus_per_node": "4",
                "trainer.max_actor_ckpt_to_keep": "1",
                "hydra.run.dir": str(continuation / "hydra-run"),
            },
        )

    def test_resume_path_guard_rejects_escape_and_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary).resolve()
            inside = root / "inside"
            inside.mkdir()
            self.assertEqual(
                validate_task0._require_real_descendant(inside, root, "fixture"),
                inside,
            )
            outside = root.parent / "outside-resume-fixture"
            with self.assertRaisesRegex(
                validate_task0.Task0ValidationError, "outside its authorized root"
            ):
                validate_task0._require_real_descendant(outside, root, "fixture")
            target = root / "target"
            target.mkdir()
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(
                validate_task0.Task0ValidationError, "contains a symlink"
            ):
                validate_task0._require_real_descendant(link, root, "fixture")

    def test_authenticated_tree_rejects_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "real").write_text("value\n", encoding="utf-8")
            (root / "link").symlink_to(root / "real")
            with self.assertRaisesRegex(
                validate_task0.Task0ValidationError, "contains a symlink"
            ):
                validate_task0._tree_manifest(root)

    def test_atomic_publication_is_executable_durable_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source.json"
            destination = root / "acceptance.json"
            source.write_text('{"ok":true}\n', encoding="utf-8")
            validate_task0.atomic_publish_file(source, destination)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertFalse(any(root.glob(".acceptance.json.publish-*")))
            with self.assertRaisesRegex(
                validate_task0.Task0ValidationError, "already exists"
            ):
                validate_task0.atomic_publish_file(source, destination)
            complete = root / "COMPLETE"
            validate_task0.atomic_publish_complete(complete)
            self.assertEqual(complete.read_text(), "ACCEPTED_OFFICIAL_TASK0\n")
            with self.assertRaisesRegex(
                validate_task0.Task0ValidationError, "already exists"
            ):
                validate_task0.atomic_publish_complete(complete)

    def test_validator_publication_cli_executes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source.json"
            destination = root / "published.json"
            source.write_text('{"status":"ok"}\n', encoding="utf-8")
            command = [
                sys.executable,
                str(pathlib.Path(validate_task0.__file__)),
                "publish-file",
                "--source", str(source),
                "--destination", str(destination),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            repeated = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already exists", repeated.stderr)
            complete = root / "COMPLETE"
            subprocess.run([
                sys.executable,
                str(pathlib.Path(validate_task0.__file__)),
                "publish-complete",
                "--destination", str(complete),
            ], check=True, capture_output=True, text=True)
            self.assertEqual(complete.read_text(), "ACCEPTED_OFFICIAL_TASK0\n")

    def test_resume_capture_wrapper_injects_and_validates_exact_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fake_python = root / "python"
            calls = root / "calls.jsonl"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, subprocess, sys\n"
                "with open(os.environ['CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n"
                "a=sys.argv[1:]\n"
                "if a and a[0] in ('-', '-c'): raise SystemExit(subprocess.call([sys.executable]+a))\n"
                "if 'emit-execution' in a:\n"
                " print(json.dumps({'effective_topology':{'world_size':4},'checkpoint_policy':{'max_actor_checkpoints_to_keep':1}})); raise SystemExit(0)\n"
                "if 'validate-hydra' in a: raise SystemExit(0)\n"
                "if '--cfg' in a: print('{}'); raise SystemExit(0)\n"
                "raise SystemExit(0)\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            hydra = root / "actual-hydra.yaml"
            argv = root / "actual-python-argv.json"
            namespace = root / "namespace"
            continuation = namespace / "continuations" / "resume-global-step-1600-r1"
            continuation.mkdir(parents=True)
            source = namespace / "global_step_1600"
            source.mkdir()
            environment = os.environ.copy()
            environment.update({
                "CALLS": str(calls),
                "OPRD_REAL_PYTHON": str(fake_python),
                "OPRD_HYDRA_CAPTURE": str(hydra),
                "OPRD_ARGV_CAPTURE": str(argv),
                "OPRD_CONTRACT_TOOL": str(root / "contract.py"),
                "OPRD_CONTRACT": str(root / "contract.json"),
                "OPRD_OFFICIAL_ROOT": str(root),
                "OPRD_EXECUTION_DEVIATION": str(root / "execution.json"),
                "OPRD_RESUME_AUTHORIZATION": "task0-7258005",
                "OPRD_RESUME_FROM_PATH": str(source),
                "OPRD_RESUME_LOCAL_DIR": str(continuation / "checkpoints"),
                "OPRD_RESUME_HYDRA_DIR": str(continuation / "hydra-run"),
                "OPRD_RESUME_MANIFEST": str(root / "resume.json"),
                "OPRD_RESUME_NAMESPACE": str(namespace),
                "OPRD_RESUME_CONTINUATION": str(continuation),
                "OPRD_RESUME_TERMINAL_LOG": str(continuation / "logs" / "terminal" / "resume.log"),
                "PROJECT_PATH": str(continuation),
            })
            wrapper = CLUSTER / "official_stage2" / "capture_bin" / "python3"
            process = subprocess.run([
                str(wrapper), "-m", "verl.trainer.main_ppo",
                "trainer.n_gpus_per_node=8",
                f"trainer.default_local_dir={continuation}/token_reward_direct_fixture-2026-08-21_00-00-00-OPD",
                f"trainer.output_log_path={continuation}/logs/terminal/token_reward_direct_fixture.log",
            ], env=environment, capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            captured = json.loads(argv.read_text())
            expected = {
                "trainer.resume_mode=resume_path",
                f"trainer.resume_from_path={source}",
                f"trainer.default_local_dir={continuation / 'checkpoints'}",
                f"trainer.output_log_path={continuation / 'logs' / 'terminal' / 'resume.log'}",
                "trainer.total_training_steps=2239",
                "trainer.n_gpus_per_node=4",
                "trainer.max_actor_ckpt_to_keep=1",
                f"hydra.run.dir={continuation / 'hydra-run'}",
            }
            self.assertTrue(expected.issubset(set(captured)))
            validation_calls = [json.loads(line) for line in calls.read_text().splitlines()]
            validate_call = next(call for call in validation_calls if "validate-hydra" in call)
            self.assertIn("--resume", validate_call)
            self.assertIn("--official-namespace", validate_call)
            self.assertIn("--continuation-dir", validate_call)

    def test_resume_dataloader_instrumentation_emits_only_after_real_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (root / "torchdata" / "stateful_dataloader").mkdir(parents=True)
            (root / "torch.py").write_text(
                "import json\ndef load(path,*a,**k): return json.load(open(path))\n"
            )
            (root / "torchdata" / "__init__.py").write_text("")
            (root / "torchdata" / "stateful_dataloader" / "__init__.py").write_text(
                "class StatefulDataLoader:\n"
                " def load_state_dict(self,state): self.loaded=state; return 'ok'\n"
            )
            state = {
                "_snapshot_step": 1600,
                "samples_yielded": 12800,
                "_sampler_iter_yielded": 1600,
                "_num_workers": 8,
                "_iterator_finished": False,
            }
            data = root / "data.pt"
            data.write_text(json.dumps(state))
            import hashlib
            digest = hashlib.sha256(data.read_bytes()).hexdigest()
            evidence = root / "evidence.json"
            instrumentation = CLUSTER / "official_stage2" / "resume_instrumentation"
            environment = os.environ.copy()
            environment.update({
                "PYTHONPATH": f"{instrumentation}{os.pathsep}{root}",
                "OPRD_RESUME_DATALOADER_EVIDENCE": str(evidence),
                "OPRD_RESUME_DATA_PATH": str(data),
                "OPRD_RESUME_DATA_SHA256": digest,
                "OPRD_RESUME_DATA_STATE": json.dumps({
                    "snapshot_step": 1600,
                    "samples_yielded": 12800,
                    "sampler_iter_yielded": 1600,
                    "num_workers": 8,
                    "iterator_finished": False,
                }),
            })
            subprocess.run([
                sys.executable, "-c",
                "import torch; from torchdata.stateful_dataloader import StatefulDataLoader; "
                f"s=torch.load({str(data)!r}); assert not __import__('pathlib').Path({str(evidence)!r}).exists(); "
                "assert StatefulDataLoader().load_state_dict(s)=='ok'",
            ], env=environment, check=True, capture_output=True, text=True)
            record = json.loads(evidence.read_text())
            self.assertEqual(record["status"], "dataloader_state_restored")
            self.assertEqual(record["data_path"], str(data.resolve()))
            self.assertEqual(record["data_sha256"], digest)

    def test_resume_acceptance_end_to_end_progression_and_composite_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            run = root / "official-task0-4h200-prod-r1"
            continuation = run / "continuations" / "resume-global-step-1600-r1"
            source = run / "source" / "global_step_1600"
            source.mkdir(parents=True)
            continuation.mkdir(parents=True)
            run = run.resolve(strict=True)
            continuation = continuation.resolve(strict=True)
            source = source.resolve(strict=True)
            contract = task0_contract.load_contract()
            execution = task0_contract.load_execution_manifest(contract)
            resume = task0_contract.load_resume_manifest(contract, execution)
            digest = task0_contract.resume_manifest_sha256(resume, contract, execution)
            original = {"source_checkpoint": str(source)}
            current_inputs = {"authenticated": True}
            preflight = {
                "status": "resume_preflight_ok",
                "project_revision": "b" * 40,
                "historical_project_revision": resume["scope"]["original_project_revision"],
                "contract_sha256": task0_contract.contract_sha256(contract),
                "execution_manifest_sha256": task0_contract.execution_manifest_sha256(execution, contract),
                "resume_manifest_sha256": digest,
                "continuation": str(continuation),
                "current_inputs": current_inputs,
                **original,
                **resume["resume_arithmetic"],
            }
            (continuation / "resume-preflight.json").write_text(json.dumps(preflight))
            resolved_resume = task0_contract.effective_resume_authorization(
                resume, contract, execution
            )
            resolved_resume_path = continuation / "resolved-resume-authorization.json"
            resolved_resume_path.write_text(
                json.dumps(resolved_resume, sort_keys=True, separators=(",", ":")) + "\n"
            )
            args = ["-m", "verl.trainer.main_ppo"] + [
                f"{key}={value}" for key, value in
                validate_task0._required_resume_arguments(resume, source, continuation).items()
            ]
            (continuation / "actual-python-argv.json").write_text(json.dumps(args))
            (continuation / "actual-hydra.yaml").write_text("{}\n")
            actor = continuation / "checkpoints" / "global_step_2239" / "actor"
            actor.mkdir(parents=True)
            for kind in ("model", "optim", "extra_state"):
                for rank in range(4):
                    (actor / f"{kind}_world_size_4_rank_{rank}.pt").write_bytes(b"fixture")
            (actor / "huggingface").mkdir()
            (actor / "huggingface" / "config.json").write_text("{}\n")
            merged = continuation / "merged-hf"
            merged.mkdir()
            (merged / "model.safetensors").write_bytes(b"merged")
            terminal = continuation / "logs" / "terminal"
            terminal.mkdir(parents=True)
            (terminal / "resume.log").write_text("completed\n")
            restore = [
                f"Load from checkpoint folder: {source}",
                "Setting global step to 1600",
                f"Resuming from {source}",
                "dataloader state successfully restored from data.pt",
            ]
            for rank in range(4):
                restore.extend([
                    f"Loaded model from x/actor/model_world_size_4_rank_{rank}.pt",
                    f"Loaded optimizer from x/actor/optim_world_size_4_rank_{rank}.pt",
                    f"Loaded rng from x/actor/extra_state_world_size_4_rank_{rank}.pt",
                    f"Loaded lr_scheduler from x/actor/extra_state_world_size_4_rank_{rank}.pt",
                ])
            steps = [f"training/global_step:{step}" for step in range(1601, 2240)]
            runtime = continuation / "official-runtime.log"
            runtime.write_text("\n".join(restore + steps) + "\n")
            data_record = next(
                record for record in resume["source_checkpoint"]["inventory"]
                if record["path"] == "data.pt"
            )
            (continuation / "dataloader-restoration.json").write_text(json.dumps({
                "status": "dataloader_state_restored",
                "data_path": str(source / "data.pt"),
                "data_sha256": data_record["sha256"],
                "state": resume["source_checkpoint"]["dataloader_state"],
            }, sort_keys=True, separators=(",", ":")) + "\n")
            context = (run, continuation, execution, resume, digest)
            fake_yaml = SimpleNamespace(safe_load=lambda value: {})
            patches = (
                mock.patch.object(validate_task0, "_resume_context", return_value=context),
                mock.patch.object(validate_task0, "_authenticate_original", return_value=original),
                mock.patch.object(validate_task0, "_authenticate_current_resume_inputs", return_value=current_inputs),
                mock.patch("contract.validate_hydra_config"),
                mock.patch.dict(sys.modules, {"yaml": fake_yaml}),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                accepted = validate_task0.resume_accept(
                    run, root, "b" * 40, task0_contract.EXECUTION_PATH,
                    task0_contract.RESUME_PATH, contract,
                )
                self.assertEqual(accepted["global_step"], 2239)
                (continuation / "acceptance.json").write_text(
                    json.dumps(accepted, sort_keys=True, separators=(",", ":"))
                )
                composite = validate_task0.resume_composite_accept(
                    run, root, "b" * 40, task0_contract.EXECUTION_PATH,
                    task0_contract.RESUME_PATH, contract,
                )
                self.assertEqual(composite["status"], "ACCEPTED_OFFICIAL_TASK0")
                canonical_resolved_bytes = resolved_resume_path.read_bytes()

                resolved_resume_path.unlink()
                for function in (validate_task0.resume_accept, validate_task0.resume_composite_accept):
                    with self.assertRaisesRegex(validate_task0.Task0ValidationError, "resolved resume authorization"):
                        function(run, root, "b" * 40, task0_contract.EXECUTION_PATH, task0_contract.RESUME_PATH, contract)
                resolved_resume_path.write_bytes(canonical_resolved_bytes)

                mutated = dict(resolved_resume)
                mutated["resume_arithmetic"] = dict(mutated["resume_arithmetic"])
                mutated["resume_arithmetic"]["remaining_updates"] = 638
                resolved_resume_path.write_text(json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n")
                for function in (validate_task0.resume_accept, validate_task0.resume_composite_accept):
                    with self.assertRaisesRegex(validate_task0.Task0ValidationError, "canonical authorization"):
                        function(run, root, "b" * 40, task0_contract.EXECUTION_PATH, task0_contract.RESUME_PATH, contract)
                resolved_resume_path.write_bytes(canonical_resolved_bytes)

                outside_resolved = root / "outside-resolved.json"
                outside_resolved.write_bytes(canonical_resolved_bytes)
                resolved_resume_path.unlink()
                resolved_resume_path.symlink_to(outside_resolved)
                for function in (validate_task0.resume_accept, validate_task0.resume_composite_accept):
                    with self.assertRaisesRegex(validate_task0.Task0ValidationError, "regular file"):
                        function(run, root, "b" * 40, task0_contract.EXECUTION_PATH, task0_contract.RESUME_PATH, contract)
                resolved_resume_path.unlink()
                resolved_resume_path.write_bytes(canonical_resolved_bytes)

                resolved_resume_path.write_text(json.dumps(resolved_resume, indent=2, sort_keys=True) + "\n")
                for function in (validate_task0.resume_accept, validate_task0.resume_composite_accept):
                    with self.assertRaisesRegex(validate_task0.Task0ValidationError, "canonical byte representation"):
                        function(run, root, "b" * 40, task0_contract.EXECUTION_PATH, task0_contract.RESUME_PATH, contract)
                resolved_resume_path.write_bytes(canonical_resolved_bytes)

                runtime.write_text("\n".join(restore + steps[:-1]) + "\n")
                with self.assertRaisesRegex(validate_task0.Task0ValidationError, "every resumed step|final step"):
                    validate_task0.resume_accept(run, root, "b" * 40, task0_contract.EXECUTION_PATH, task0_contract.RESUME_PATH, contract)
                runtime.write_text("\n".join(restore + steps + ["training/global_step:2240"]) + "\n")
                with self.assertRaisesRegex(validate_task0.Task0ValidationError, "beyond the target"):
                    validate_task0.resume_accept(run, root, "b" * 40, task0_contract.EXECUTION_PATH, task0_contract.RESUME_PATH, contract)
                evidence_path = continuation / "dataloader-restoration.json"
                evidence_bytes = evidence_path.read_bytes()
                evidence_path.unlink()
                runtime.write_text("\n".join([line for line in restore if "dataloader" not in line] + steps) + "\n")
                with self.assertRaisesRegex(validate_task0.Task0ValidationError, "dataloader restoration evidence"):
                    validate_task0.resume_accept(run, root, "b" * 40, task0_contract.EXECUTION_PATH, task0_contract.RESUME_PATH, contract)
                evidence_path.write_bytes(evidence_bytes)
                for negative_restore in (
                    "failed to restore dataloader state; starting over",
                    "could not load sampler, using fresh iterator",
                ):
                    runtime.write_text("\n".join(restore + [negative_restore] + steps) + "\n")
                    with self.assertRaisesRegex(validate_task0.Task0ValidationError, "negative dataloader"):
                        validate_task0.resume_accept(run, root, "b" * 40, task0_contract.EXECUTION_PATH, task0_contract.RESUME_PATH, contract)
                runtime.write_text("\n".join(restore + steps + ["Traceback (most recent call last):", "ValueError: fatal", "RuntimeError: DataLoader worker (pid 1) is killed by signal: Killed"]) + "\n")
                with self.assertRaisesRegex(validate_task0.Task0ValidationError, "unexplained traceback"):
                    validate_task0.resume_accept(run, root, "b" * 40, task0_contract.EXECUTION_PATH, task0_contract.RESUME_PATH, contract)

    def test_resume_cli_exposes_all_three_phases(self):
        help_text = pathlib.Path(validate_task0.__file__).read_text()
        self.assertIn('"resume-preflight"', help_text)
        self.assertIn('"resume-accept"', help_text)
        self.assertIn('"resume-composite-accept"', help_text)


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
