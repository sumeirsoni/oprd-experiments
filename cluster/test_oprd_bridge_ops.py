import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "nexus"))
import oprd_bridge_ops as ops


class AtomicJsonAndPathTests(unittest.TestCase):
    def test_atomic_json_round_trip_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "record.json"
            ops.atomic_write_json(path, {"old": True})
            ops.atomic_write_json(path, {"new": [1, 2, 3]})
            self.assertEqual(ops.read_json(path), {"new": [1, 2, 3]})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_containment_rejects_outside_and_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            scratch = root / "scratch"
            outside = root / "outside"
            scratch.mkdir()
            outside.mkdir()
            with self.assertRaises(ops.UnsafePathError):
                ops.canonical_path_under_scratch(scratch, outside / "artifact")

            link = scratch / "escape"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ops.UnsafePathError):
                ops.canonical_path_under_scratch(scratch, link / "artifact")

    def test_source_bundle_digest_changes_with_any_runtime_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for relative in ops.SOURCE_BUNDLE_PATHS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            first = ops.compute_source_bundle_sha256(root)
            changed = root / ops.SOURCE_BUNDLE_PATHS[-1]
            changed.write_text("changed", encoding="utf-8")
            self.assertNotEqual(first, ops.compute_source_bundle_sha256(root))

    def test_containment_requires_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary)
            with self.assertRaisesRegex(ops.UnsafePathError, "absolute"):
                ops.canonical_path_under_scratch(scratch, pathlib.Path("relative"))


class PreflightTests(unittest.TestCase):
    def test_capacity_rejects_low_blocks_and_low_inodes(self):
        low_blocks = types.SimpleNamespace(
            f_bavail=9, f_favail=100, f_frsize=4096, f_bsize=4096
        )
        with self.assertRaisesRegex(ops.OpsError, "free blocks"):
            ops.check_statvfs_capacity(
                low_blocks, min_free_blocks=10, min_free_inodes=1
            )

        low_inodes = types.SimpleNamespace(
            f_bavail=100, f_favail=4, f_frsize=4096, f_bsize=4096
        )
        with self.assertRaisesRegex(ops.OpsError, "free inodes"):
            ops.check_statvfs_capacity(
                low_inodes, min_free_blocks=10, min_free_inodes=5
            )

        low_bytes = types.SimpleNamespace(
            f_bavail=10, f_favail=100, f_frsize=1024, f_bsize=1024
        )
        with self.assertRaisesRegex(ops.OpsError, "free bytes"):
            ops.check_statvfs_capacity(
                low_bytes,
                min_free_blocks=1,
                min_free_inodes=1,
                min_free_bytes=20_000,
            )

    def test_lfs_quota_parser_enforces_block_and_inode_headroom(self):
        parsed = ops._parse_lfs_quota(
            "Filesystem kbytes quota limit grace files quota limit grace\n"
            "/scratch 100 150 200 - 10 15 20 -\n"
        )
        self.assertEqual(parsed["remaining_bytes"], 100 * 1024)
        self.assertEqual(parsed["remaining_inodes"], 10)

    def test_preflight_uses_configured_python_and_reports_quota_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary) / "scratch"
            scratch.mkdir()
            hf_home = scratch / "hf-cache"
            writable = scratch / "records"
            script = scratch / "phase.sbatch"
            script.write_text("#!/bin/sh\n", encoding="utf-8")
            python = scratch / "python3.12"
            python.write_text(
                "#!/bin/sh\n"
                "if [ \"$1\" = \"--version\" ]; then\n"
                "  echo 'Python 3.12.9'\n"
                "  exit 0\n"
                "fi\n"
                "case \"${2:-}\" in\n"
                "  *importlib.metadata*) echo '{\"torch\":\"test\"}' ;;\n"
                "esac\n"
                "exit 0\n",
                encoding="utf-8",
            )
            python.chmod(0o755)
            sbatch = scratch / "sbatch"
            sacct = scratch / "sacct"
            ffmpeg = scratch / "ffmpeg"
            for executable in (sbatch, sacct):
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(0o755)
            ffmpeg.write_text(
                "#!/bin/sh\necho 'ffmpeg version test'\n", encoding="utf-8"
            )
            ffmpeg.chmod(0o755)

            requirements = scratch / "requirements.txt"
            requirements.write_text("torch>=2\n", encoding="utf-8")
            marker = scratch / "env-ready.json"
            ops.atomic_write_json(
                marker,
                {
                    "schema_version": 1,
                    "python_executable": os.fspath(python.resolve()),
                    "requirements_sha256": hashlib.sha256(
                        requirements.read_bytes()
                    ).hexdigest(),
                    "packages": {"torch": "test"},
                    "ffmpeg_path": os.fspath(ffmpeg.resolve()),
                    "ffmpeg_version": "ffmpeg version test",
                },
            )
            commands = {
                "sbatch": os.fspath(sbatch),
                "sacct": os.fspath(sacct),
                "ffmpeg": os.fspath(ffmpeg),
            }
            report = ops.run_preflight(
                scratch=scratch,
                hf_home=hf_home,
                python=python,
                required_imports=["json", "some.fake.module"],
                required_scripts=[script],
                writable_paths=[writable],
                environment_marker=marker,
                requirements_file=requirements,
                min_free_blocks=1,
                min_free_inodes=1,
                quota_user="test-user",
                allow_quota_unavailable=True,
                which_fn=lambda command: commands.get(command),
            )
            self.assertTrue(report["ok"], report)
            self.assertEqual(report["checks"]["python"]["version"], "3.12.9")
            self.assertEqual(
                report["checks"]["environment_marker"]["packages"]["torch"],
                "test",
            )
            self.assertEqual(report["quota"]["status"], "unavailable")

            requirements.write_text("torch>=3\n", encoding="utf-8")
            changed = ops.run_preflight(
                scratch=scratch,
                hf_home=hf_home,
                python=python,
                required_imports=[],
                required_scripts=[script],
                environment_marker=marker,
                requirements_file=requirements,
                min_free_blocks=1,
                min_free_inodes=1,
                quota_user="test-user",
                allow_quota_unavailable=True,
                which_fn=lambda command: commands.get(command),
            )
            self.assertFalse(changed["ok"])
            self.assertTrue(
                any("requirements digest" in error for error in changed["errors"]),
                changed,
            )

    def test_preflight_rejects_non_312_python(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary) / "scratch"
            scratch.mkdir()
            python = scratch / "python"
            python.write_text("#!/bin/sh\necho 'Python 3.11.8'\n", encoding="utf-8")
            python.chmod(0o755)
            executable = os.fspath(python)
            report = ops.run_preflight(
                scratch=scratch,
                hf_home=scratch / "hf",
                python=python,
                required_imports=[],
                required_scripts=[],
                which_fn=lambda command: executable
                if command in {"sbatch", "sacct"}
                else None,
            )
            self.assertFalse(report["ok"])
            self.assertTrue(
                any("must be 3.12.x" in error for error in report["errors"]),
                report,
            )


class ClaimTests(unittest.TestCase):
    def test_claim_excludes_conflicts_allows_same_identity_and_releases_explicitly(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary) / "scratch"
            artifacts = scratch / "artifacts"
            records = scratch / "records"
            artifacts.mkdir(parents=True)
            records.mkdir()
            artifact = artifacts / "bridge"
            record = records / "run.json"
            ops.initialize_record(
                scratch=scratch,
                record_path=record,
                run_id="run-1",
                build_dir=scratch / "build",
                artifact_dir=artifact,
                phase_order=("one",),
            )

            first = ops.claim_artifact(
                scratch=scratch,
                artifact_dir=artifact,
                run_id="run-1",
                record_path=record,
            )
            self.assertFalse(first["reused"])
            second = ops.claim_artifact(
                scratch=scratch,
                artifact_dir=artifact,
                run_id="run-1",
                record_path=record,
            )
            self.assertTrue(second["reused"])
            with self.assertRaises(ops.ClaimConflictError):
                ops.claim_artifact(
                    scratch=scratch,
                    artifact_dir=artifact,
                    run_id="run-2",
                    record_path=records / "other.json",
                )
            with self.assertRaises(ops.ClaimConflictError):
                ops.release_artifact_claim(
                    scratch=scratch,
                    artifact_dir=artifact,
                    run_id="run-2",
                    record_path=record,
                )

            ops.record_job(record, phase="one", job_id="101")
            with self.assertRaises(ops.ActiveJobsError):
                ops.release_artifact_claim(
                    scratch=scratch,
                    artifact_dir=artifact,
                    run_id="run-1",
                    record_path=record,
                )
            recorded = ops.read_json(record)
            recorded["phases"]["one"]["attempts"][-1]["classification"] = "failed"
            ops.atomic_write_json(record, recorded)

            released = ops.release_artifact_claim(
                scratch=scratch,
                artifact_dir=artifact,
                run_id="run-1",
                record_path=record,
            )
            self.assertTrue(released["ok"])
            self.assertFalse(ops.claim_directory(artifact).exists())


    def test_resume_claim_can_protect_existing_authenticated_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary) / "scratch"
            scratch.mkdir()
            artifact = scratch / "artifact"
            artifact.mkdir()
            manifest = artifact / "manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")
            digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (artifact / "COMPLETE").write_text(
                f"manifest_sha256 {digest}\n", encoding="ascii"
            )
            record = scratch / "records" / "run.json"
            ops.initialize_record(
                scratch=scratch,
                record_path=record,
                run_id="run-1",
                build_dir=scratch / "build",
                artifact_dir=artifact,
                phase_order=("stage2-smoke",),
            )
            with self.assertRaisesRegex(
                ops.ClaimConflictError, "producing build"
            ):
                ops.claim_artifact(
                    scratch=scratch,
                    artifact_dir=artifact,
                    run_id="run-1",
                    record_path=record,
                    allow_existing_artifact=True,
                )


class RecordTests(unittest.TestCase):
    def initialize(self, scratch, phases=("one", "two")):
        build = scratch / "build"
        artifact = scratch / "artifact"
        record = scratch / "records" / "run.json"
        ops.initialize_record(
            scratch=scratch,
            record_path=record,
            run_id="run-1",
            build_dir=build,
            artifact_dir=artifact,
            phase_order=phases,
        )
        return record

    def test_record_update_does_not_create_parent_before_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            outside = pathlib.Path(temporary) / "outside" / "record.json"
            with self.assertRaisesRegex(ops.OpsError, "missing or unsafe"):
                ops.record_job(outside, phase="one", job_id="101")
            self.assertFalse(outside.parent.exists())

    def test_record_updates_append_attempts_and_keep_statuses_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary) / "scratch"
            scratch.mkdir()
            record_path = self.initialize(scratch)
            ops.record_job(record_path, phase="one", job_id="101")
            ops.record_job(record_path, phase="one", job_id="102")
            ops.record_job(record_path, phase="two", job_id="103")
            record = ops.mark_submitted(record_path)

            attempts = record["phases"]["one"]["attempts"]
            self.assertEqual([attempt["job_id"] for attempt in attempts], ["101", "102"])
            self.assertEqual([attempt["attempt"] for attempt in attempts], [1, 2])
            self.assertEqual(record["submission_status"], "submitted")
            self.assertEqual(record["dag_status"], "active")

    def test_submission_intent_blocks_resume_until_job_id_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary) / "scratch"
            scratch.mkdir()
            record_path = self.initialize(scratch, phases=("one",))
            pending = ops.begin_job_submission(
                record_path,
                phase="one",
                dependency_job_id=None,
                script="phase.sbatch",
            )
            with self.assertRaisesRegex(ops.ActiveJobsError, "submission intents"):
                ops.determine_resume_start_index(pending)
            completed = ops.record_job(
                record_path,
                phase="one",
                job_id="101",
                script="phase.sbatch",
            )
            self.assertNotIn(
                "pending_submission", completed["phases"]["one"]
            )

    def test_submission_failure_is_distinct_from_dag_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary) / "scratch"
            scratch.mkdir()
            record_path = self.initialize(scratch, phases=("one",))
            record = ops.mark_submission_failed(record_path, error="sbatch failed")
            self.assertEqual(record["submission_status"], "failed")
            self.assertEqual(record["dag_status"], "incomplete")
            self.assertEqual(record["submission_error"], "sbatch failed")

    def test_record_job_is_idempotent_for_latest_same_phase_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary) / "scratch"
            scratch.mkdir()
            record_path = self.initialize(scratch, phases=("one",))
            ops.record_job(record_path, phase="one", job_id="101")
            record = ops.record_job(record_path, phase="one", job_id="101")
            self.assertEqual(len(record["phases"]["one"]["attempts"]), 1)


class SlurmTests(unittest.TestCase):
    def test_classifies_active_success_and_common_failures(self):
        for state in sorted(ops.ACTIVE_STATES):
            with self.subTest(state=state):
                self.assertEqual(
                    ops.classify_slurm_state(state, "0:0")["classification"],
                    "active",
                )
        self.assertEqual(
            ops.classify_slurm_state("COMPLETED", "0:0")["classification"],
            "succeeded",
        )
        self.assertEqual(
            ops.classify_slurm_state("COMPLETED", "1:0")["classification"],
            "failed",
        )
        required_failures = {
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "NODE_FAIL",
            "OUT_OF_MEMORY",
            "PREEMPTED",
            "BOOT_FAIL",
            "DEADLINE",
            "REVOKED",
        }
        for state in sorted(required_failures):
            with self.subTest(state=state):
                self.assertEqual(
                    ops.classify_slurm_state(state, "0:0")["classification"],
                    "failed",
                )

    def test_parse_matches_exact_ids_and_fails_closed_for_missing_unknown(self):
        parsed = ops.parse_sacct_output(
            "123|COMPLETED|0:0\n"
            "123.batch|FAILED|1:0\n"
            "456|MYSTERY|0:0\n",
            ["123", "456", "789"],
        )
        self.assertEqual(parsed["123"]["classification"], "succeeded")
        self.assertEqual(parsed["456"]["classification"], "unresolved")
        self.assertEqual(parsed["456"]["state"], "MYSTERY")
        self.assertEqual(parsed["789"]["classification"], "unresolved")
        self.assertEqual(parsed["789"]["state"], "MISSING")

    def test_reconciliation_records_exact_status_and_active_resume_refuses(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary) / "scratch"
            scratch.mkdir()
            record_path = RecordTests().initialize(scratch)
            ops.record_job(record_path, phase="one", job_id="101")
            ops.record_job(record_path, phase="two", job_id="102")
            ops.mark_submitted(record_path)

            def runner(args, **kwargs):
                self.assertIn("101,102", args)
                return subprocess.CompletedProcess(
                    args,
                    0,
                    stdout="101|COMPLETED|0:0\n102|RUNNING|0:0\n",
                    stderr="",
                )

            record = ops.reconcile_record(record_path, runner=runner)
            self.assertEqual(record["dag_status"], "active")
            self.assertEqual(
                record["phases"]["one"]["attempts"][-1]["classification"],
                "succeeded",
            )
            with self.assertRaises(ops.ActiveJobsError):
                ops.determine_resume_start_index(record)


class ResumeTests(unittest.TestCase):
    def test_resume_refuses_unresolved_accounting_state(self):
        record = {
            "schema_version": ops.SCHEMA_VERSION,
            "submission_status": "submitted",
            "dag_status": "unresolved",
            "phase_order": ["one"],
            "phases": {
                "one": {
                    "attempts": [
                        {
                            "attempt": 1,
                            "job_id": "1",
                            "classification": "unresolved",
                        }
                    ]
                }
            },
        }
        with self.assertRaisesRegex(ops.ActiveJobsError, "unresolved"):
            ops.determine_resume_start_index(record)

    def test_resume_chooses_first_missing_or_unsuccessful_and_preserves_attempts(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary) / "scratch"
            scratch.mkdir()
            record_path = RecordTests().initialize(
                scratch, phases=("one", "two", "three")
            )
            ops.record_job(record_path, phase="one", job_id="101")
            ops.record_job(record_path, phase="two", job_id="102")
            record = ops.read_json(record_path)
            record["phases"]["one"]["attempts"][-1]["classification"] = "succeeded"
            record["phases"]["two"]["attempts"][-1]["classification"] = "failed"
            ops.atomic_write_json(record_path, record)

            self.assertEqual(ops.determine_resume_start_index(record), 1)
            before = list(record["phases"]["two"]["attempts"])
            updated = ops.record_job(record_path, phase="two", job_id="202")
            self.assertEqual(updated["phases"]["two"]["attempts"][:-1], before)
            self.assertEqual(updated["phases"]["two"]["attempts"][-1]["attempt"], 2)

    def test_resume_uses_validation_phase_after_all_jobs_succeed(self):
        record = {
            "schema_version": ops.SCHEMA_VERSION,
            "submission_status": "submitted",
            "dag_status": "failed",
            "phase_order": ["publish", "stage2-smoke"],
            "phases": {
                "publish": {
                    "attempts": [
                        {"attempt": 1, "job_id": "1", "classification": "succeeded"}
                    ]
                },
                "stage2-smoke": {
                    "attempts": [
                        {"attempt": 1, "job_id": "2", "classification": "succeeded"}
                    ]
                },
            },
            "success_validation": {
                "ok": False,
                "resume_phase": "stage2-smoke",
                "error": "bad smoke marker",
            },
        }
        self.assertEqual(ops.determine_resume_start_index(record), 1)


class SuccessValidationTests(unittest.TestCase):
    def make_success_tree(self, root):
        build = root / "build"
        artifact = root / "artifact"
        smoke = build / ops.SMOKE_DIRECTORY
        build.mkdir()
        artifact.mkdir()
        smoke.mkdir()

        manifest = b'{"artifact_type":"oprd_bridge"}\n'
        (artifact / "manifest.json").write_bytes(manifest)
        digest = hashlib.sha256(manifest).hexdigest()
        (artifact / "COMPLETE").write_text(
            f"manifest_sha256 {digest}\n", encoding="ascii"
        )
        verify_path = build / "verify.json"
        ops.atomic_write_json(
            verify_path,
            {
                "status": "ok",
                "checked_phases": list(ops.VERIFY_CHECKED_PHASES),
                "artifact_dir": os.fspath(artifact),
                "artifact_manifest_sha256": digest,
            },
        )
        evidence = build / "phase-evidence"
        evidence.mkdir()
        phases = {}
        for phase in ops.BUILDER_REQUIRED_PHASES:
            if phase == "publish":
                output = artifact
            elif phase == "verify":
                output = verify_path
            else:
                output = evidence / f"{phase}.txt"
                output.write_text(phase, encoding="utf-8")
            phases[phase] = {
                "status": "complete",
                "outputs": [
                    {
                        "path": os.fspath(output),
                        "sha256": ops._sha256_tree(output),
                    }
                ],
            }
        ops.atomic_write_json(
            build / "state.json", {"schema_version": 1, "phases": phases}
        )

        stage2_identity = {
            "prompts": "BytedTsinghua-SIA/DAPO-Math-17k",
            "dataset_revision": "4" * 40,
            "dataset_split": "train",
            "prompt_key": "prompt",
            "n_prompts": 8,
            "seed": 0,
            "rollout_tokens": 4096,
            "max_tokens": 512,
            "supervise_last_k": 2000,
            "require_finished": True,
            "rollout_refresh": 0,
            "batch_tokens": 2048,
            "learning_rate": 1e-5,
            "bridge_loss_coefficient": 10.0,
            "backend": "vllm",
        }
        ops.atomic_write_json(
            smoke / "results.json",
            {
                "losses": [1.0, 0.5],
                "start_step": 0,
                "resumed_from": None,
                "bridge_reference": {
                    "manifest_sha256": digest,
                    "stage2_identity": stage2_identity,
                },
            },
        )
        checkpoint = smoke / "train_state_last.pt"
        checkpoint.write_bytes(b"opaque checkpoint bytes")
        results_hash = hashlib.sha256((smoke / "results.json").read_bytes()).hexdigest()
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        ops.atomic_write_json(
            smoke / ops.SMOKE_COMPLETE_FILENAME,
            {
                "schema_version": 2,
                "status": "ok",
                "artifact_manifest_sha256": digest,
                "stage2_identity_sha256": hashlib.sha256(
                    json.dumps(
                        stage2_identity, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest(),
                "fresh_uninterrupted": True,
                "state_step": 2,
                "student_tensors_finite": True,
                "results_sha256": results_hash,
                "train_state_last_sha256": checkpoint_hash,
            },
        )
        return build, artifact

    def test_valid_success_tree_passes_without_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            build, artifact = self.make_success_tree(pathlib.Path(temporary))
            result = ops.validate_success(build, artifact)
            self.assertTrue(result["ok"])

    def test_existing_artifact_claim_requires_matching_builder_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            build, artifact = self.make_success_tree(root)
            record = root / "records" / "run.json"
            ops.initialize_record(
                scratch=root,
                record_path=record,
                run_id="run-1",
                build_dir=build,
                artifact_dir=artifact,
                phase_order=("stage2-smoke",),
            )
            claimed = ops.claim_artifact(
                scratch=root,
                artifact_dir=artifact,
                run_id="run-1",
                record_path=record,
                allow_existing_artifact=True,
                build_dir=build,
            )
            self.assertTrue(claimed["ok"])

    def test_success_rejects_artifact_and_smoke_inconsistency(self):
        with tempfile.TemporaryDirectory() as temporary:
            build, artifact = self.make_success_tree(pathlib.Path(temporary))
            (artifact / "COMPLETE").write_text(
                f"manifest_sha256 {'0' * 64}\n", encoding="ascii"
            )
            with self.assertRaisesRegex(
                ops.SuccessValidationError, "output checksum failed"
            ) as caught:
                ops.validate_success(build, artifact)
            self.assertEqual(caught.exception.resume_phase, "publish")

        with tempfile.TemporaryDirectory() as temporary:
            build, artifact = self.make_success_tree(pathlib.Path(temporary))
            results = build / ops.SMOKE_DIRECTORY / "results.json"
            results.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                ops.SuccessValidationError, "results hash does not match"
            ) as caught:
                ops.validate_success(build, artifact)
            self.assertEqual(caught.exception.resume_phase, "stage2-smoke")


if __name__ == "__main__":
    unittest.main()
