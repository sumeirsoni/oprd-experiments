import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest


REPOSITORY = pathlib.Path(__file__).resolve().parent.parent
SOURCE_NEXUS = REPOSITORY / "cluster" / "nexus"
SUBMITTER = SOURCE_NEXUS / "submit_oprd_bridge.sh"


class SubmitBridgeTests(unittest.TestCase):
    def make_cluster_tree(self, root):
        scratch = root / "scratch"
        cluster = scratch / "oprd-experiments" / "cluster"
        nexus = cluster / "nexus"
        bin_dir = root / "bin"
        nexus.mkdir(parents=True)
        bin_dir.mkdir()
        (cluster / "requirements.txt").write_text("test-only\n", encoding="utf-8")
        for name in ("build_oprd_bridge.py", "oprd_bridge.py", "train_harness.py"):
            (cluster / name).write_text(f"# test {name}\n", encoding="utf-8")
        shutil.copy2(SOURCE_NEXUS / "oprd_bridge_ops.py", nexus / "oprd_bridge_ops.py")
        shutil.copy2(SUBMITTER, nexus / "submit_oprd_bridge.sh")
        for name in (
            "oprd_bridge_corpus.sbatch",
            "oprd_bridge_teacher.sbatch",
            "oprd_bridge_student_groups.sbatch",
            "oprd_bridge_publish.sbatch",
            "oprd_bridge_stage2_smoke.sbatch",
        ):
            shutil.copy2(SOURCE_NEXUS / name, nexus / name)

        fake_python = bin_dir / "python3.12"
        fake_python.write_text(
            "#!/bin/bash\n"
            "if [[ ${1:-} == --version ]]; then echo 'Python 3.12.9'; exit 0; fi\n"
            "if [[ ${1:-} == -c && ${2:-} == *importlib.metadata* ]]; then echo '{\"torch\":\"test\"}'; exit 0; fi\n"
            "if [[ ${1:-} == -c && ${2:-} == import\\ importlib,* ]]; then exit 0; fi\n"
            f"exec {shlex_quote(sys.executable)} \"$@\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        venv = scratch / "oprd-experiments" / "venv"
        venv.mkdir(parents=True)
        marker = venv / ".oprd-env-ready.json"
        requirements = cluster / "requirements.txt"
        marker.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "python_executable": str(fake_python.resolve()),
                    "requirements_sha256": hashlib.sha256(
                        requirements.read_bytes()
                    ).hexdigest(),
                    "packages": {"torch": "test"},
                    "ffmpeg_path": str((bin_dir / "ffmpeg").resolve()),
                    "ffmpeg_version": "ffmpeg version test",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        modules = root / "modules.sh"
        modules.write_text("module() { return 0; }\n", encoding="utf-8")
        (bin_dir / "ffmpeg").write_text(
            "#!/bin/bash\necho 'ffmpeg version test'\n", encoding="utf-8"
        )
        (bin_dir / "sacct").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
        (bin_dir / "quota").write_text(
            "#!/bin/bash\necho 'manual quota report'\n", encoding="utf-8"
        )
        counter = root / "sbatch.counter"
        log = root / "sbatch.log"
        counter.write_text("0\n", encoding="ascii")
        (bin_dir / "sbatch").write_text(
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            "n=$(<\"$SBATCH_COUNTER\")\n"
            "n=$((n + 1))\n"
            "printf '%s\\n' \"$n\" > \"$SBATCH_COUNTER\"\n"
            "printf '%s\\n' \"$*\" >> \"$SBATCH_LOG\"\n"
            "printf '%s\\n' \"$((1000 + n))\"\n",
            encoding="utf-8",
        )
        for name in ("ffmpeg", "quota", "sacct", "sbatch"):
            (bin_dir / name).chmod(0o755)
        return scratch, cluster, fake_python, modules, bin_dir, counter, log

    def test_stage2_smoke_uses_validated_no_replace_publication(self):
        script = (SOURCE_NEXUS / "oprd_bridge_stage2_smoke.sbatch").read_text()
        self.assertIn("SLURM_RESTART_COUNT", script)
        self.assertIn("renameat2", script)
        self.assertIn("RENAME_NOREPLACE", script)
        self.assertIn("validate-success", script)
        self.assertIn("start_step", script)
        self.assertIn("resumed_from", script)
        self.assertIn("--require-finished", script)
        self.assertIn("artifact_contract != contract", script)
        self.assertNotIn("SMOKE_STAGING=${SMOKE_STAGING", script)

    def test_fresh_submission_records_exact_ordered_eleven_job_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (
                scratch,
                cluster,
                fake_python,
                modules,
                bin_dir,
                counter,
                log,
            ) = self.make_cluster_tree(root)
            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "USER": "test-user",
                "SCRATCH_ROOT": str(scratch),
                "ROOT": str(scratch / "oprd-experiments"),
                "PYTHON": str(fake_python),
                "MODULES_INIT": str(modules),
                "BRIDGE_RUN_ID": "unit-test-run",
                "SUBMIT_MIN_FREE_GIB": "0",
                "SUBMIT_MIN_FREE_INODES": "0",
                "ALLOW_QUOTA_UNAVAILABLE": "1",
                "SBATCH_COUNTER": str(counter),
                "SBATCH_LOG": str(log),
            }
            completed = subprocess.run(
                ["bash", str(SUBMITTER)],
                cwd=REPOSITORY,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            lines = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 11)
            self.assertTrue(all("--export=ALL" not in line for line in lines))
            self.assertTrue(all("--comment=oprd:unit-test-run:" in line for line in lines))
            self.assertNotIn("--dependency", lines[0])
            for index, line in enumerate(lines[1:], 1):
                self.assertIn(f"--dependency=afterok:{1000 + index}", line)
            for group_index, line in enumerate(lines[2:9]):
                self.assertIn(f"GROUP_INDEX={group_index}", line)
            self.assertIn("oprd_bridge_publish.sbatch", lines[9])
            self.assertIn("oprd_bridge_stage2_smoke.sbatch", lines[10])

            record_path = cluster / "job_ids" / "oprd_bridge_unit-test-run.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["submission_status"], "submitted")
            self.assertEqual(record["dag_status"], "active")
            self.assertEqual(
                record["phase_order"],
                [
                    "corpus",
                    "teacher",
                    *[f"student-group-{index}" for index in range(7)],
                    "publish",
                    "stage2-smoke",
                ],
            )
            self.assertTrue(
                all(
                    len(record["phases"][phase]["attempts"]) == 1
                    for phase in record["phase_order"]
                )
            )
            self.assertEqual(record["configuration"]["hf_home"], str(scratch / "hf-cache"))
            self.assertEqual(
                record["phases"]["teacher"]["attempts"][0][
                    "dependency_job_id"
                ],
                "1001",
            )
            self.assertTrue(
                record["phases"]["publish"]["attempts"][0]["script"].endswith(
                    "oprd_bridge_publish.sbatch"
                )
            )
            self.assertTrue(pathlib.Path(record["claim_dir"]).is_dir())

            ops_path = cluster / "nexus" / "oprd_bridge_ops.py"
            begun = subprocess.run(
                [
                    str(fake_python),
                    str(ops_path),
                    "begin-job",
                    "--record",
                    str(record_path),
                    "--phase",
                    "stage2-smoke",
                    "--script",
                    str(cluster / "nexus" / "oprd_bridge_stage2_smoke.sbatch"),
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(begun.returncode, 0, begun.stderr)
            adopted = subprocess.run(
                [
                    "bash",
                    str(SUBMITTER),
                    "--adopt-job",
                    str(record_path),
                    "stage2-smoke",
                    "2000",
                ],
                cwd=REPOSITORY,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(adopted.returncode, 0, adopted.stdout + adopted.stderr)
            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertNotIn("pending_submission", record["phases"]["stage2-smoke"])
            self.assertEqual(
                record["phases"]["stage2-smoke"]["attempts"][-1]["job_id"],
                "2000",
            )

    def test_resume_submits_only_failed_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            (
                scratch,
                cluster,
                fake_python,
                modules,
                bin_dir,
                counter,
                log,
            ) = self.make_cluster_tree(root)
            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "USER": "test-user",
                "SCRATCH_ROOT": str(scratch),
                "ROOT": str(scratch / "oprd-experiments"),
                "PYTHON": str(fake_python),
                "MODULES_INIT": str(modules),
                "BRIDGE_RUN_ID": "resume-run",
                "SUBMIT_MIN_FREE_GIB": "0",
                "SUBMIT_MIN_FREE_INODES": "0",
                "ALLOW_QUOTA_UNAVAILABLE": "1",
                "SBATCH_COUNTER": str(counter),
                "SBATCH_LOG": str(log),
            }
            first = subprocess.run(
                ["bash", str(SUBMITTER)],
                cwd=REPOSITORY,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            record_path = cluster / "job_ids" / "oprd_bridge_resume-run.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            pathlib.Path(record["build_dir"]).mkdir(parents=True)

            sacct_lines = []
            for job_id in range(1001, 1012):
                state = "COMPLETED" if job_id <= 1004 else "CANCELLED"
                exit_code = "0:0" if state == "COMPLETED" else "1:0"
                if job_id == 1005:
                    state = "FAILED"
                sacct_lines.append(f"{job_id}|{state}|{exit_code}")
            (bin_dir / "sacct").write_text(
                "#!/bin/bash\n"
                "cat <<'EOF'\n"
                + "\n".join(sacct_lines)
                + "\nEOF\n",
                encoding="utf-8",
            )
            (bin_dir / "sacct").chmod(0o755)

            resumed = subprocess.run(
                ["bash", str(SUBMITTER), "--resume", str(record_path)],
                cwd=REPOSITORY,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            lines = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 18)
            resumed_lines = lines[11:]
            self.assertEqual(len(resumed_lines), 7)
            self.assertNotIn("--dependency", resumed_lines[0])
            self.assertIn("GROUP_INDEX=2", resumed_lines[0])
            for offset, line in enumerate(resumed_lines[1:], 1):
                self.assertIn(f"--dependency=afterok:{1011 + offset}", line)

            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(
                len(record["phases"]["student-group-2"]["attempts"]), 2
            )
            self.assertEqual(
                record["phases"]["student-group-2"]["attempts"][-1]["job_id"],
                "1012",
            )
            self.assertEqual(
                record["phases"]["stage2-smoke"]["attempts"][-1]["job_id"],
                "1018",
            )


def shlex_quote(value):
    import shlex

    return shlex.quote(str(value))


if __name__ == "__main__":
    unittest.main()
