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
        (bin_dir / "gcc").write_text(
            "#!/bin/bash\n"
            "if [[ ${1:-} == --version ]]; then echo 'gcc test'; exit 0; fi\n"
            "while [[ $# -gt 0 ]]; do\n"
            "  if [[ $1 == -o ]]; then shift; printf x > \"$1\"; exit 0; fi\n"
            "  shift\n"
            "done\n"
            "exit 1\n",
            encoding="utf-8",
        )
        (bin_dir / "ld").write_text(
            "#!/bin/bash\necho 'GNU ld test'\n", encoding="utf-8"
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
        for name in ("ffmpeg", "gcc", "ld", "quota", "sacct", "sbatch"):
            (bin_dir / name).chmod(0o755)
        return scratch, cluster, fake_python, modules, bin_dir, counter, log

    def test_setup_environment_caps_setuptools_and_checks_dependencies(self):
        script = (SOURCE_NEXUS / "setup_env.sbatch").read_text()
        install_index = script.index('"setuptools<82"')
        check_index = script.index('"$VENV/bin/python" -m pip check')
        marker_index = script.index('  "$CLUSTER_DIR/requirements.txt" "$ENV_READY"')
        self.assertLess(install_index, check_index)
        self.assertLess(check_index, marker_index)
        self.assertIn('            "setuptools",', script)

    def test_bridge_uses_shared_managed_python(self):
        setup = (SOURCE_NEXUS / "setup_env.sbatch").read_text()
        self.assertIn('MANAGED_PYTHON_VERSION=3.12.7', setup)
        self.assertIn('"uv==0.9.8"', setup)
        self.assertIn('UV_PYTHON_INSTALL_DIR', setup)
        self.assertIn('--no-python-downloads python find', setup)
        self.assertIn('SLURM_RESTART_COUNT', setup)
        self.assertIn('venv.pre-managed.$ATTEMPT_TAG', setup)
        self.assertIn('.oprd-env-ready.pre-setup.$ATTEMPT_TAG.json', setup)
        self.assertLess(
            setup.index('.oprd-env-ready.pre-setup.$ATTEMPT_TAG.json'),
            setup.index('"$VENV/bin/python" -m pip install --index-url'),
        )
        self.assertIn('"managed_python_executable"', setup)
        self.assertIn('"imageio-ffmpeg==0.6.0"', setup)
        self.assertIn('imageio_ffmpeg.get_ffmpeg_exe()', setup)
        self.assertIn('BRIDGE_RUNTIME_PATH=', setup)
        self.assertIn('VLLM_CACHE_ROOT=', setup)
        self.assertIn('TRITON_CACHE_DIR=', setup)
        self.assertIn('check-toolchain --scratch "$SCRATCH_ROOT"', setup)
        self.assertIn('"$VENV/bin/python" "$OPS" check-paths', setup)
        self.assertLess(
            setup.index('"$VENV/bin/python" "$OPS" check-paths'),
            setup.index('mkdir -p "$BRIDGE_HF_HOME"'),
        )
        self.assertNotIn('module load ffmpeg/', setup)

        runtime_scripts = (
            "submit_oprd_bridge.sh",
            "oprd_bridge_corpus.sbatch",
            "oprd_bridge_teacher.sbatch",
            "oprd_bridge_student_groups.sbatch",
            "oprd_bridge_publish.sbatch",
            "oprd_bridge_stage2_smoke.sbatch",
        )
        for name in runtime_scripts:
            with self.subTest(script=name):
                script = (SOURCE_NEXUS / name).read_text()
                self.assertNotIn("module load Python3/", script)
                self.assertNotIn("module load ffmpeg/", script)
                self.assertIn('export PATH="$BRIDGE_RUNTIME_PATH"', script)
                self.assertIn("VLLM_CACHE_ROOT", script)
                self.assertIn("TRITON_CACHE_DIR", script)
                if name != "submit_oprd_bridge.sh":
                    self.assertIn(
                        'check-toolchain --scratch "$SCRATCH_ROOT"', script
                    )
                self.assertIn("set +u", script)

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
                "BRIDGE_RUNTIME_PATH": f"{bin_dir}:{os.environ['PATH']}",
                "VLLM_CACHE_ROOT": str(scratch / "vllm-cache"),
                "TRITON_CACHE_DIR": str(scratch / "triton-cache"),
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
            self.assertTrue(
                all(
                    f"PATH={environment['BRIDGE_RUNTIME_PATH']}" in line
                    for line in lines
                )
            )
            self.assertTrue(
                all(
                    f"VLLM_CACHE_ROOT={environment['VLLM_CACHE_ROOT']}" in line
                    for line in lines
                )
            )
            self.assertTrue(
                all(
                    f"TRITON_CACHE_DIR={environment['TRITON_CACHE_DIR']}" in line
                    for line in lines
                )
            )
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
                record["configuration"]["runtime_path"],
                environment["BRIDGE_RUNTIME_PATH"],
            )
            self.assertEqual(
                record["configuration"]["vllm_cache_root"],
                environment["VLLM_CACHE_ROOT"],
            )
            self.assertEqual(
                record["configuration"]["triton_cache_dir"],
                environment["TRITON_CACHE_DIR"],
            )
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

    def test_submission_rejects_cache_path_with_export_delimiter(self):
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
                "BRIDGE_RUNTIME_PATH": f"{bin_dir}:{os.environ['PATH']}",
                "VLLM_CACHE_ROOT": str(scratch / "bad,cache"),
                "TRITON_CACHE_DIR": str(scratch / "triton-cache"),
                "BRIDGE_RUN_ID": "delimiter-run",
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
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "VLLM_CACHE_ROOT cannot contain commas or newlines",
                completed.stderr,
            )
            self.assertEqual(counter.read_text(encoding="ascii"), "0\n")
            self.assertFalse(
                (cluster / "job_ids" / "oprd_bridge_delimiter-run.json").exists()
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
                "BRIDGE_RUNTIME_PATH": f"{bin_dir}:{os.environ['PATH']}",
                "VLLM_CACHE_ROOT": str(scratch / "vllm-cache"),
                "TRITON_CACHE_DIR": str(scratch / "triton-cache"),
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
