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
SOURCE_CLUSTER = REPOSITORY / "cluster"
SOURCE_NEXUS = SOURCE_CLUSTER / "nexus"
SUBMITTER = SOURCE_NEXUS / "submit_oprd_bridge.sh"
TIER2_WRAPPER = SOURCE_NEXUS / "tier2_bridge_bakeoff.sbatch"


class SubmitBridgeTests(unittest.TestCase):
    def make_cluster_tree(self, root):
        scratch = root / "scratch"
        cluster = scratch / "oprd-experiments" / "cluster"
        nexus = cluster / "nexus"
        bin_dir = root / "bin"
        cuda_home = root / "cuda-12.8.1"
        cuda_bin = cuda_home / "bin"
        cuda_lib64 = cuda_home / "lib64"
        nexus.mkdir(parents=True)
        bin_dir.mkdir()
        cuda_bin.mkdir(parents=True)
        cuda_lib64.mkdir()
        (cluster / "requirements.txt").write_text("test-only\n", encoding="utf-8")
        for name in ("build_oprd_bridge.py", "oprd_bridge.py", "train_harness.py"):
            (cluster / name).write_text(f"# test {name}\n", encoding="utf-8")
        shutil.copy2(
            SOURCE_CLUSTER / "oprd_atomic_directory.py",
            cluster / "oprd_atomic_directory.py",
        )
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

        libcudart_path = (cuda_lib64 / "libcudart.so.12").resolve()
        fake_python = bin_dir / "python3.12"
        fake_python.write_text(
            "#!/bin/bash\n"
            "if [[ ${1:-} == --version ]]; then echo 'Python 3.12.9'; exit 0; fi\n"
            "if [[ ${1:-} == -c && ${2:-} == *importlib.metadata* ]]; then echo '{\"flashinfer-python\":\"test\",\"torch\":\"test\"}'; exit 0; fi\n"
            f"if [[ ${{1:-}} == -c && ${{2:-}} == *mapped_libcudart_paths* ]]; then printf '%s\\n' {shlex_quote(json.dumps({'mapped_libcudart_paths': [str(libcudart_path)]}))}; exit 0; fi\n"
            "if [[ ${1:-} == -c && ${2:-} == import\\ importlib,* ]]; then exit 0; fi\n"
            f"exec {shlex_quote(sys.executable)} \"$@\"\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        nvcc = cuda_bin / "nvcc"
        nvcc.write_text(
            "#!/bin/bash\n"
            "if [[ ${1:-} == --version ]]; then\n"
            "  echo 'Cuda compilation tools, release 12.8, V12.8.93'\n"
            "  exit 0\n"
            "fi\n"
            "while [[ $# -gt 0 ]]; do\n"
            "  if [[ $1 == -o ]]; then shift; printf x > \"$1\"; exit 0; fi\n"
            "  shift\n"
            "done\n"
            "exit 1\n",
            encoding="utf-8",
        )
        nvcc.chmod(0o755)
        libcudart = cuda_lib64 / "libcudart.so.12"
        libcudart.write_bytes(b"test libcudart")

        venv = scratch / "oprd-experiments" / "venv"
        venv.mkdir(parents=True)
        marker = venv / ".oprd-env-ready.json"
        requirements = cluster / "requirements.txt"
        marker.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "python_executable": str(fake_python.resolve()),
                    "requirements_sha256": hashlib.sha256(
                        requirements.read_bytes()
                    ).hexdigest(),
                    "packages": {
                        "flashinfer-python": "test",
                        "torch": "test",
                    },
                    "ffmpeg_path": str((bin_dir / "ffmpeg").resolve()),
                    "ffmpeg_version": "ffmpeg version test",
                    "cuda_home": str(cuda_home.resolve()),
                    "cuda_nvcc": str(nvcc.resolve()),
                    "cuda_nvcc_version": (
                        "Cuda compilation tools, release 12.8, V12.8.93"
                    ),
                    "cuda_nvcc_sha256": hashlib.sha256(
                        nvcc.read_bytes()
                    ).hexdigest(),
                    "runtime_ld_library_path": str(cuda_lib64.resolve()),
                    "libcudart_path": str(libcudart.resolve()),
                    "libcudart_sha256": hashlib.sha256(
                        libcudart.read_bytes()
                    ).hexdigest(),
                    "flashinfer_workspace_base": str(
                        (scratch / "flashinfer-workspace").resolve()
                    ),
                    "flashinfer_package_version": "test",
                    "flashinfer_runtime": {
                        "ok": True,
                        "paths": {
                            "flashinfer_cache_dir": str(
                                (
                                    scratch
                                    / "flashinfer-workspace"
                                    / ".cache"
                                    / "flashinfer"
                                ).resolve()
                            ),
                            "flashinfer_workspace_dir": str(
                                (
                                    scratch
                                    / "flashinfer-workspace"
                                    / ".cache"
                                    / "flashinfer"
                                    / "test"
                                    / "90a"
                                ).resolve()
                            ),
                            "flashinfer_jit_dir": str(
                                (
                                    scratch
                                    / "flashinfer-workspace"
                                    / ".cache"
                                    / "flashinfer"
                                    / "test"
                                    / "90a"
                                    / "cached_ops"
                                ).resolve()
                            ),
                            "flashinfer_gen_src_dir": str(
                                (
                                    scratch
                                    / "flashinfer-workspace"
                                    / ".cache"
                                    / "flashinfer"
                                    / "test"
                                    / "90a"
                                    / "generated"
                                ).resolve()
                            ),
                        },
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

        modules = root / "modules.sh"
        modules.write_text(
            "module() {\n"
            "  if [[ ${1:-} == load && ${2:-} == cuda/12.8.1 ]]; then\n"
            f"    export CUDA_HOME={shlex_quote(cuda_home.resolve())}\n"
            "  fi\n"
            "  return 0\n"
            "}\n",
            encoding="utf-8",
        )
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
        (bin_dir / "readelf").write_text(
            "#!/bin/bash\n"
            "echo ' 0x0000000000000001 (NEEDED) Shared library: [libcudart.so.12]'\n",
            encoding="utf-8",
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
        for name in ("ffmpeg", "gcc", "ld", "readelf", "quota", "sacct", "sbatch"):
            (bin_dir / name).chmod(0o755)
        return scratch, cluster, fake_python, modules, bin_dir, counter, log

    def test_setup_environment_caps_setuptools_and_checks_dependencies(self):
        script = (SOURCE_NEXUS / "setup_env.sbatch").read_text()
        install_index = script.index('"setuptools<82"')
        check_index = script.index('"$VENV/bin/python" -m pip check')
        marker_index = script.index('  "$CLUSTER_DIR/requirements.txt" "$ENV_READY"')
        self.assertLess(install_index, check_index)
        self.assertLess(check_index, marker_index)
        self.assertIn('        "setuptools",', script)

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
        self.assertIn('FLASHINFER_WORKSPACE_BASE=', setup)
        self.assertIn('BRIDGE_RUNTIME_LD_LIBRARY_PATH=', setup)
        self.assertIn('export LD_LIBRARY_PATH="$BRIDGE_RUNTIME_LD_LIBRARY_PATH"', setup)
        self.assertIn('check-toolchain --scratch "$SCRATCH_ROOT"', setup)
        self.assertIn('check-cuda-runtime', setup)
        self.assertIn('"schema_version": 2', setup)
        self.assertIn('"flashinfer-python",', setup)
        self.assertIn('get_sampling_module()', setup)
        self.assertNotIn('module purge', setup)
        self.assertNotIn('module reset', setup)
        self.assertIn('"$VENV/bin/python" "$OPS" check-paths', setup)
        self.assertLess(
            setup.index('"$VENV/bin/python" "$OPS" check-paths'),
            setup.index('mkdir -p "$BRIDGE_HF_HOME"'),
        )
        self.assertIn('--path "$FLASHINFER_WORKSPACE_BASE"', setup)
        self.assertLess(
            setup.index('export VLLM_CACHE_ROOT TRITON_CACHE_DIR FLASHINFER_WORKSPACE_BASE'),
            setup.index('from flashinfer.sampling import get_sampling_module'),
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
                self.assertIn("FLASHINFER_WORKSPACE_BASE", script)
                self.assertIn("BRIDGE_CUDA_HOME", script)
                self.assertIn("BRIDGE_CUDA_NVCC", script)
                self.assertIn("BRIDGE_RUNTIME_LD_LIBRARY_PATH", script)
                self.assertIn('export CUDA_HOME="$BRIDGE_CUDA_HOME"', script)
                self.assertIn(
                    'export LD_LIBRARY_PATH="$BRIDGE_RUNTIME_LD_LIBRARY_PATH"',
                    script,
                )
                self.assertNotIn("module purge", script)
                self.assertNotIn("module reset", script)
                if name != "submit_oprd_bridge.sh":
                    self.assertIn(
                        'check-toolchain --scratch "$SCRATCH_ROOT"', script
                    )
                    self.assertIn('--cuda-home "$BRIDGE_CUDA_HOME"', script)
                    self.assertIn('--cuda-nvcc "$BRIDGE_CUDA_NVCC"', script)
                    self.assertIn(
                        '--runtime-ld-library-path "$BRIDGE_RUNTIME_LD_LIBRARY_PATH"',
                        script,
                    )
                    self.assertIn(
                        '--flashinfer-workspace-base "$FLASHINFER_WORKSPACE_BASE"',
                        script,
                    )
                    self.assertLess(
                        script.index('"$OPS" check-runtime'),
                        script.index('"$OPS" check-environment'),
                    )
                self.assertIn("set +u", script)

    def test_tier2_wrapper_freezes_the_paper_matrix(self):
        script = TIER2_WRAPPER.read_text()
        self.assertIn("#SBATCH --time=72:00:00", script)
        self.assertIn("#SBATCH --array=0-26%4", script)
        self.assertIn("#SBATCH --export=NONE", script)
        self.assertIn("/cmlscratch/%u/oprd-experiments-main/cluster/logs", script)
        self.assertIn('readonly ROOT="$SCRATCH_ROOT/oprd-experiments-main"', script)
        self.assertIn('export HF_HOME="$SCRATCH_ROOT/hf-cache"', script)
        self.assertIn('export XDG_CACHE_HOME="$SCRATCH_ROOT/vllm-cache"', script)
        self.assertIn('export VLLM_CACHE_ROOT="$SCRATCH_ROOT/vllm-cache"', script)
        self.assertIn("export COMPILER_PATH=/usr/bin", script)
        self.assertIn("/usr/bin/gcc -print-prog-name=ld", script)
        self.assertIn('export TRITON_CACHE_DIR="$SCRATCH_ROOT/triton-cache"', script)
        self.assertIn(
            'export FLASHINFER_WORKSPACE_BASE="$SCRATCH_ROOT/flashinfer-workspace"',
            script,
        )
        self.assertIn("[[ $# -eq 3 ]]", script)
        self.assertIn('"$PYTHON" "$OPS" validate-published', script)
        self.assertLess(
            script.index('"$PYTHON" "$OPS" validate-published'),
            script.index('srun "$PYTHON" train_harness.py'),
        )
        self.assertIn("depth_gram", script)
        self.assertIn("opd_topk", script)
        self.assertIn("untrained_baseline", script)
        self.assertIn("--steps 0", script)
        self.assertIn("--steps 2000", script)
        self.assertIn("--rollout-tokens 16000", script)
        self.assertIn("--supervise-last-k 2000", script)
        self.assertIn("--max-tokens 24000", script)
        self.assertIn('mkdir "$OUT" || fail', script)
        self.assertNotIn("SKIP_ARMS", script)
        self.assertNotIn("SKIP_PAIRS", script)
        self.assertNotIn("phi4mini", script)
        self.assertNotIn("rm -", script)
        self.assertNotIn("${STEPS:-", script)
        self.assertNotIn("${RESULT_NS:-", script)

    def test_stage2_smoke_uses_shared_atomic_directory_helper(self):
        script = (SOURCE_NEXUS / "oprd_bridge_stage2_smoke.sbatch").read_text()
        helper_move = '"$PYTHON" "$ATOMIC_DIRECTORY" move'
        self.assertIn("SLURM_RESTART_COUNT", script)
        self.assertIn("ATOMIC_DIRECTORY=$CLUSTER_DIR/oprd_atomic_directory.py", script)
        self.assertEqual(script.count(helper_move), 2)
        self.assertEqual(script.count('--trusted-root "$SCRATCH_ROOT"'), 2)
        self.assertIn('--source "$SMOKE_OUT"', script)
        self.assertIn('--destination "$SMOKE_QUARANTINE"', script)
        self.assertIn('--source "$SMOKE_STAGING"', script)
        self.assertIn('--destination "$SMOKE_OUT"', script)
        self.assertNotIn("renameat2", script)
        self.assertNotIn("RENAME_NOREPLACE", script)
        self.assertNotIn("ctypes", script)
        self.assertIn("start_step", script)
        self.assertIn("resumed_from", script)
        self.assertIn("--require-finished", script)
        self.assertIn("artifact_contract != contract", script)
        self.assertNotIn("SMOKE_STAGING=${SMOKE_STAGING", script)
        self.assertLess(
            script.index("STAGE2_SMOKE_ALREADY_VALID"),
            script.index("from flashinfer.sampling import get_sampling_module"),
        )
        final_move = script.rindex(helper_move)
        final_validation = script.rindex('"$PYTHON" "$OPS" validate-success')
        published = script.index("STAGE2_SMOKE_PUBLISHED")
        self.assertLess(final_move, final_validation)
        self.assertLess(final_validation, published)

    def test_submitter_probes_atomic_publish_before_any_mutation(self):
        script = SUBMITTER.read_text()
        self.assertIn('"$CLUSTER_DIR/oprd_atomic_directory.py"', script)
        self.assertIn('--script "$CLUSTER_DIR/oprd_atomic_directory.py"', script)
        self.assertIn('--publication-parent "$CLUSTER_DIR"', script)
        self.assertIn(
            '"$PYTHON" "$OPS" source-bundle-sha256 --root "$ROOT"',
            script,
        )
        self.assertNotIn("digest.update(len(name).to_bytes", script)
        preflight = script.index("load_runtime_modules\nrun_preflight\n")
        initialize = script.index('"$PYTHON" "$OPS" init-record')
        claim = script.index('"$PYTHON" "$OPS" claim', initialize)
        submit = script.index('JOB_ID=$(submit_job', claim)
        self.assertLess(preflight, initialize)
        self.assertLess(initialize, claim)
        self.assertLess(claim, submit)

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
                "BRIDGE_CUDA_HOME": str((root / "cuda-12.8.1").resolve()),
                "BRIDGE_CUDA_NVCC": str(
                    (root / "cuda-12.8.1" / "bin" / "nvcc").resolve()
                ),
                "BRIDGE_RUNTIME_LD_LIBRARY_PATH": str(
                    (root / "cuda-12.8.1" / "lib64").resolve()
                ),
                "VLLM_CACHE_ROOT": str(scratch / "vllm-cache"),
                "TRITON_CACHE_DIR": str(scratch / "triton-cache"),
                "FLASHINFER_WORKSPACE_BASE": str(
                    (scratch / "flashinfer-workspace").resolve()
                ),
                "FAKE_LIBCUDART": str(
                    (
                        root
                        / "cuda-12.8.1"
                        / "lib64"
                        / "libcudart.so.12"
                    ).resolve()
                ),
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
            self.assertTrue(
                all(
                    f"CUDA_HOME={environment['BRIDGE_CUDA_HOME']}" in line
                    and f"FLASHINFER_NVCC={environment['BRIDGE_CUDA_NVCC']}" in line
                    and (
                        "LD_LIBRARY_PATH="
                        f"{environment['BRIDGE_RUNTIME_LD_LIBRARY_PATH']}"
                    ) in line
                    and (
                        "FLASHINFER_WORKSPACE_BASE="
                        f"{environment['FLASHINFER_WORKSPACE_BASE']}"
                    ) in line
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
                record["configuration"]["cuda_home"],
                environment["BRIDGE_CUDA_HOME"],
            )
            self.assertEqual(
                record["configuration"]["cuda_nvcc"],
                environment["BRIDGE_CUDA_NVCC"],
            )
            self.assertEqual(
                record["configuration"]["runtime_ld_library_path"],
                environment["BRIDGE_RUNTIME_LD_LIBRARY_PATH"],
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
                record["configuration"]["flashinfer_workspace_base"],
                environment["FLASHINFER_WORKSPACE_BASE"],
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
            legacy_record = json.loads(record_path.read_text(encoding="utf-8"))
            for key in (
                "cuda_home",
                "cuda_nvcc",
                "runtime_ld_library_path",
                "flashinfer_workspace_base",
            ):
                legacy_record["configuration"].pop(key)
            record_path.write_text(
                json.dumps(legacy_record, sort_keys=True) + "\n",
                encoding="utf-8",
            )
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
            self.assertNotIn("cuda_home", record["configuration"])
            self.assertNotIn("flashinfer_workspace_base", record["configuration"])
            self.assertEqual(
                record["phases"]["stage2-smoke"]["attempts"][-1]["job_id"],
                "2000",
            )

    def test_atomic_publish_probe_failure_submits_no_jobs_or_claim(self):
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
            cuda_home = (root / "cuda-12.8.1").resolve()
            publication_parent = (
                cluster
                / "artifacts"
                / "oprd_bridge"
                / "qwen3_4b_to_1p7b"
                / "smoke"
            )
            cluster.chmod(0o777)
            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "USER": "test-user",
                "SCRATCH_ROOT": str(scratch),
                "ROOT": str(scratch / "oprd-experiments"),
                "PYTHON": str(fake_python),
                "MODULES_INIT": str(modules),
                "BRIDGE_RUNTIME_PATH": f"{bin_dir}:{os.environ['PATH']}",
                "BRIDGE_CUDA_HOME": str(cuda_home),
                "BRIDGE_CUDA_NVCC": str(cuda_home / "bin" / "nvcc"),
                "BRIDGE_RUNTIME_LD_LIBRARY_PATH": str(cuda_home / "lib64"),
                "VLLM_CACHE_ROOT": str(scratch / "vllm-cache"),
                "TRITON_CACHE_DIR": str(scratch / "triton-cache"),
                "FLASHINFER_WORKSPACE_BASE": str(
                    (scratch / "flashinfer-workspace").resolve()
                ),
                "FAKE_LIBCUDART": str(
                    (cuda_home / "lib64" / "libcudart.so.12").resolve()
                ),
                "BRIDGE_RUN_ID": "atomic-probe-failure",
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
            self.assertIn('"atomic_directory_publish"', completed.stdout)
            self.assertIn("group- or world-writable", completed.stdout)
            self.assertEqual(counter.read_text(encoding="ascii"), "0\n")
            self.assertFalse(log.exists())
            self.assertFalse(
                (
                    cluster
                    / "job_ids"
                    / "oprd_bridge_atomic-probe-failure.json"
                ).exists()
            )
            self.assertFalse(
                (publication_parent / ".rank8_all_layers.claim").exists()
            )

    def test_submission_rejects_export_delimiters_before_side_effects(self):
        cases = {
            "VLLM_CACHE_ROOT": "bad,cache",
            "FLASHINFER_WORKSPACE_BASE": "bad\nworkspace",
            "BRIDGE_RUNTIME_LD_LIBRARY_PATH": "bad,library-path",
            "BRIDGE_CUDA_HOME": "bad\ncuda-home",
            "BRIDGE_CUDA_NVCC": "bad,nvcc",
        }
        for field, invalid_value in cases.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
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
                cuda_home = (root / "cuda-12.8.1").resolve()
                environment = {
                    **os.environ,
                    "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "USER": "test-user",
                    "SCRATCH_ROOT": str(scratch),
                    "ROOT": str(scratch / "oprd-experiments"),
                    "PYTHON": str(fake_python),
                    "MODULES_INIT": str(modules),
                    "BRIDGE_RUNTIME_PATH": f"{bin_dir}:{os.environ['PATH']}",
                    "BRIDGE_CUDA_HOME": str(cuda_home),
                    "BRIDGE_CUDA_NVCC": str(cuda_home / "bin" / "nvcc"),
                    "BRIDGE_RUNTIME_LD_LIBRARY_PATH": str(cuda_home / "lib64"),
                    "VLLM_CACHE_ROOT": str(scratch / "vllm-cache"),
                    "TRITON_CACHE_DIR": str(scratch / "triton-cache"),
                    "FLASHINFER_WORKSPACE_BASE": str(
                        (scratch / "flashinfer-workspace").resolve()
                    ),
                    "FAKE_LIBCUDART": str(
                        (cuda_home / "lib64" / "libcudart.so.12").resolve()
                    ),
                    "BRIDGE_RUN_ID": "delimiter-run",
                    "SUBMIT_MIN_FREE_GIB": "0",
                    "SUBMIT_MIN_FREE_INODES": "0",
                    "ALLOW_QUOTA_UNAVAILABLE": "1",
                    "SBATCH_COUNTER": str(counter),
                    "SBATCH_LOG": str(log),
                }
                environment[field] = invalid_value
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
                    f"{field} cannot contain commas or newlines",
                    completed.stderr,
                )
                self.assertEqual(counter.read_text(encoding="ascii"), "0\n")
                self.assertFalse(
                    (
                        cluster
                        / "job_ids"
                        / "oprd_bridge_delimiter-run.json"
                    ).exists()
                )

    def test_legacy_record_supports_status_and_explicit_claim_release(self):
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
            cuda_home = (root / "cuda-12.8.1").resolve()
            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "USER": "test-user",
                "SCRATCH_ROOT": str(scratch),
                "ROOT": str(scratch / "oprd-experiments"),
                "PYTHON": str(fake_python),
                "MODULES_INIT": str(modules),
                "BRIDGE_RUNTIME_PATH": f"{bin_dir}:{os.environ['PATH']}",
                "BRIDGE_CUDA_HOME": str(cuda_home),
                "BRIDGE_CUDA_NVCC": str(cuda_home / "bin" / "nvcc"),
                "BRIDGE_RUNTIME_LD_LIBRARY_PATH": str(cuda_home / "lib64"),
                "VLLM_CACHE_ROOT": str(scratch / "vllm-cache"),
                "TRITON_CACHE_DIR": str(scratch / "triton-cache"),
                "FLASHINFER_WORKSPACE_BASE": str(
                    (scratch / "flashinfer-workspace").resolve()
                ),
                "BRIDGE_RUN_ID": "legacy-run",
                "SUBMIT_MIN_FREE_GIB": "0",
                "SUBMIT_MIN_FREE_INODES": "0",
                "ALLOW_QUOTA_UNAVAILABLE": "1",
                "SBATCH_COUNTER": str(counter),
                "SBATCH_LOG": str(log),
            }
            submitted = subprocess.run(
                ["bash", str(SUBMITTER)],
                cwd=REPOSITORY,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                submitted.returncode, 0, submitted.stdout + submitted.stderr
            )
            record_path = (
                cluster / "job_ids" / "oprd_bridge_legacy-run.json"
            )
            record = json.loads(record_path.read_text(encoding="utf-8"))
            for key in (
                "cuda_home",
                "cuda_nvcc",
                "runtime_ld_library_path",
                "flashinfer_workspace_base",
            ):
                record["configuration"].pop(key)
            record_path.write_text(
                json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
            )
            (bin_dir / "sacct").write_text(
                "#!/bin/bash\n"
                "cat <<'EOF'\n"
                + "\n".join(
                    f"{job_id}|CANCELLED|1:0"
                    for job_id in range(1001, 1012)
                )
                + "\nEOF\n",
                encoding="utf-8",
            )
            (bin_dir / "sacct").chmod(0o755)

            status = subprocess.run(
                ["bash", str(SUBMITTER), "--status", str(record_path)],
                cwd=REPOSITORY,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(status.returncode, 1)
            self.assertNotIn("incomplete immutable configuration", status.stderr)
            reconciled = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(reconciled["dag_status"], "failed")
            self.assertNotIn("cuda_home", reconciled["configuration"])
            claim = pathlib.Path(reconciled["claim_dir"])
            self.assertTrue(claim.is_dir())

            released = subprocess.run(
                [
                    "bash",
                    str(SUBMITTER),
                    "--release-claim",
                    str(record_path),
                ],
                cwd=REPOSITORY,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                released.returncode, 0, released.stdout + released.stderr
            )
            self.assertFalse(claim.exists())
            unchanged = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertNotIn("cuda_home", unchanged["configuration"])
            self.assertNotIn(
                "flashinfer_workspace_base", unchanged["configuration"]
            )

    def test_resume_rejects_changed_atomic_helper_without_mutation(self):
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
            cuda_home = (root / "cuda-12.8.1").resolve()
            environment = {
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "USER": "test-user",
                "SCRATCH_ROOT": str(scratch),
                "ROOT": str(scratch / "oprd-experiments"),
                "PYTHON": str(fake_python),
                "MODULES_INIT": str(modules),
                "BRIDGE_RUNTIME_PATH": f"{bin_dir}:{os.environ['PATH']}",
                "BRIDGE_CUDA_HOME": str(cuda_home),
                "BRIDGE_CUDA_NVCC": str(cuda_home / "bin" / "nvcc"),
                "BRIDGE_RUNTIME_LD_LIBRARY_PATH": str(cuda_home / "lib64"),
                "VLLM_CACHE_ROOT": str(scratch / "vllm-cache"),
                "TRITON_CACHE_DIR": str(scratch / "triton-cache"),
                "FLASHINFER_WORKSPACE_BASE": str(
                    (scratch / "flashinfer-workspace").resolve()
                ),
                "BRIDGE_RUN_ID": "legacy-resume-run",
                "SUBMIT_MIN_FREE_GIB": "0",
                "SUBMIT_MIN_FREE_INODES": "0",
                "ALLOW_QUOTA_UNAVAILABLE": "1",
                "SBATCH_COUNTER": str(counter),
                "SBATCH_LOG": str(log),
            }
            submitted = subprocess.run(
                ["bash", str(SUBMITTER)],
                cwd=REPOSITORY,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                submitted.returncode, 0, submitted.stdout + submitted.stderr
            )
            record_path = (
                cluster / "job_ids" / "oprd_bridge_legacy-resume-run.json"
            )
            record = json.loads(record_path.read_text(encoding="utf-8"))
            pathlib.Path(record["build_dir"]).mkdir(parents=True)
            for key in (
                "cuda_home",
                "cuda_nvcc",
                "runtime_ld_library_path",
                "flashinfer_workspace_base",
            ):
                record["configuration"].pop(key)
            record_path.write_text(
                json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
            )
            claim = pathlib.Path(record["claim_dir"])
            build = pathlib.Path(record["build_dir"])
            record_before_resume = record_path.read_bytes()
            helper = cluster / "oprd_atomic_directory.py"
            helper.write_bytes(helper.read_bytes() + b"\n# authenticated helper changed\n")

            resumed = subprocess.run(
                ["bash", str(SUBMITTER), "--resume", str(record_path)],
                cwd=REPOSITORY,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(resumed.returncode, 0)
            self.assertIn("bridge source bundle changed", resumed.stderr)
            self.assertEqual(counter.read_text(encoding="ascii"), "11\n")
            self.assertTrue(record_path.is_file())
            self.assertTrue(build.is_dir())
            self.assertTrue(claim.is_dir())
            self.assertEqual(record_path.read_bytes(), record_before_resume)
            unchanged = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertNotIn("cuda_home", unchanged["configuration"])

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
                "BRIDGE_CUDA_HOME": str((root / "cuda-12.8.1").resolve()),
                "BRIDGE_CUDA_NVCC": str(
                    (root / "cuda-12.8.1" / "bin" / "nvcc").resolve()
                ),
                "BRIDGE_RUNTIME_LD_LIBRARY_PATH": str(
                    (root / "cuda-12.8.1" / "lib64").resolve()
                ),
                "VLLM_CACHE_ROOT": str(scratch / "vllm-cache"),
                "TRITON_CACHE_DIR": str(scratch / "triton-cache"),
                "FLASHINFER_WORKSPACE_BASE": str(
                    (scratch / "flashinfer-workspace").resolve()
                ),
                "FAKE_LIBCUDART": str(
                    (
                        root
                        / "cuda-12.8.1"
                        / "lib64"
                        / "libcudart.so.12"
                    ).resolve()
                ),
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

            resumed_environment = {
                **environment,
                "BRIDGE_CUDA_HOME": "/ambient/cuda",
                "BRIDGE_CUDA_NVCC": "/ambient/cuda/bin/nvcc",
                "BRIDGE_RUNTIME_LD_LIBRARY_PATH": "/ambient/cuda/lib64",
                "FLASHINFER_WORKSPACE_BASE": "/ambient/flashinfer",
            }
            resumed = subprocess.run(
                ["bash", str(SUBMITTER), "--resume", str(record_path)],
                cwd=REPOSITORY,
                env=resumed_environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            lines = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 18)
            resumed_lines = lines[11:]
            self.assertEqual(len(resumed_lines), 7)
            self.assertTrue(
                all(
                    f"CUDA_HOME={environment['BRIDGE_CUDA_HOME']}" in line
                    and f"FLASHINFER_NVCC={environment['BRIDGE_CUDA_NVCC']}" in line
                    and (
                        "LD_LIBRARY_PATH="
                        f"{environment['BRIDGE_RUNTIME_LD_LIBRARY_PATH']}"
                    ) in line
                    and (
                        "FLASHINFER_WORKSPACE_BASE="
                        f"{environment['FLASHINFER_WORKSPACE_BASE']}"
                    ) in line
                    and "/ambient/" not in line
                    for line in resumed_lines
                )
            )
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
