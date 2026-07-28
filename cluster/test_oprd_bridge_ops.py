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


def make_cuda_runtime_tree(root):
    cuda_home = root / "cuda-12.8.1"
    bin_dir = cuda_home / "bin"
    lib64 = cuda_home / "lib64"
    bin_dir.mkdir(parents=True)
    lib64.mkdir()
    nvcc = bin_dir / "nvcc"
    nvcc.write_bytes(b"fake nvcc")
    nvcc.chmod(0o755)
    libcudart = lib64 / ops.CUDA_RUNTIME_SONAME
    libcudart.write_bytes(b"fake libcudart")
    python = root / "python3.12"
    python.write_bytes(b"fake python")
    python.chmod(0o755)
    readelf = root / "readelf"
    readelf.write_bytes(b"fake readelf")
    readelf.chmod(0o755)
    return tuple(
        path.resolve() for path in (cuda_home, nvcc, libcudart, python, readelf)
    )


def cuda_runtime_identity(cuda_home, nvcc, libcudart, runtime_ld_library_path):
    return {
        "cuda_home": os.fspath(cuda_home.resolve()),
        "cuda_nvcc": os.fspath(nvcc),
        "cuda_nvcc_version": "Cuda compilation tools, release 12.8, V12.8.93",
        "cuda_nvcc_sha256": hashlib.sha256(nvcc.read_bytes()).hexdigest(),
        "cuda_lib64": os.fspath(cuda_home / "lib64"),
        "runtime_ld_library_path": runtime_ld_library_path,
        "libcudart_path": os.fspath(libcudart.resolve()),
        "libcudart_sha256": hashlib.sha256(libcudart.read_bytes()).hexdigest(),
    }


def write_environment_marker(
    marker,
    requirements,
    python,
    ffmpeg,
    runtime_identity,
    workspace,
    packages=None,
):
    package_versions = packages or {
        "flashinfer-python": "0.4.1",
        "torch": "test",
    }
    canonical_workspace = workspace.resolve()
    flashinfer_runtime = {
        "ok": True,
        "paths": {
            "flashinfer_cache_dir": os.fspath(
                canonical_workspace / ".cache" / "flashinfer"
            ),
            "flashinfer_workspace_dir": os.fspath(
                canonical_workspace / ".cache" / "flashinfer" / "0.4.1" / "90a"
            ),
            "flashinfer_jit_dir": os.fspath(
                canonical_workspace
                / ".cache"
                / "flashinfer"
                / "0.4.1"
                / "90a"
                / "cached_ops"
            ),
            "flashinfer_gen_src_dir": os.fspath(
                canonical_workspace
                / ".cache"
                / "flashinfer"
                / "0.4.1"
                / "90a"
                / "generated"
            ),
        },
    }
    ops.atomic_write_json(
        marker,
        {
            "schema_version": 2,
            "python_executable": os.fspath(python.resolve()),
            "requirements_sha256": hashlib.sha256(
                requirements.read_bytes()
            ).hexdigest(),
            "packages": package_versions,
            "ffmpeg_path": os.fspath(ffmpeg.resolve()),
            "ffmpeg_version": "ffmpeg version test",
            "cuda_home": runtime_identity["cuda_home"],
            "cuda_nvcc": runtime_identity["cuda_nvcc"],
            "cuda_nvcc_version": runtime_identity["cuda_nvcc_version"],
            "cuda_nvcc_sha256": runtime_identity["cuda_nvcc_sha256"],
            "runtime_ld_library_path": runtime_identity[
                "runtime_ld_library_path"
            ],
            "libcudart_path": runtime_identity["libcudart_path"],
            "libcudart_sha256": runtime_identity["libcudart_sha256"],
            "flashinfer_workspace_base": os.fspath(canonical_workspace),
            "flashinfer_package_version": package_versions["flashinfer-python"],
            "flashinfer_runtime": flashinfer_runtime,
        },
    )


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

    def test_validate_paths_under_scratch_rejects_escapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            scratch = root / "scratch"
            outside = root / "outside"
            scratch.mkdir()
            outside.mkdir()
            result = ops.validate_paths_under_scratch(
                scratch, [scratch / "cache", scratch / "nested" / "cache"]
            )
            self.assertEqual(result["scratch"], os.fspath(scratch.resolve()))
            with self.assertRaises(ops.UnsafePathError):
                ops.validate_paths_under_scratch(scratch, [scratch / ".." / "outside"])
            link = scratch / "link"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ops.UnsafePathError):
                ops.validate_paths_under_scratch(scratch, [link / "cache"])

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


class CudaRuntimeProbeTests(unittest.TestCase):
    def make_runner(
        self,
        *,
        nvcc,
        python,
        readelf,
        libcudart,
        dynamic=None,
        observed=None,
    ):
        dynamic_output = dynamic or (
            " 0x0000000000000001 (NEEDED)             "
            "Shared library: [libcudart.so.12]\n"
        )

        def runner(arguments, **kwargs):
            if observed is not None:
                observed.append((list(arguments), dict(kwargs)))
            if arguments == [os.fspath(nvcc), "--version"]:
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    "nvcc: NVIDIA (R) Cuda compiler driver\n"
                    "Cuda compilation tools, release 12.8, V12.8.93\n",
                    "",
                )
            if arguments[0] == os.fspath(nvcc):
                output = pathlib.Path(arguments[arguments.index("-o") + 1])
                output.write_bytes(b"shared object")
                return subprocess.CompletedProcess(arguments, 0, "", "")
            if arguments[0] == os.fspath(readelf.resolve()):
                return subprocess.CompletedProcess(
                    arguments, 0, dynamic_output, ""
                )
            if arguments[0] == os.fspath(python):
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    json.dumps(
                        {
                            "mapped_libcudart_paths": [
                                os.fspath(libcudart.resolve())
                            ]
                        }
                    )
                    + "\n",
                    "",
                )
            raise AssertionError(arguments)

        return runner

    def test_compile_inspect_and_load_uses_sanitized_environment_and_cleans(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary)
            cuda_home, nvcc, libcudart, python, readelf = (
                make_cuda_runtime_tree(scratch)
            )
            observed = []
            runtime_path = os.fspath(scratch)
            runtime_ld = os.fspath(cuda_home / "lib64")
            result = ops.probe_cuda_runtime(
                scratch,
                python,
                runtime_path,
                cuda_home,
                nvcc,
                runtime_ld,
                which_fn=lambda command: os.fspath(readelf)
                if command == "readelf"
                else None,
                runner=self.make_runner(
                    nvcc=nvcc,
                    python=python,
                    readelf=readelf,
                    libcudart=libcudart,
                    observed=observed,
                ),
            )
            self.assertEqual(result["libcudart_path"], os.fspath(libcudart))
            self.assertEqual(
                result["cuda_nvcc_sha256"],
                hashlib.sha256(nvcc.read_bytes()).hexdigest(),
            )
            self.assertEqual(list(scratch.glob(".oprd-cuda-runtime.*")), [])
            for _, kwargs in observed:
                if "env" not in kwargs:
                    continue
                self.assertEqual(
                    kwargs["env"],
                    {
                        "PATH": runtime_path,
                        "CUDA_HOME": os.fspath(cuda_home),
                        "LD_LIBRARY_PATH": runtime_ld,
                        "PYTHONNOUSERSITE": "1",
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "LC_ALL": "C",
                    },
                )
                self.assertNotIn("LD_PRELOAD", kwargs["env"])
                self.assertEqual(
                    kwargs["timeout"], ops.CUDA_RUNTIME_TIMEOUT_SECONDS
                )
            compile_args = observed[1][0]
            self.assertIn("-shared", compile_args)
            self.assertIn("--cudart=none", compile_args)
            self.assertIn("-Xcompiler=-fPIC", compile_args)
            self.assertIn(f"-L{cuda_home / 'lib64'}", compile_args)
            self.assertIn("-lcudart", compile_args)
            self.assertFalse(any("rpath" in value.lower() for value in compile_args))

    def test_cuda_runtime_cli_requires_explicit_runtime_inputs(self):
        parsed = ops.build_parser().parse_args(
            [
                "check-cuda-runtime",
                "--scratch",
                "/scratch",
                "--python",
                "/venv/bin/python",
                "--runtime-path",
                "/venv/bin:/usr/bin",
                "--cuda-home",
                "/opt/cuda-12.8.1",
                "--cuda-nvcc",
                "/opt/cuda-12.8.1/bin/nvcc",
                "--runtime-ld-library-path",
                "/opt/cuda-12.8.1/lib64",
            ]
        )
        self.assertEqual(parsed.command, "check-cuda-runtime")
        self.assertEqual(parsed.runtime_ld_library_path, "/opt/cuda-12.8.1/lib64")

    def test_timeout_and_compile_failure_always_clean_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary)
            cuda_home, nvcc, libcudart, python, readelf = (
                make_cuda_runtime_tree(scratch)
            )

            def runner(arguments, **kwargs):
                if arguments[-1] == "--version":
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        "Cuda compilation tools, release 12.8, V12.8.93\n",
                        "",
                    )
                raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])

            with self.assertRaisesRegex(ops.OpsError, "command failed to run"):
                ops.probe_cuda_runtime(
                    scratch,
                    python,
                    os.fspath(scratch),
                    cuda_home,
                    nvcc,
                    os.fspath(cuda_home / "lib64"),
                    which_fn=lambda command: os.fspath(readelf),
                    runner=runner,
                )
            self.assertEqual(list(scratch.glob(".oprd-cuda-runtime.*")), [])

            def failed_runner(arguments, **kwargs):
                if arguments[-1] == "--version":
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        "Cuda compilation tools, release 12.8, V12.8.93\n",
                        "",
                    )
                return subprocess.CompletedProcess(
                    arguments, 1, "", "compile failed"
                )

            with self.assertRaisesRegex(ops.OpsError, "compile failed"):
                ops.probe_cuda_runtime(
                    scratch,
                    python,
                    os.fspath(scratch),
                    cuda_home,
                    nvcc,
                    os.fspath(cuda_home / "lib64"),
                    which_fn=lambda command: os.fspath(readelf),
                    runner=failed_runner,
                )
            self.assertEqual(list(scratch.glob(".oprd-cuda-runtime.*")), [])

    def test_rejects_wrong_toolkit_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary)
            cuda_home, nvcc, libcudart, python, readelf = (
                make_cuda_runtime_tree(scratch)
            )
            wrong_nvcc = scratch / "nvcc"
            wrong_nvcc.write_bytes(b"wrong")
            wrong_nvcc.chmod(0o755)
            with self.assertRaisesRegex(ops.OpsError, "exactly"):
                ops.probe_cuda_runtime(
                    scratch,
                    python,
                    os.fspath(scratch),
                    cuda_home,
                    wrong_nvcc,
                    os.fspath(cuda_home / "lib64"),
                    which_fn=lambda command: os.fspath(readelf),
                )
            libcudart.unlink()
            with self.assertRaisesRegex(ops.OpsError, "runtime library is missing"):
                ops.probe_cuda_runtime(
                    scratch,
                    python,
                    os.fspath(scratch),
                    cuda_home,
                    nvcc,
                    os.fspath(cuda_home / "lib64"),
                    which_fn=lambda command: os.fspath(readelf),
                )

    def test_rejects_wrong_mapped_runtime_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary)
            cuda_home, nvcc, libcudart, python, readelf = (
                make_cuda_runtime_tree(scratch)
            )
            wrong_runtime = scratch / "wrong" / ops.CUDA_RUNTIME_SONAME
            wrong_runtime.parent.mkdir()
            wrong_runtime.write_bytes(b"wrong runtime")
            with self.assertRaisesRegex(ops.OpsError, "mapped libcudart path"):
                ops.probe_cuda_runtime(
                    scratch,
                    python,
                    os.fspath(scratch),
                    cuda_home,
                    nvcc,
                    os.fspath(cuda_home / "lib64"),
                    which_fn=lambda command: os.fspath(readelf),
                    runner=self.make_runner(
                        nvcc=nvcc,
                        python=python,
                        readelf=readelf,
                        libcudart=wrong_runtime,
                    ),
                )
            self.assertEqual(list(scratch.glob(".oprd-cuda-runtime.*")), [])

    def test_readelf_requires_needed_and_forbids_runtime_search_paths(self):
        for dynamic, message in (
            ("Dynamic section is empty\n", "does not require"),
            (
                "(NEEDED) Shared library: [libcudart.so.12]\n"
                "(RUNPATH) Library runpath: [/tmp]\n",
                "forbidden RPATH",
            ),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                scratch = pathlib.Path(temporary)
                cuda_home, nvcc, libcudart, python, readelf = (
                    make_cuda_runtime_tree(scratch)
                )
                with self.assertRaisesRegex(ops.OpsError, message):
                    ops.probe_cuda_runtime(
                        scratch,
                        python,
                        os.fspath(scratch),
                        cuda_home,
                        nvcc,
                        os.fspath(cuda_home / "lib64"),
                        which_fn=lambda command: os.fspath(readelf),
                        runner=self.make_runner(
                            nvcc=nvcc,
                            python=python,
                            readelf=readelf,
                            libcudart=libcudart,
                            dynamic=dynamic,
                        ),
                    )
                self.assertEqual(
                    list(scratch.glob(".oprd-cuda-runtime.*")), []
                )


class EnvironmentMarkerTests(unittest.TestCase):
    def make_environment(self, root):
        scratch = root / "scratch"
        scratch.mkdir()
        cuda_home, nvcc, libcudart, python, readelf = (
            make_cuda_runtime_tree(root)
        )
        ffmpeg = root / "ffmpeg"
        ffmpeg.write_bytes(b"ffmpeg")
        ffmpeg.chmod(0o755)
        requirements = scratch / "requirements.txt"
        requirements.write_text("torch>=2\n", encoding="utf-8")
        workspace = scratch / "flashinfer"
        workspace.mkdir()
        workspace = workspace.resolve()
        runtime_ld = os.fspath(cuda_home / "lib64")
        identity = cuda_runtime_identity(
            cuda_home, nvcc, libcudart, runtime_ld
        )
        marker = scratch / "env-ready.json"
        write_environment_marker(
            marker,
            requirements,
            python,
            ffmpeg,
            identity,
            workspace,
        )
        return (
            scratch,
            cuda_home,
            nvcc,
            libcudart,
            python,
            readelf,
            ffmpeg,
            requirements,
            workspace,
            identity,
            marker,
        )

    def validate_marker(self, values, **overrides):
        (
            scratch,
            _,
            _,
            _,
            python,
            _,
            ffmpeg,
            requirements,
            workspace,
            identity,
            marker,
        ) = values
        arguments = {
            "scratch": scratch,
            "cuda_home": identity["cuda_home"],
            "cuda_nvcc": identity["cuda_nvcc"],
            "cuda_nvcc_version": identity["cuda_nvcc_version"],
            "cuda_nvcc_sha256": identity["cuda_nvcc_sha256"],
            "runtime_ld_library_path": identity["runtime_ld_library_path"],
            "libcudart_path": identity["libcudart_path"],
            "libcudart_sha256": identity["libcudart_sha256"],
            "flashinfer_workspace_base": os.fspath(workspace),
            "flashinfer_package_version": "0.4.1",
            "actual_packages": {
                "flashinfer-python": "0.4.1",
                "torch": "test",
            },
            "ffmpeg_path": os.fspath(ffmpeg),
            "ffmpeg_version": "ffmpeg version test",
        }
        arguments.update(overrides)
        return ops.validate_environment_marker(
            marker, requirements, python, **arguments
        )

    def test_schema_2_marker_validates_and_rejects_mismatches(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = self.make_environment(pathlib.Path(temporary))
            result = self.validate_marker(values)
            self.assertEqual(result["cuda_nvcc_version"], values[-2]["cuda_nvcc_version"])
            marker = values[-1]
            original = ops.read_json(marker)
            for field in (
                "cuda_home",
                "cuda_nvcc",
                "cuda_nvcc_version",
                "cuda_nvcc_sha256",
                "runtime_ld_library_path",
                "libcudart_path",
                "libcudart_sha256",
                "flashinfer_workspace_base",
                "flashinfer_package_version",
            ):
                with self.subTest(field=field):
                    payload = dict(original)
                    payload[field] = "wrong"
                    ops.atomic_write_json(marker, payload)
                    with self.assertRaises(ops.OpsError):
                        self.validate_marker(values)
            payload = dict(original)
            payload["schema_version"] = 1
            ops.atomic_write_json(marker, payload)
            with self.assertRaisesRegex(ops.OpsError, "schema 2"):
                self.validate_marker(values)

    def test_marker_requires_successful_contained_flashinfer_warmup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            values = self.make_environment(root)
            marker = values[-1]
            original = ops.read_json(marker)

            missing = json.loads(json.dumps(original))
            missing.pop("flashinfer_runtime")
            ops.atomic_write_json(marker, missing)
            with self.assertRaisesRegex(ops.OpsError, "successful FlashInfer"):
                self.validate_marker(values)

            failed = json.loads(json.dumps(original))
            failed["flashinfer_runtime"]["ok"] = False
            ops.atomic_write_json(marker, failed)
            with self.assertRaisesRegex(ops.OpsError, "successful FlashInfer"):
                self.validate_marker(values)

            incomplete = json.loads(json.dumps(original))
            incomplete["flashinfer_runtime"]["paths"].pop(
                "flashinfer_gen_src_dir"
            )
            ops.atomic_write_json(marker, incomplete)
            with self.assertRaisesRegex(ops.OpsError, "incomplete FlashInfer"):
                self.validate_marker(values)

            escaping = json.loads(json.dumps(original))
            escaping["flashinfer_runtime"]["paths"][
                "flashinfer_cache_dir"
            ] = os.fspath((root / "outside").resolve())
            ops.atomic_write_json(marker, escaping)
            with self.assertRaisesRegex(ops.OpsError, "escapes workspace"):
                self.validate_marker(values)

    def test_current_environment_probes_identity_without_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            values = self.make_environment(pathlib.Path(temporary))
            (
                scratch,
                cuda_home,
                nvcc,
                _,
                python,
                _,
                ffmpeg,
                requirements,
                workspace,
                identity,
                marker,
            ) = values
            commands = []

            def runner(arguments, **kwargs):
                commands.append(arguments)
                if arguments[0] == os.fspath(python):
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        '{"flashinfer-python":"0.4.1","torch":"test"}\n',
                        "",
                    )
                if arguments[0] == os.fspath(ffmpeg):
                    return subprocess.CompletedProcess(
                        arguments, 0, "ffmpeg version test\n", ""
                    )
                raise AssertionError(arguments)

            result = ops.validate_current_environment(
                marker,
                requirements,
                python,
                scratch=scratch,
                runtime_path=os.fspath(scratch),
                cuda_home=cuda_home,
                cuda_nvcc=nvcc,
                runtime_ld_library_path=identity[
                    "runtime_ld_library_path"
                ],
                flashinfer_workspace_base=workspace,
                which_fn=lambda command: os.fspath(ffmpeg)
                if command == "ffmpeg"
                else None,
                runner=runner,
                cuda_runtime_identity=identity,
            )
            self.assertEqual(result["flashinfer_package_version"], "0.4.1")
            self.assertFalse(any("cuda" in " ".join(command).lower() for command in commands))

    def test_flashinfer_workspace_must_be_canonical_under_scratch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            values = self.make_environment(root)
            outside = root / "outside"
            outside.mkdir()
            with self.assertRaises(ops.UnsafePathError):
                self.validate_marker(
                    values, flashinfer_workspace_base=os.fspath(outside)
                )


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

    def test_native_toolchain_probe_links_shared_object_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary)
            gcc = scratch / "gcc"
            ld = scratch / "ld"
            gcc.write_text("probe", encoding="utf-8")
            ld.write_text("probe", encoding="utf-8")
            gcc.chmod(0o755)
            ld.chmod(0o755)
            observed_timeouts = []

            def runner(arguments, **kwargs):
                observed_timeouts.append(kwargs["timeout"])
                if arguments[-1] == "--version":
                    name = pathlib.Path(arguments[0]).name
                    return subprocess.CompletedProcess(
                        arguments, 0, f"{name} test\n", ""
                    )
                output = pathlib.Path(arguments[arguments.index("-o") + 1])
                output.write_bytes(b"shared-object")
                return subprocess.CompletedProcess(arguments, 0, "", "")

            result = ops.probe_native_toolchain(
                scratch,
                which_fn=lambda command: {
                    "gcc": os.fspath(gcc),
                    "ld": os.fspath(ld),
                }.get(command),
                runner=runner,
            )
            self.assertEqual(result["compiler_version"], "gcc test")
            self.assertEqual(result["linker_version"], "ld test")
            self.assertEqual(
                observed_timeouts,
                [ops.TOOLCHAIN_TIMEOUT_SECONDS] * 3,
            )
            self.assertEqual(list(scratch.glob(".oprd-toolchain.*")), [])

    def test_native_toolchain_probe_rejects_missing_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary)
            with self.assertRaisesRegex(ops.OpsError, "compiler is unavailable"):
                ops.probe_native_toolchain(scratch, which_fn=lambda command: None)

            gcc = scratch / "gcc"
            gcc.write_text("probe", encoding="utf-8")
            gcc.chmod(0o755)
            with self.assertRaisesRegex(ops.OpsError, "linker is unavailable"):
                ops.probe_native_toolchain(
                    scratch,
                    which_fn=lambda command: os.fspath(gcc)
                    if command == "gcc"
                    else None,
                )

    def test_native_toolchain_probe_rejects_failed_or_missing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary)
            gcc = scratch / "gcc"
            ld = scratch / "ld"
            for executable in (gcc, ld):
                executable.write_text("probe", encoding="utf-8")
                executable.chmod(0o755)
            commands = {"gcc": os.fspath(gcc), "ld": os.fspath(ld)}

            def failed_runner(arguments, **kwargs):
                if arguments[-1] == "--version":
                    return subprocess.CompletedProcess(arguments, 0, "test\n", "")
                return subprocess.CompletedProcess(arguments, 1, "", "link failed")

            with self.assertRaisesRegex(ops.OpsError, "link failed"):
                ops.probe_native_toolchain(
                    scratch,
                    which_fn=lambda command: commands.get(command),
                    runner=failed_runner,
                )
            self.assertEqual(list(scratch.glob(".oprd-toolchain.*")), [])

            def missing_output_runner(arguments, **kwargs):
                return subprocess.CompletedProcess(arguments, 0, "test\n", "")

            with self.assertRaisesRegex(ops.OpsError, "did not produce"):
                ops.probe_native_toolchain(
                    scratch,
                    which_fn=lambda command: commands.get(command),
                    runner=missing_output_runner,
                )
            self.assertEqual(list(scratch.glob(".oprd-toolchain.*")), [])

    def test_native_toolchain_probe_reports_compile_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary)
            gcc = scratch / "gcc"
            ld = scratch / "ld"
            for executable in (gcc, ld):
                executable.write_text("probe", encoding="utf-8")
                executable.chmod(0o755)
            commands = {"gcc": os.fspath(gcc), "ld": os.fspath(ld)}

            def runner(arguments, **kwargs):
                if arguments[-1] == "--version":
                    return subprocess.CompletedProcess(arguments, 0, "test\n", "")
                raise subprocess.TimeoutExpired(arguments, kwargs["timeout"])

            with self.assertRaisesRegex(
                ops.OpsError, "compiler/linker probe failed"
            ):
                ops.probe_native_toolchain(
                    scratch,
                    which_fn=lambda command: commands.get(command),
                    runner=runner,
                )
            self.assertEqual(list(scratch.glob(".oprd-toolchain.*")), [])

    def test_preflight_uses_configured_runtime_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary) / "scratch"
            scratch.mkdir()
            cuda_home, nvcc, libcudart, python, readelf = (
                make_cuda_runtime_tree(scratch)
            )
            hf_home = scratch / "hf-cache"
            writable = scratch / "records"
            workspace = scratch / "flashinfer"
            workspace.mkdir()
            workspace = workspace.resolve()
            script = scratch / "phase.sbatch"
            script.write_text("#!/bin/sh\n", encoding="utf-8")
            sbatch = scratch / "sbatch"
            sacct = scratch / "sacct"
            ffmpeg = scratch / "ffmpeg"
            gcc = scratch / "gcc"
            ld = scratch / "ld"
            for executable in (sbatch, sacct, ffmpeg, gcc, ld):
                executable.write_bytes(b"fake executable")
                executable.chmod(0o755)
            requirements = scratch / "requirements.txt"
            requirements.write_text("torch>=2\n", encoding="utf-8")
            runtime_ld = os.fspath(cuda_home / "lib64")
            identity = cuda_runtime_identity(
                cuda_home, nvcc, libcudart, runtime_ld
            )
            marker = scratch / "env-ready.json"
            write_environment_marker(
                marker,
                requirements,
                python,
                ffmpeg,
                identity,
                workspace,
            )
            commands = {
                "sbatch": os.fspath(sbatch),
                "sacct": os.fspath(sacct),
                "ffmpeg": os.fspath(ffmpeg),
                "gcc": os.fspath(gcc),
                "ld": os.fspath(ld),
                "readelf": os.fspath(readelf),
            }
            import_timeouts = []

            def recording_runner(arguments, **kwargs):
                executable = arguments[0]
                if executable == os.fspath(python) and arguments[1:] == ["--version"]:
                    return subprocess.CompletedProcess(
                        arguments, 0, "Python 3.12.9\n", ""
                    )
                if (
                    executable == os.fspath(python)
                    and len(arguments) >= 3
                    and arguments[1] == "-c"
                    and "import_module" in arguments[2]
                ):
                    import_timeouts.append(kwargs.get("timeout"))
                    return subprocess.CompletedProcess(arguments, 0, "", "")
                if (
                    executable == os.fspath(python)
                    and "importlib.metadata" in arguments[2]
                ):
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        '{"flashinfer-python":"0.4.1","torch":"test"}\n',
                        "",
                    )
                if executable == os.fspath(python):
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        json.dumps(
                            {
                                "mapped_libcudart_paths": [
                                    os.fspath(libcudart.resolve())
                                ]
                            }
                        )
                        + "\n",
                        "",
                    )
                if executable == os.fspath(nvcc) and arguments[-1] == "--version":
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        "Cuda compilation tools, release 12.8, V12.8.93\n",
                        "",
                    )
                if executable == os.fspath(gcc.resolve()) and arguments[-1] == "--version":
                    return subprocess.CompletedProcess(
                        arguments, 0, "gcc test\n", ""
                    )
                if executable == os.fspath(ld.resolve()):
                    return subprocess.CompletedProcess(
                        arguments, 0, "GNU ld test\n", ""
                    )
                if executable in {os.fspath(nvcc), os.fspath(gcc.resolve())}:
                    output = pathlib.Path(arguments[arguments.index("-o") + 1])
                    output.write_bytes(b"shared object")
                    return subprocess.CompletedProcess(arguments, 0, "", "")
                if executable == os.fspath(readelf.resolve()):
                    return subprocess.CompletedProcess(
                        arguments,
                        0,
                        "(NEEDED) Shared library: [libcudart.so.12]\n",
                        "",
                    )
                if executable == os.fspath(ffmpeg):
                    return subprocess.CompletedProcess(
                        arguments, 0, "ffmpeg version test\n", ""
                    )
                raise AssertionError(arguments)

            preflight_arguments = {
                "scratch": scratch,
                "hf_home": hf_home,
                "python": python,
                "required_imports": ["json", "some.fake.module"],
                "required_scripts": [script],
                "runtime_path": os.fspath(scratch),
                "cuda_home": cuda_home,
                "cuda_nvcc": nvcc,
                "runtime_ld_library_path": runtime_ld,
                "flashinfer_workspace_base": workspace,
                "writable_paths": [writable],
                "environment_marker": marker,
                "requirements_file": requirements,
                "min_free_blocks": 1,
                "min_free_inodes": 1,
                "quota_user": "test-user",
                "allow_quota_unavailable": True,
                "which_fn": lambda command: commands.get(command),
                "runner": recording_runner,
            }
            report = ops.run_preflight(**preflight_arguments)
            self.assertTrue(report["ok"], report)
            self.assertEqual(
                import_timeouts,
                [ops.PREFLIGHT_IMPORT_TIMEOUT_SECONDS] * 2,
            )
            self.assertEqual(report["checks"]["python"]["version"], "3.12.9")
            self.assertEqual(
                report["checks"]["cuda_runtime"]["libcudart_path"],
                os.fspath(libcudart),
            )
            self.assertEqual(
                report["checks"]["environment_marker"]["packages"]["torch"],
                "test",
            )
            self.assertEqual(report["quota"]["status"], "unavailable")

            saved_runtime = libcudart.read_bytes()
            libcudart.unlink()
            failed_cuda = ops.run_preflight(**preflight_arguments)
            self.assertFalse(failed_cuda["ok"])
            self.assertFalse(failed_cuda["checks"]["cuda_runtime"]["ok"])
            self.assertTrue(
                any("CUDA runtime" in error for error in failed_cuda["errors"]),
                failed_cuda,
            )
            libcudart.write_bytes(saved_runtime)

            requirements.write_text("torch>=3\n", encoding="utf-8")
            changed = ops.run_preflight(**preflight_arguments)
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
                runtime_path=os.fspath(scratch),
                cuda_home=scratch / "cuda",
                cuda_nvcc=scratch / "cuda" / "bin" / "nvcc",
                runtime_ld_library_path=os.fspath(scratch / "cuda" / "lib64"),
                flashinfer_workspace_base=scratch / "flashinfer",
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


class RuntimeClaimTests(unittest.TestCase):
    def test_runtime_configuration_must_match_immutable_record_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as temporary:
            scratch = pathlib.Path(temporary) / "scratch"
            scratch.mkdir()
            repository = scratch / "repository"
            for relative in ops.SOURCE_BUNDLE_PATHS:
                path = repository / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative, encoding="utf-8")
            environment_marker = repository / "venv" / ".oprd-env-ready.json"
            environment_marker.parent.mkdir()
            environment_marker.write_text("{}\n", encoding="utf-8")
            artifact = scratch / "artifact"
            record = scratch / "records" / "run.json"
            configuration = {
                "root": os.fspath(repository),
                "source_bundle_sha256": ops.compute_source_bundle_sha256(
                    repository
                ),
                "environment_marker_sha256": hashlib.sha256(
                    environment_marker.read_bytes()
                ).hexdigest(),
                "cuda_home": "/opt/cuda-12.8.1",
                "cuda_nvcc": "/opt/cuda-12.8.1/bin/nvcc",
                "runtime_ld_library_path": "/opt/cuda-12.8.1/lib64",
                "flashinfer_workspace_base": os.fspath(
                    scratch / "flashinfer"
                ),
            }
            ops.initialize_record(
                scratch=scratch,
                record_path=record,
                run_id="run-1",
                build_dir=scratch / "build",
                artifact_dir=artifact,
                phase_order=("one",),
                configuration=configuration,
            )
            ops.claim_artifact(
                scratch=scratch,
                artifact_dir=artifact,
                run_id="run-1",
                record_path=record,
            )
            result = ops.validate_runtime_claim(
                scratch=scratch,
                paths=[scratch / "cache"],
                artifact_dir=artifact,
                run_id="run-1",
                record_path=record,
                cuda_home=configuration["cuda_home"],
                cuda_nvcc=configuration["cuda_nvcc"],
                runtime_ld_library_path=configuration[
                    "runtime_ld_library_path"
                ],
                flashinfer_workspace_base=configuration[
                    "flashinfer_workspace_base"
                ],
            )
            self.assertTrue(result["ok"])
            for field in (
                "cuda_home",
                "cuda_nvcc",
                "runtime_ld_library_path",
                "flashinfer_workspace_base",
            ):
                with self.subTest(field=field), self.assertRaisesRegex(
                    ops.ClaimConflictError,
                    f"runtime configuration {field}",
                ):
                    ops.validate_runtime_claim(
                        scratch=scratch,
                        paths=[],
                        artifact_dir=artifact,
                        run_id="run-1",
                        record_path=record,
                        **{field: configuration[field] + "/"},
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
