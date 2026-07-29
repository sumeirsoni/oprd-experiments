#!/usr/bin/env python3
"""Safe, restartable operations for the OPRD bridge smoke DAG.

This module intentionally uses only the Python standard library. Expensive model
packages are checked in a configured Python subprocess and are never imported here.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import errno
import fcntl
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


CLUSTER_DIRECTORY = pathlib.Path(__file__).resolve().parent.parent
if os.fspath(CLUSTER_DIRECTORY) not in sys.path:
    sys.path.insert(0, os.fspath(CLUSTER_DIRECTORY))

from oprd_atomic_directory import probe_atomic_directory_move


SCHEMA_VERSION = 1
CLAIM_FILENAME = "claim.json"
SMOKE_DIRECTORY = "stage2-backward-smoke"
SMOKE_COMPLETE_FILENAME = "SMOKE_COMPLETE.json"
PREFLIGHT_IMPORT_TIMEOUT_SECONDS = 180
TOOLCHAIN_TIMEOUT_SECONDS = 60
TEMP_DIRECTORY_CLEANUP_ATTEMPTS = 8
TEMP_DIRECTORY_CLEANUP_INITIAL_DELAY_SECONDS = 0.1
CUDA_RUNTIME_TIMEOUT_SECONDS = 60
CUDA_RUNTIME_SONAME = "libcudart.so.12"
FLASHINFER_DISTRIBUTION = "flashinfer-python"
FLASHINFER_RUNTIME_PATH_FIELDS = (
    "flashinfer_cache_dir",
    "flashinfer_workspace_dir",
    "flashinfer_jit_dir",
    "flashinfer_gen_src_dir",
)

DAG_PHASES = (
    "corpus",
    "teacher",
    *(f"student-group-{index}" for index in range(7)),
    "publish",
    "stage2-smoke",
)
BUILDER_REQUIRED_PHASES = (
    "init",
    "generate-corpus",
    "fit-teacher-pca",
    "materialize-teacher-targets",
    *(f"train-student-group-{index}" for index in range(7)),
    "publish",
    "verify",
)
VERIFY_CHECKED_PHASES = BUILDER_REQUIRED_PHASES[:-1]
SOURCE_BUNDLE_PATHS = (
    "cluster/build_oprd_bridge.py",
    "cluster/oprd_bridge.py",
    "cluster/oprd_atomic_directory.py",
    "cluster/train_harness.py",
    "cluster/requirements.txt",
    "cluster/nexus/oprd_bridge_ops.py",
    "cluster/nexus/submit_oprd_bridge.sh",
    "cluster/nexus/oprd_bridge_corpus.sbatch",
    "cluster/nexus/oprd_bridge_teacher.sbatch",
    "cluster/nexus/oprd_bridge_student_groups.sbatch",
    "cluster/nexus/oprd_bridge_publish.sbatch",
    "cluster/nexus/oprd_bridge_stage2_smoke.sbatch",
)

ACTIVE_STATES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "SUSPENDED",
        "RESIZING",
        "REQUEUED",
        "REQUEUE_FED",
        "REQUEUE_HOLD",
        "SIGNALING",
        "STAGE_OUT",
    }
)
FAILURE_STATES = frozenset(
    {
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "BOOT_FAIL",
        "DEADLINE",
        "REVOKED",
        "STOPPED",
        "SPECIAL_EXIT",
    }
)
JOB_ID_RE = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^Python\s+(\d+)\.(\d+)\.(\d+)(?:\s|$)")

Runner = Callable[..., subprocess.CompletedProcess[str]]


class OpsError(RuntimeError):
    """Base class for an operational safety or validation failure."""


class UnsafePathError(OpsError):
    """A path is not a safe canonical descendant of scratch."""


class ClaimConflictError(OpsError):
    """An artifact claim belongs to a different run or record."""


class ActiveJobsError(OpsError):
    """Resume was requested while at least one recorded job is active."""


class SuccessValidationError(OpsError):
    """Final builder, artifact, or smoke outputs are inconsistent."""

    def __init__(self, message: str, *, resume_phase: str):
        super().__init__(message)
        self.resume_phase = resume_phase


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _fsync_directory(path: pathlib.Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_json(path: os.PathLike[str] | str, value: Any) -> pathlib.Path:
    """Atomically replace *path* with canonical, fsynced JSON."""

    destination = pathlib.Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_json(path: os.PathLike[str] | str) -> Any:
    """Read valid UTF-8 JSON without following a symlink at the final path."""

    source = pathlib.Path(path)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OpsError(f"cannot read valid JSON from {source}: {exc}") from exc


def _lexical_absolute(path: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _reject_descendant_symlinks(root: pathlib.Path, candidate: pathlib.Path) -> None:
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        # Some systems expose the same filesystem path through a prefix alias, such
        # as /var and /private/var on macOS. Canonical containment below remains the
        # authority in that case. A symlink below a lexically matching scratch root
        # is still rejected component by component.
        try:
            candidate.resolve(strict=False).relative_to(root.resolve(strict=True))
        except (OSError, RuntimeError, ValueError) as exc:
            raise UnsafePathError(
                f"path is not under canonical scratch: {candidate}"
            ) from exc
        return
    current = root
    if current.is_symlink():
        raise UnsafePathError(f"scratch must not be a symlink: {root}")
    for part in relative.parts:
        current = current / part
        try:
            if current.is_symlink():
                raise UnsafePathError(f"symlink is not allowed in scratch path: {current}")
        except OSError as exc:
            raise UnsafePathError(f"cannot inspect path component {current}: {exc}") from exc


def canonical_path_under_scratch(
    scratch: os.PathLike[str] | str,
    path: os.PathLike[str] | str,
    *,
    must_exist: bool = False,
    allow_scratch: bool = False,
) -> pathlib.Path:
    """Return a canonical path strictly under scratch, rejecting path symlinks."""

    scratch_input = pathlib.Path(scratch)
    path_input = pathlib.Path(path)
    if not scratch_input.is_absolute() or not path_input.is_absolute():
        raise UnsafePathError("scratch and contained paths must be absolute")

    lexical_scratch = _lexical_absolute(scratch_input)
    lexical_path = _lexical_absolute(path_input)
    if not lexical_scratch.exists() or not lexical_scratch.is_dir():
        raise UnsafePathError(f"scratch directory does not exist: {lexical_scratch}")
    _reject_descendant_symlinks(lexical_scratch, lexical_path)

    try:
        canonical_scratch = lexical_scratch.resolve(strict=True)
        canonical_path = lexical_path.resolve(strict=must_exist)
        relative = canonical_path.relative_to(canonical_scratch)
    except (OSError, RuntimeError, ValueError) as exc:
        raise UnsafePathError(
            f"path escapes canonical scratch {lexical_scratch}: {lexical_path}"
        ) from exc
    if not allow_scratch and not relative.parts:
        raise UnsafePathError(f"path must be below scratch, not scratch itself: {path}")
    return canonical_path


def validate_paths_under_scratch(
    scratch: os.PathLike[str] | str,
    paths: Sequence[os.PathLike[str] | str],
) -> dict[str, Any]:
    canonical_scratch = canonical_path_under_scratch(
        scratch, scratch, must_exist=True, allow_scratch=True
    )
    canonical_paths = [
        canonical_path_under_scratch(canonical_scratch, path) for path in paths
    ]
    return {
        "scratch": os.fspath(canonical_scratch),
        "paths": [os.fspath(path) for path in canonical_paths],
    }


def _probe_writable_path(path: pathlib.Path) -> None:
    probe_directory = path if path.is_dir() else path.parent
    while not probe_directory.exists() and probe_directory != probe_directory.parent:
        probe_directory = probe_directory.parent
    if not probe_directory.is_dir():
        raise OpsError(f"no existing directory can contain path: {path}")
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".oprd-write-probe.", dir=probe_directory
        )
        os.close(descriptor)
        pathlib.Path(temporary_name).unlink()
    except OSError as exc:
        raise OpsError(f"path is not writable: {path}: {exc}") from exc


def check_statvfs_capacity(
    stats: Any,
    *,
    min_free_blocks: int,
    min_free_inodes: int,
    min_free_bytes: int = 0,
) -> dict[str, int]:
    """Validate available filesystem blocks and inodes from a statvfs result."""

    available_blocks = int(stats.f_bavail)
    available_inodes = int(stats.f_favail)
    fragment_size = int(stats.f_frsize or stats.f_bsize)
    if available_blocks < min_free_blocks:
        raise OpsError(
            f"only {available_blocks} free blocks, require at least {min_free_blocks}"
        )
    if available_inodes < min_free_inodes:
        raise OpsError(
            f"only {available_inodes} free inodes, require at least {min_free_inodes}"
        )
    available_bytes = available_blocks * fragment_size
    if available_bytes < min_free_bytes:
        raise OpsError(
            f"only {available_bytes} free bytes, require at least {min_free_bytes}"
        )
    return {
        "available_blocks": available_blocks,
        "available_inodes": available_inodes,
        "fragment_size": fragment_size,
        "available_bytes": available_bytes,
    }


def _require_exact_marker_value(
    marker: Mapping[str, Any], field: str, expected: str
) -> None:
    if marker.get(field) != expected:
        raise OpsError(
            f"environment-ready marker {field} does not match current configuration"
        )


def _canonical_flashinfer_workspace(
    scratch: os.PathLike[str] | str,
    workspace: os.PathLike[str] | str,
) -> pathlib.Path:
    canonical = canonical_path_under_scratch(scratch, workspace)
    if os.fspath(pathlib.Path(workspace)) != os.fspath(canonical):
        raise OpsError("FlashInfer workspace base must be a canonical scratch path")
    if canonical.exists() and not canonical.is_dir():
        raise OpsError(f"FlashInfer workspace base is not a directory: {canonical}")
    return canonical


def _validate_flashinfer_runtime_evidence(
    marker: Mapping[str, Any], workspace: pathlib.Path
) -> dict[str, Any]:
    evidence = marker.get("flashinfer_runtime")
    if not isinstance(evidence, Mapping) or evidence.get("ok") is not True:
        raise OpsError(
            "environment-ready marker has no successful FlashInfer runtime warm-up"
        )
    paths = evidence.get("paths")
    if not isinstance(paths, Mapping) or set(paths) != set(
        FLASHINFER_RUNTIME_PATH_FIELDS
    ):
        raise OpsError(
            "environment-ready marker has incomplete FlashInfer runtime paths"
        )
    checked_paths: dict[str, str] = {}
    for field in FLASHINFER_RUNTIME_PATH_FIELDS:
        raw_path = paths.get(field)
        if not isinstance(raw_path, str) or not raw_path:
            raise OpsError(
                f"environment-ready marker has invalid FlashInfer path: {field}"
            )
        lexical = pathlib.Path(raw_path)
        if not lexical.is_absolute():
            raise OpsError(
                f"environment-ready marker FlashInfer path is not absolute: {field}"
            )
        try:
            canonical = lexical.resolve(strict=False)
            relative = canonical.relative_to(workspace)
        except (OSError, RuntimeError, ValueError) as exc:
            raise OpsError(
                f"environment-ready marker FlashInfer path escapes workspace: {field}"
            ) from exc
        if lexical != canonical or not relative.parts:
            raise OpsError(
                f"environment-ready marker FlashInfer path is not canonical: {field}"
            )
        checked_paths[field] = os.fspath(canonical)
    return {"ok": True, "paths": checked_paths}


def validate_environment_marker(
    marker_path: os.PathLike[str] | str,
    requirements_path: os.PathLike[str] | str,
    python_path: os.PathLike[str] | str,
    *,
    scratch: os.PathLike[str] | str,
    cuda_home: str,
    cuda_nvcc: str,
    cuda_nvcc_version: str,
    cuda_nvcc_sha256: str,
    runtime_ld_library_path: str,
    libcudart_path: str,
    libcudart_sha256: str,
    flashinfer_workspace_base: str,
    flashinfer_package_version: str,
    actual_packages: Mapping[str, str] | None = None,
    ffmpeg_path: str | None = None,
    ffmpeg_version: str | None = None,
) -> dict[str, Any]:
    marker_file = pathlib.Path(marker_path)
    requirements_file = pathlib.Path(requirements_path)
    marker = read_json(marker_file)
    if not isinstance(marker, Mapping) or marker.get("schema_version") != 2:
        raise OpsError(f"invalid environment-ready marker schema 2: {marker_file}")
    expected_requirements = _sha256_file(requirements_file)
    if marker.get("requirements_sha256") != expected_requirements:
        raise OpsError(
            "environment-ready marker requirements digest does not match "
            f"{requirements_file}"
        )
    recorded_python = marker.get("python_executable")
    try:
        expected_python = pathlib.Path(python_path).resolve(strict=True)
        actual_python = pathlib.Path(str(recorded_python)).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OpsError(f"cannot validate environment marker Python: {exc}") from exc
    if actual_python != expected_python:
        raise OpsError(
            "environment-ready marker Python does not match configured Python"
        )

    packages = marker.get("packages")
    if not isinstance(packages, Mapping) or not all(
        isinstance(name, str)
        and name
        and isinstance(version, str)
        and version
        for name, version in packages.items()
    ):
        raise OpsError("environment-ready marker has no valid package versions")
    if packages.get(FLASHINFER_DISTRIBUTION) != flashinfer_package_version:
        raise OpsError(
            "environment-ready marker FlashInfer package version is inconsistent"
        )
    if actual_packages is not None and dict(packages) != dict(actual_packages):
        raise OpsError(
            "current installed package versions do not match environment-ready marker"
        )

    canonical_workspace = _canonical_flashinfer_workspace(
        scratch, flashinfer_workspace_base
    )
    flashinfer_runtime = _validate_flashinfer_runtime_evidence(
        marker, canonical_workspace
    )
    exact_values = {
        "cuda_home": cuda_home,
        "cuda_nvcc": cuda_nvcc,
        "cuda_nvcc_version": cuda_nvcc_version,
        "cuda_nvcc_sha256": cuda_nvcc_sha256,
        "runtime_ld_library_path": runtime_ld_library_path,
        "libcudart_path": libcudart_path,
        "libcudart_sha256": libcudart_sha256,
        "flashinfer_workspace_base": os.fspath(canonical_workspace),
        "flashinfer_package_version": flashinfer_package_version,
    }
    for field, expected in exact_values.items():
        _require_exact_marker_value(marker, field, expected)
    if SHA256_RE.fullmatch(cuda_nvcc_sha256) is None or _sha256_file(
        pathlib.Path(cuda_nvcc)
    ) != cuda_nvcc_sha256:
        raise OpsError("current nvcc hash does not match environment-ready marker")
    if SHA256_RE.fullmatch(libcudart_sha256) is None or _sha256_file(
        pathlib.Path(libcudart_path)
    ) != libcudart_sha256:
        raise OpsError("current libcudart hash does not match environment-ready marker")

    if ffmpeg_path is not None:
        recorded_ffmpeg = marker.get("ffmpeg_path")
        try:
            if pathlib.Path(str(recorded_ffmpeg)).resolve(strict=True) != pathlib.Path(
                ffmpeg_path
            ).resolve(strict=True):
                raise OpsError(
                    "current FFmpeg executable does not match environment-ready marker"
                )
        except (OSError, RuntimeError) as exc:
            raise OpsError(f"cannot validate FFmpeg executable: {exc}") from exc
    if ffmpeg_version is not None and marker.get("ffmpeg_version") != ffmpeg_version:
        raise OpsError("current FFmpeg version does not match environment-ready marker")
    return {
        "path": os.fspath(marker_file.resolve(strict=True)),
        "requirements_path": os.fspath(requirements_file.resolve(strict=True)),
        "requirements_sha256": expected_requirements,
        "python_executable": os.fspath(expected_python),
        "packages": dict(packages),
        "ffmpeg_path": marker.get("ffmpeg_path"),
        "ffmpeg_version": marker.get("ffmpeg_version"),
        "flashinfer_runtime": flashinfer_runtime,
        **exact_values,
    }


def _run_text(
    runner: Runner,
    args: Sequence[str],
    *,
    timeout: int = 30,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    run_kwargs: dict[str, Any] = {
        "text": True,
        "capture_output": True,
        "timeout": timeout,
        "check": False,
    }
    if env is not None:
        run_kwargs["env"] = dict(env)
    try:
        return runner(list(args), **run_kwargs)
    except (OSError, subprocess.SubprocessError) as exc:
        raise OpsError(f"command failed to run: {args[0]}: {exc}") from exc


def _remove_temporary_directory(
    path: os.PathLike[str] | str,
    *,
    rmtree_fn: Callable[[os.PathLike[str] | str], None] = shutil.rmtree,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    directory = pathlib.Path(path)
    for attempt in range(TEMP_DIRECTORY_CLEANUP_ATTEMPTS):
        try:
            rmtree_fn(directory)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            retryable = exc.errno in {errno.EBUSY, errno.ENOTEMPTY}
            if not retryable or attempt + 1 == TEMP_DIRECTORY_CLEANUP_ATTEMPTS:
                raise OpsError(
                    f"cannot clean temporary directory after "
                    f"{attempt + 1} attempts: {directory}: {exc}"
                ) from exc
            sleep_fn(
                TEMP_DIRECTORY_CLEANUP_INITIAL_DELAY_SECONDS * (2**attempt)
            )


def probe_native_toolchain(
    scratch: os.PathLike[str] | str,
    *,
    compiler_command: str = "gcc",
    linker_command: str = "ld",
    which_fn: Callable[[str], str | None] = shutil.which,
    runner: Runner = subprocess.run,
) -> dict[str, str]:
    scratch_path = canonical_path_under_scratch(
        scratch, scratch, must_exist=True, allow_scratch=True
    )
    compiler_value = which_fn(compiler_command)
    if compiler_value is None:
        raise OpsError(f"required compiler is unavailable: {compiler_command}")
    linker_value = which_fn(linker_command)
    if linker_value is None:
        raise OpsError(f"required linker is unavailable: {linker_command}")
    try:
        compiler = pathlib.Path(compiler_value).resolve(strict=True)
        linker = pathlib.Path(linker_value).resolve(strict=True)
    except OSError as exc:
        raise OpsError(f"cannot resolve native toolchain executable: {exc}") from exc
    compiler_probe = _run_text(
        runner,
        [os.fspath(compiler), "--version"],
        timeout=TOOLCHAIN_TIMEOUT_SECONDS,
    )
    compiler_lines = (compiler_probe.stdout or compiler_probe.stderr).splitlines()
    if compiler_probe.returncode != 0 or not compiler_lines:
        detail = (compiler_probe.stderr or compiler_probe.stdout).strip()
        raise OpsError(f"compiler version probe failed: {detail}")
    linker_probe = _run_text(
        runner,
        [os.fspath(linker), "--version"],
        timeout=TOOLCHAIN_TIMEOUT_SECONDS,
    )
    linker_lines = (linker_probe.stdout or linker_probe.stderr).splitlines()
    if linker_probe.returncode != 0 or not linker_lines:
        detail = (linker_probe.stderr or linker_probe.stdout).strip()
        raise OpsError(f"linker version probe failed: {detail}")
    temporary_path = pathlib.Path(
        tempfile.mkdtemp(prefix=".oprd-toolchain.", dir=scratch_path)
    )
    try:
        source = temporary_path / "probe.c"
        output = temporary_path / "probe.so"
        source.write_text(
            "int oprd_toolchain_probe(void) { return 0; }\n", encoding="utf-8"
        )
        try:
            compiled = _run_text(
                runner,
                [
                    os.fspath(compiler),
                    os.fspath(source),
                    "-O2",
                    "-shared",
                    "-fPIC",
                    "-o",
                    os.fspath(output),
                ],
                timeout=TOOLCHAIN_TIMEOUT_SECONDS,
            )
        except OpsError as exc:
            raise OpsError(f"compiler/linker probe failed: {exc}") from exc
        if compiled.returncode != 0:
            detail = (compiled.stderr or compiled.stdout).strip()
            raise OpsError(f"compiler/linker probe failed: {detail}")
        if not output.is_file() or output.is_symlink() or output.stat().st_size == 0:
            raise OpsError("compiler/linker probe did not produce a shared object")
    finally:
        _remove_temporary_directory(temporary_path)
    return {
        "compiler_path": os.fspath(compiler),
        "compiler_version": compiler_lines[0],
        "linker_path": os.fspath(linker),
        "linker_version": linker_lines[0],
    }


def _validate_path_list(value: str, label: str) -> list[pathlib.Path]:
    if not value:
        raise OpsError(f"{label} must be nonempty")
    entries = value.split(os.pathsep)
    if any(not entry for entry in entries):
        raise OpsError(f"{label} contains an empty path entry")
    paths = [pathlib.Path(entry) for entry in entries]
    if any(not path.is_absolute() for path in paths):
        raise OpsError(f"{label} entries must be absolute")
    return paths


def _cuda_runtime_environment(
    *,
    runtime_path: str,
    cuda_home: pathlib.Path,
    runtime_ld_library_path: str,
) -> dict[str, str]:
    _validate_path_list(runtime_path, "runtime PATH")
    _validate_path_list(runtime_ld_library_path, "runtime LD_LIBRARY_PATH")
    return {
        "PATH": runtime_path,
        "CUDA_HOME": os.fspath(cuda_home),
        "LD_LIBRARY_PATH": runtime_ld_library_path,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LC_ALL": "C",
    }


def _nvcc_version_line(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if "Cuda compilation tools" in line:
            return line
    if lines:
        return lines[-1]
    raise OpsError("nvcc version probe returned no version text")


def probe_cuda_runtime(
    scratch: os.PathLike[str] | str,
    python_path: os.PathLike[str] | str,
    runtime_path: str,
    cuda_home: os.PathLike[str] | str,
    cuda_nvcc: os.PathLike[str] | str,
    runtime_ld_library_path: str,
    *,
    readelf_command: str = "readelf",
    which_fn: Callable[[str], str | None] = shutil.which,
    runner: Runner = subprocess.run,
) -> dict[str, str]:
    """Compile and load a minimal CUDA runtime object without touching a GPU."""

    scratch_path = canonical_path_under_scratch(
        scratch, scratch, must_exist=True, allow_scratch=True
    )
    raw_cuda_home = pathlib.Path(cuda_home)
    if not raw_cuda_home.is_absolute():
        raise OpsError("CUDA home must be absolute")
    try:
        canonical_cuda_home = raw_cuda_home.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OpsError(f"cannot resolve CUDA home: {exc}") from exc
    if raw_cuda_home != canonical_cuda_home or not canonical_cuda_home.is_dir():
        raise OpsError("CUDA home must be a canonical directory")

    expected_nvcc = canonical_cuda_home / "bin" / "nvcc"
    raw_nvcc = pathlib.Path(cuda_nvcc)
    if raw_nvcc != expected_nvcc:
        raise OpsError("configured nvcc must be exactly $CUDA_HOME/bin/nvcc")
    if not expected_nvcc.is_file() or not os.access(expected_nvcc, os.X_OK):
        raise OpsError(f"configured nvcc is missing or not executable: {expected_nvcc}")
    try:
        resolved_nvcc = expected_nvcc.resolve(strict=True)
        resolved_nvcc.relative_to(canonical_cuda_home)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OpsError(f"configured nvcc escapes CUDA home: {exc}") from exc

    lib64 = canonical_cuda_home / "lib64"
    if not lib64.is_dir():
        raise OpsError(f"CUDA lib64 directory is missing: {lib64}")
    try:
        resolved_lib64 = lib64.resolve(strict=True)
        resolved_lib64.relative_to(canonical_cuda_home)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OpsError(f"CUDA lib64 escapes CUDA home: {exc}") from exc
    ld_paths = _validate_path_list(
        runtime_ld_library_path, "runtime LD_LIBRARY_PATH"
    )
    try:
        first_ld_path = ld_paths[0].resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OpsError(f"cannot resolve first runtime library path: {exc}") from exc
    if first_ld_path != resolved_lib64:
        raise OpsError("runtime LD_LIBRARY_PATH must begin with CUDA_HOME/lib64")

    runtime_link = lib64 / CUDA_RUNTIME_SONAME
    if not runtime_link.is_file():
        raise OpsError(f"CUDA runtime library is missing: {runtime_link}")
    try:
        resolved_runtime = runtime_link.resolve(strict=True)
        resolved_runtime.relative_to(canonical_cuda_home)
    except (OSError, RuntimeError, ValueError) as exc:
        raise OpsError(f"CUDA runtime library escapes CUDA home: {exc}") from exc
    if not resolved_runtime.is_file():
        raise OpsError(f"resolved CUDA runtime is not a file: {resolved_runtime}")

    configured_python = pathlib.Path(python_path)
    if (
        not configured_python.is_absolute()
        or not configured_python.is_file()
        or not os.access(configured_python, os.X_OK)
    ):
        raise OpsError(
            f"configured Python is missing or not executable: {configured_python}"
        )
    readelf_value = which_fn(readelf_command)
    if readelf_value is None:
        raise OpsError(f"required ELF inspector is unavailable: {readelf_command}")
    try:
        readelf = pathlib.Path(readelf_value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise OpsError(f"cannot resolve ELF inspector: {exc}") from exc
    if not readelf.is_file() or not os.access(readelf, os.X_OK):
        raise OpsError(f"ELF inspector is not executable: {readelf}")

    child_environment = _cuda_runtime_environment(
        runtime_path=runtime_path,
        cuda_home=canonical_cuda_home,
        runtime_ld_library_path=runtime_ld_library_path,
    )
    nvcc_probe = _run_text(
        runner,
        [os.fspath(expected_nvcc), "--version"],
        timeout=CUDA_RUNTIME_TIMEOUT_SECONDS,
        env=child_environment,
    )
    if nvcc_probe.returncode != 0:
        detail = (nvcc_probe.stderr or nvcc_probe.stdout).strip()
        raise OpsError(f"nvcc version probe failed: {detail}")
    nvcc_version = _nvcc_version_line(nvcc_probe.stdout or nvcc_probe.stderr)

    with tempfile.TemporaryDirectory(
        prefix=".oprd-cuda-runtime.", dir=scratch_path
    ) as temporary:
        temporary_path = pathlib.Path(temporary)
        source = temporary_path / "probe.cu"
        output = temporary_path / "probe.so"
        source.write_text(
            "#include <cuda_runtime_api.h>\n"
            'extern "C" cudaError_t oprd_cuda_runtime_probe(int *version) {\n'
            "    return cudaRuntimeGetVersion(version);\n"
            "}\n",
            encoding="utf-8",
        )
        compiled = _run_text(
            runner,
            [
                os.fspath(expected_nvcc),
                os.fspath(source),
                "-shared",
                "--cudart=none",
                "-Xcompiler=-fPIC",
                f"-L{lib64}",
                "-lcudart",
                "-o",
                os.fspath(output),
            ],
            timeout=CUDA_RUNTIME_TIMEOUT_SECONDS,
            env=child_environment,
        )
        if compiled.returncode != 0:
            detail = (compiled.stderr or compiled.stdout).strip()
            raise OpsError(f"CUDA runtime compile probe failed: {detail}")
        if not output.is_file() or output.is_symlink() or output.stat().st_size == 0:
            raise OpsError("CUDA runtime compile probe did not produce a shared object")
        if output.stat().st_uid != os.geteuid():
            raise OpsError("CUDA runtime compile probe output is not owned by this user")

        inspected = _run_text(
            runner,
            [os.fspath(readelf), "-d", os.fspath(output)],
            timeout=CUDA_RUNTIME_TIMEOUT_SECONDS,
            env=child_environment,
        )
        if inspected.returncode != 0:
            detail = (inspected.stderr or inspected.stdout).strip()
            raise OpsError(f"readelf CUDA runtime inspection failed: {detail}")
        dynamic = inspected.stdout
        needed = re.findall(
            r"\(NEEDED\).*?Shared library:\s*\[([^]]+)\]", dynamic
        )
        if CUDA_RUNTIME_SONAME not in needed:
            raise OpsError(
                f"CUDA probe object does not require {CUDA_RUNTIME_SONAME}"
            )
        if re.search(r"\((?:RPATH|RUNPATH)\)", dynamic):
            raise OpsError("CUDA probe object contains forbidden RPATH or RUNPATH")

        loader_script = (
            "import ctypes,json,os,pathlib,sys; "
            "ctypes.CDLL(sys.argv[1],mode=os.RTLD_GLOBAL); "
            "ctypes.CDLL(sys.argv[2]); "
            "lines=pathlib.Path('/proc/self/maps').read_text().splitlines(); "
            "paths=sorted({str(pathlib.Path(line.split()[-1]).resolve()) "
            "for line in lines if line.split() and line.split()[-1].startswith('/') "
            "and pathlib.Path(line.split()[-1]).name.startswith(sys.argv[1])}); "
            "print(json.dumps({'mapped_libcudart_paths':paths},sort_keys=True))"
        )
        loaded = _run_text(
            runner,
            [
                os.fspath(configured_python),
                "-c",
                loader_script,
                CUDA_RUNTIME_SONAME,
                os.fspath(output),
            ],
            timeout=CUDA_RUNTIME_TIMEOUT_SECONDS,
            env=child_environment,
        )
        if loaded.returncode != 0:
            detail = (loaded.stderr or loaded.stdout).strip()
            raise OpsError(f"CUDA runtime load probe failed: {detail}")
        try:
            load_report = json.loads(loaded.stdout)
        except json.JSONDecodeError as exc:
            raise OpsError("CUDA runtime load probe returned invalid JSON") from exc
        expected_mapped = [os.fspath(resolved_runtime)]
        if (
            not isinstance(load_report, Mapping)
            or load_report.get("mapped_libcudart_paths") != expected_mapped
        ):
            raise OpsError(
                "mapped libcudart path does not match configured CUDA toolkit runtime"
            )

    return {
        "cuda_home": os.fspath(canonical_cuda_home),
        "cuda_nvcc": os.fspath(expected_nvcc),
        "cuda_nvcc_version": nvcc_version,
        "cuda_nvcc_sha256": _sha256_file(resolved_nvcc),
        "cuda_lib64": os.fspath(lib64),
        "runtime_ld_library_path": runtime_ld_library_path,
        "libcudart_path": os.fspath(resolved_runtime),
        "libcudart_sha256": _sha256_file(resolved_runtime),
    }


def probe_package_versions(
    python_path: pathlib.Path,
    package_names: Sequence[str],
    *,
    runner: Runner = subprocess.run,
) -> dict[str, str]:
    completed = _run_text(
        runner,
        [
            os.fspath(python_path),
            "-c",
            (
                "import importlib.metadata,json,sys; "
                "print(json.dumps({name:importlib.metadata.version(name) "
                "for name in sys.argv[1:]},sort_keys=True))"
            ),
            *package_names,
        ],
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise OpsError(f"cannot read installed package versions: {detail}")
    try:
        versions = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise OpsError("installed package version probe returned invalid JSON") from exc
    if not isinstance(versions, dict) or set(versions) != set(package_names):
        raise OpsError("installed package version probe returned an incomplete mapping")
    if not all(isinstance(value, str) and value for value in versions.values()):
        raise OpsError("installed package version probe returned an invalid version")
    return versions


def validate_current_environment(
    marker_path: os.PathLike[str] | str,
    requirements_path: os.PathLike[str] | str,
    python_path: os.PathLike[str] | str,
    *,
    scratch: os.PathLike[str] | str,
    runtime_path: str,
    cuda_home: os.PathLike[str] | str,
    cuda_nvcc: os.PathLike[str] | str,
    runtime_ld_library_path: str,
    flashinfer_workspace_base: os.PathLike[str] | str,
    ffmpeg_command: str = "ffmpeg",
    readelf_command: str = "readelf",
    which_fn: Callable[[str], str | None] = shutil.which,
    runner: Runner = subprocess.run,
    cuda_runtime_identity: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    marker_value = read_json(marker_path)
    marker_packages = (
        marker_value.get("packages")
        if isinstance(marker_value, Mapping)
        else None
    )
    if not isinstance(marker_packages, Mapping) or not all(
        isinstance(name, str) and name for name in marker_packages
    ):
        raise OpsError("environment-ready marker has no valid package mapping")
    if FLASHINFER_DISTRIBUTION not in marker_packages:
        raise OpsError("environment-ready marker has no FlashInfer package version")
    actual_packages = probe_package_versions(
        pathlib.Path(python_path), sorted(marker_packages), runner=runner
    )
    ffmpeg_executable = which_fn(ffmpeg_command)
    if ffmpeg_executable is None:
        raise OpsError(f"required command is unavailable: {ffmpeg_command}")
    ffmpeg_probe = _run_text(runner, [ffmpeg_executable, "-version"])
    ffmpeg_lines = (ffmpeg_probe.stdout or ffmpeg_probe.stderr).splitlines()
    if ffmpeg_probe.returncode != 0 or not ffmpeg_lines:
        raise OpsError("FFmpeg version probe failed")
    runtime_identity = dict(
        cuda_runtime_identity
        or probe_cuda_runtime(
            scratch,
            python_path,
            runtime_path,
            cuda_home,
            cuda_nvcc,
            runtime_ld_library_path,
            readelf_command=readelf_command,
            which_fn=which_fn,
            runner=runner,
        )
    )
    required_runtime_fields = {
        "cuda_home",
        "cuda_nvcc",
        "cuda_nvcc_version",
        "cuda_nvcc_sha256",
        "runtime_ld_library_path",
        "libcudart_path",
        "libcudart_sha256",
    }
    if not required_runtime_fields.issubset(runtime_identity) or not all(
        isinstance(runtime_identity[field], str) and runtime_identity[field]
        for field in required_runtime_fields
    ):
        raise OpsError("CUDA runtime identity is incomplete")
    workspace = _canonical_flashinfer_workspace(
        scratch, flashinfer_workspace_base
    )
    flashinfer_version = actual_packages[FLASHINFER_DISTRIBUTION]
    return validate_environment_marker(
        marker_path,
        requirements_path,
        python_path,
        scratch=scratch,
        cuda_home=runtime_identity["cuda_home"],
        cuda_nvcc=runtime_identity["cuda_nvcc"],
        cuda_nvcc_version=runtime_identity["cuda_nvcc_version"],
        cuda_nvcc_sha256=runtime_identity["cuda_nvcc_sha256"],
        runtime_ld_library_path=runtime_identity["runtime_ld_library_path"],
        libcudart_path=runtime_identity["libcudart_path"],
        libcudart_sha256=runtime_identity["libcudart_sha256"],
        flashinfer_workspace_base=os.fspath(workspace),
        flashinfer_package_version=flashinfer_version,
        actual_packages=actual_packages,
        ffmpeg_path=ffmpeg_executable,
        ffmpeg_version=ffmpeg_lines[0],
    )


def _parse_lfs_quota(output: str) -> dict[str, int] | None:
    for line in reversed(output.splitlines()):
        fields = line.split()
        if len(fields) < 8 or not fields[0].startswith("/"):
            continue
        try:
            used_kib = int(fields[1].rstrip("*"))
            soft_kib = int(fields[2].rstrip("*"))
            hard_kib = int(fields[3].rstrip("*"))
            used_inodes = int(fields[5].rstrip("*"))
            soft_inodes = int(fields[6].rstrip("*"))
            hard_inodes = int(fields[7].rstrip("*"))
        except ValueError:
            continue
        block_limit = hard_kib or soft_kib
        inode_limit = hard_inodes or soft_inodes
        if block_limit <= 0 or inode_limit <= 0:
            return None
        return {
            "remaining_bytes": max(0, block_limit - used_kib) * 1024,
            "remaining_inodes": max(0, inode_limit - used_inodes),
            "block_limit_kib": block_limit,
            "inode_limit": inode_limit,
        }
    return None


def probe_quota(
    scratch: pathlib.Path,
    *,
    user: str,
    min_free_bytes: int = 0,
    min_free_inodes: int = 0,
    which_fn: Callable[[str], str | None] = shutil.which,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Best-effort quota reporting without treating absence as success."""

    attempts: list[dict[str, Any]] = []
    candidates = (
        ("lfs", ["quota", "-u", user, os.fspath(scratch)]),
        ("quota", ["-s"]),
    )
    found = False
    for command, arguments in candidates:
        executable = which_fn(command)
        if executable is None:
            continue
        found = True
        completed = _run_text(runner, [executable, *arguments])
        output = (completed.stdout or completed.stderr).strip()
        attempts.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "output": output[-16384:],
            }
        )
        if completed.returncode == 0 and output:
            parsed = _parse_lfs_quota(output) if command == "lfs" else None
            if parsed is None:
                return {
                    "status": "reported_unparsed",
                    "command": command,
                    "output": output[-16384:],
                    "attempts": attempts,
                }
            sufficient = (
                parsed["remaining_bytes"] >= min_free_bytes
                and parsed["remaining_inodes"] >= min_free_inodes
            )
            return {
                "status": "reported" if sufficient else "insufficient",
                "command": command,
                "output": output[-16384:],
                "headroom": parsed,
                "attempts": attempts,
            }
    if not found:
        return {
            "status": "unavailable",
            "reason": "neither lfs nor quota is available",
            "attempts": [],
        }
    return {
        "status": "error",
        "reason": "available quota commands did not return a report",
        "attempts": attempts,
    }


def run_preflight(
    *,
    scratch: os.PathLike[str] | str,
    hf_home: os.PathLike[str] | str,
    publication_parent: os.PathLike[str] | str,
    python: os.PathLike[str] | str,
    required_imports: Sequence[str],
    required_scripts: Sequence[os.PathLike[str] | str],
    runtime_path: str,
    cuda_home: os.PathLike[str] | str,
    cuda_nvcc: os.PathLike[str] | str,
    runtime_ld_library_path: str,
    flashinfer_workspace_base: os.PathLike[str] | str,
    writable_paths: Sequence[os.PathLike[str] | str] = (),
    environment_marker: os.PathLike[str] | str | None = None,
    requirements_file: os.PathLike[str] | str | None = None,
    min_free_blocks: int = 0,
    min_free_inodes: int = 0,
    min_free_bytes: int = 0,
    sbatch_command: str = "sbatch",
    sacct_command: str = "sacct",
    ffmpeg_command: str = "ffmpeg",
    compiler_command: str = "gcc",
    linker_command: str = "ld",
    readelf_command: str = "readelf",
    quota_user: str | None = None,
    allow_quota_unavailable: bool = False,
    statvfs_fn: Callable[[os.PathLike[str] | str], Any] = os.statvfs,
    which_fn: Callable[[str], str | None] = shutil.which,
    runner: Runner = subprocess.run,
    writable_probe: Callable[[pathlib.Path], None] = _probe_writable_path,
    atomic_directory_probe: Callable[..., Any] = probe_atomic_directory_move,
) -> dict[str, Any]:
    """Run a fully parameterized cluster preflight and return a JSON report."""

    report: dict[str, Any] = {"ok": False, "checks": {}, "errors": []}
    errors: list[str] = report["errors"]
    scratch_path: pathlib.Path | None = None
    hf_path: pathlib.Path | None = None
    cuda_runtime_identity: dict[str, str] | None = None

    try:
        scratch_path = canonical_path_under_scratch(
            scratch, scratch, must_exist=True, allow_scratch=True
        )
        writable_probe(scratch_path)
        report["checks"]["scratch"] = {
            "path": os.fspath(scratch_path),
            "writable": True,
        }
    except OpsError as exc:
        errors.append(str(exc))

    if scratch_path is None:
        report["checks"]["atomic_directory_publish"] = {
            "ok": False,
            "error": "scratch validation failed",
        }
        errors.append("atomic directory publish: scratch validation failed")
    else:
        try:
            publish_parent = canonical_path_under_scratch(
                scratch_path,
                publication_parent,
                must_exist=True,
            )
            if not publish_parent.is_dir() or publish_parent.is_symlink():
                raise OpsError(
                    f"publication parent is not a real directory: {publish_parent}"
                )
            probe_result = atomic_directory_probe(
                publish_parent,
                trusted_root=scratch_path,
            )
            method = getattr(probe_result, "method", None)
            if not isinstance(method, str) or not method:
                raise OpsError(
                    f"atomic directory probe returned an invalid method: {method!r}"
                )
            report["checks"]["atomic_directory_publish"] = {
                "ok": True,
                "parent": os.fspath(publish_parent),
                "method": method,
            }
        except Exception as exc:
            report["checks"]["atomic_directory_publish"] = {
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
            errors.append(f"atomic directory publish: {exc}")

    if scratch_path is not None:
        try:
            hf_path = canonical_path_under_scratch(scratch, hf_home)
            if hf_path.exists() and not hf_path.is_dir():
                raise OpsError(f"HF_HOME exists but is not a directory: {hf_path}")
            writable_probe(hf_path)
            report["checks"]["hf_home"] = {
                "path": os.fspath(hf_path),
                "writable": True,
            }
        except OpsError as exc:
            errors.append(f"HF_HOME: {exc}")

        try:
            workspace = _canonical_flashinfer_workspace(
                scratch, flashinfer_workspace_base
            )
            writable_probe(workspace)
            report["checks"]["flashinfer_workspace_base"] = {
                "path": os.fspath(workspace),
                "writable": True,
            }
        except OpsError as exc:
            errors.append(f"FlashInfer workspace base: {exc}")

        checked_writable: list[str] = []
        for raw_path in writable_paths:
            try:
                candidate = canonical_path_under_scratch(scratch, raw_path)
                if candidate.exists() and not candidate.is_dir():
                    raise OpsError(
                        f"configured writable path exists but is not a directory: {candidate}"
                    )
                writable_probe(candidate)
                checked_writable.append(os.fspath(candidate))
            except OpsError as exc:
                errors.append(f"writable path {raw_path}: {exc}")
        report["checks"]["writable_paths"] = checked_writable

        try:
            report["checks"]["filesystem"] = check_statvfs_capacity(
                statvfs_fn(scratch_path),
                min_free_blocks=min_free_blocks,
                min_free_inodes=min_free_inodes,
                min_free_bytes=min_free_bytes,
            )
        except (OSError, OpsError) as exc:
            errors.append(f"filesystem capacity: {exc}")

    python_path = pathlib.Path(python)
    if not python_path.is_absolute():
        errors.append("configured Python path must be absolute")
    elif not python_path.is_file() or not os.access(python_path, os.X_OK):
        errors.append(f"configured Python is missing or not executable: {python_path}")
    else:
        try:
            completed = _run_text(runner, [os.fspath(python_path), "--version"])
            version_output = (completed.stdout or completed.stderr).strip()
            match = VERSION_RE.match(version_output)
            if completed.returncode != 0 or match is None:
                raise OpsError(f"cannot determine Python version: {version_output!r}")
            version = tuple(int(part) for part in match.groups())
            if version[:2] != (3, 12):
                raise OpsError(
                    f"configured Python must be 3.12.x, found {'.'.join(map(str, version))}"
                )
            report["checks"]["python"] = {
                "path": os.fspath(python_path),
                "version": ".".join(map(str, version)),
            }
            imports: dict[str, bool] = {}
            for module in required_imports:
                if not module or not all(
                    part.isidentifier() for part in module.split(".")
                ):
                    raise OpsError(f"invalid import name: {module!r}")
                imported = _run_text(
                    runner,
                    [
                        os.fspath(python_path),
                        "-c",
                        "import importlib,sys; importlib.import_module(sys.argv[1])",
                        module,
                    ],
                    timeout=PREFLIGHT_IMPORT_TIMEOUT_SECONDS,
                )
                if imported.returncode != 0:
                    detail = (imported.stderr or imported.stdout).strip()
                    raise OpsError(f"required import failed: {module}: {detail}")
                imports[module] = True
            report["checks"]["imports"] = imports
        except OpsError as exc:
            errors.append(str(exc))

    if scratch_path is not None:
        try:
            cuda_runtime_identity = probe_cuda_runtime(
                scratch_path,
                python_path,
                runtime_path,
                cuda_home,
                cuda_nvcc,
                runtime_ld_library_path,
                readelf_command=readelf_command,
                which_fn=which_fn,
                runner=runner,
            )
            report["checks"]["cuda_runtime"] = {
                "ok": True,
                **cuda_runtime_identity,
            }
        except OpsError as exc:
            report["checks"]["cuda_runtime"] = {
                "ok": False,
                "error": str(exc),
            }
            errors.append(f"CUDA runtime: {exc}")
    else:
        report["checks"]["cuda_runtime"] = {
            "ok": False,
            "error": "scratch validation failed",
        }
        errors.append("CUDA runtime: scratch validation failed")

    if (environment_marker is None) != (requirements_file is None):
        errors.append(
            "environment marker and requirements file must be supplied together"
        )
    elif environment_marker is not None and requirements_file is not None:
        try:
            if scratch_path is None:
                raise OpsError("scratch validation failed")
            marker_path = canonical_path_under_scratch(
                scratch, environment_marker, must_exist=True
            )
            requirements_path = canonical_path_under_scratch(
                scratch, requirements_file, must_exist=True
            )
            if cuda_runtime_identity is None:
                raise OpsError("CUDA runtime validation failed")
            report["checks"]["environment_marker"] = validate_current_environment(
                marker_path,
                requirements_path,
                python_path,
                scratch=scratch_path,
                runtime_path=runtime_path,
                cuda_home=cuda_home,
                cuda_nvcc=cuda_nvcc,
                runtime_ld_library_path=runtime_ld_library_path,
                flashinfer_workspace_base=flashinfer_workspace_base,
                ffmpeg_command=ffmpeg_command,
                readelf_command=readelf_command,
                which_fn=which_fn,
                runner=runner,
                cuda_runtime_identity=cuda_runtime_identity,
            )
        except OpsError as exc:
            errors.append(f"environment marker: {exc}")

    scripts: list[str] = []
    for raw_script in required_scripts:
        script = pathlib.Path(raw_script)
        if (
            not script.is_file()
            or script.is_symlink()
            or not os.access(script, os.R_OK)
        ):
            errors.append(f"required script is missing or unreadable: {script}")
        else:
            scripts.append(os.fspath(script.resolve()))
    report["checks"]["scripts"] = scripts

    commands: dict[str, str] = {}
    for command in (sbatch_command, sacct_command):
        executable = which_fn(command)
        if executable is None:
            errors.append(f"required command is unavailable: {command}")
        else:
            commands[command] = executable
    report["checks"]["commands"] = commands

    if scratch_path is not None:
        try:
            report["checks"]["toolchain"] = probe_native_toolchain(
                scratch_path,
                compiler_command=compiler_command,
                linker_command=linker_command,
                which_fn=which_fn,
                runner=runner,
            )
        except OpsError as exc:
            errors.append(f"native toolchain: {exc}")

    if scratch_path is None:
        report["quota"] = {
            "status": "unavailable",
            "reason": "scratch validation failed",
            "attempts": [],
        }
    else:
        try:
            report["quota"] = probe_quota(
                scratch_path,
                user=quota_user or os.environ.get("USER", ""),
                min_free_bytes=min_free_bytes,
                min_free_inodes=min_free_inodes,
                which_fn=which_fn,
                runner=runner,
            )
            quota_status = report["quota"].get("status")
            if quota_status == "error":
                errors.append(
                    "quota reporting failed: "
                    + str(report["quota"].get("reason", "unknown error"))
                )
            elif quota_status == "insufficient":
                errors.append("quota headroom is below the required blocks or inodes")
            elif quota_status in {"unavailable", "reported_unparsed"} and not allow_quota_unavailable:
                errors.append(
                    "quota headroom is unavailable or unparsed; use the explicit override only after manual quota verification"
                )
        except OpsError as exc:
            report["quota"] = {
                "status": "error",
                "reason": str(exc),
                "attempts": [],
            }
            errors.append(f"quota reporting failed: {exc}")

    report["ok"] = not errors
    return report


def claim_directory(artifact_dir: os.PathLike[str] | str) -> pathlib.Path:
    artifact = pathlib.Path(artifact_dir)
    return artifact.parent / f".{artifact.name}.claim"


def _claim_identity(
    artifact: pathlib.Path, run_id: str, record_path: pathlib.Path
) -> dict[str, str]:
    return {
        "artifact_dir": os.fspath(artifact),
        "run_id": run_id,
        "record_path": os.fspath(record_path),
    }


def claim_artifact(
    *,
    scratch: os.PathLike[str] | str,
    artifact_dir: os.PathLike[str] | str,
    run_id: str,
    record_path: os.PathLike[str] | str,
    allow_existing_artifact: bool = False,
    build_dir: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Atomically claim an artifact destination using a sibling directory."""

    if not run_id or not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise OpsError(f"unsafe run ID: {run_id!r}")
    artifact = canonical_path_under_scratch(scratch, artifact_dir)
    record = canonical_path_under_scratch(scratch, record_path)
    claim = claim_directory(artifact)
    canonical_path_under_scratch(scratch, claim)
    if not artifact.parent.is_dir():
        raise OpsError(f"artifact parent does not exist: {artifact.parent}")
    if allow_existing_artifact and (artifact.exists() or artifact.is_symlink()):
        if build_dir is None:
            raise ClaimConflictError(
                "existing-artifact claim requires the producing build directory"
            )
        validate_published_artifact(build_dir, artifact)

    identity = _claim_identity(artifact, run_id, record)
    try:
        claim.mkdir(mode=0o700)
    except FileExistsError:
        existing = read_json(claim / CLAIM_FILENAME)
        if not isinstance(existing, Mapping):
            raise ClaimConflictError(f"artifact claim is invalid: {claim}")
        existing_identity = {
            key: existing.get(key) for key in ("artifact_dir", "run_id", "record_path")
        }
        if existing_identity != identity:
            raise ClaimConflictError(
                f"artifact is claimed by another run or record: {claim}"
            )
        if allow_existing_artifact and (artifact.exists() or artifact.is_symlink()):
            if build_dir is None:
                raise ClaimConflictError(
                    "existing-artifact claim requires the producing build directory"
                )
            validate_published_artifact(build_dir, artifact)
        return {"ok": True, "claim_dir": os.fspath(claim), "reused": True}

    try:
        if artifact.exists() or artifact.is_symlink():
            if not allow_existing_artifact:
                raise ClaimConflictError(
                    f"artifact destination already exists: {artifact}"
                )
            if artifact.is_symlink() or not artifact.is_dir():
                raise ClaimConflictError(
                    f"existing artifact is not a real directory: {artifact}"
                )
            try:
                complete = (artifact / "COMPLETE").read_text(
                    encoding="ascii"
                ).strip().split()
            except (OSError, UnicodeError) as exc:
                raise ClaimConflictError(
                    f"existing artifact has no valid COMPLETE marker: {artifact}"
                ) from exc
            if (
                len(complete) != 2
                or complete[0] != "manifest_sha256"
                or SHA256_RE.fullmatch(complete[1]) is None
                or _sha256_file(artifact / "manifest.json") != complete[1]
            ):
                raise ClaimConflictError(
                    f"existing artifact is not manifest-authenticated: {artifact}"
                )
        atomic_write_json(
            claim / CLAIM_FILENAME,
            {**identity, "schema_version": SCHEMA_VERSION, "claimed_at": utc_now()},
        )
        _fsync_directory(artifact.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            (claim / CLAIM_FILENAME).unlink()
        with contextlib.suppress(OSError):
            claim.rmdir()
        raise
    return {"ok": True, "claim_dir": os.fspath(claim), "reused": False}


def validate_runtime_claim(
    *,
    scratch: os.PathLike[str] | str,
    paths: Sequence[os.PathLike[str] | str],
    artifact_dir: os.PathLike[str] | str,
    run_id: str,
    record_path: os.PathLike[str] | str,
    cuda_home: str | None = None,
    cuda_nvcc: str | None = None,
    runtime_ld_library_path: str | None = None,
    flashinfer_workspace_base: str | None = None,
) -> dict[str, Any]:
    artifact = canonical_path_under_scratch(scratch, artifact_dir)
    record = canonical_path_under_scratch(
        scratch, record_path, must_exist=True
    )
    checked = [
        os.fspath(canonical_path_under_scratch(scratch, path)) for path in paths
    ]
    claim = canonical_path_under_scratch(
        scratch, claim_directory(artifact), must_exist=True
    )
    marker = read_json(claim / CLAIM_FILENAME)
    identity = _claim_identity(artifact, run_id, record)
    if not isinstance(marker, Mapping) or any(
        marker.get(key) != value for key, value in identity.items()
    ):
        raise ClaimConflictError(f"runtime claim identity does not match: {claim}")
    recorded = _load_record(record)
    if (
        recorded.get("run_id") != run_id
        or recorded.get("artifact_dir") != os.fspath(artifact)
    ):
        raise ClaimConflictError("runtime record identity does not match claim")
    configuration = recorded.get("configuration", {})
    if not isinstance(configuration, Mapping):
        raise OpsError("record has no immutable configuration")
    supplied_runtime = {
        "cuda_home": cuda_home,
        "cuda_nvcc": cuda_nvcc,
        "runtime_ld_library_path": runtime_ld_library_path,
        "flashinfer_workspace_base": flashinfer_workspace_base,
    }
    for key, supplied in supplied_runtime.items():
        if supplied is not None and configuration.get(key) != supplied:
            raise ClaimConflictError(
                f"runtime configuration {key} does not match immutable record"
            )
    root = configuration.get("root")
    expected_source = (
        configuration.get("source_bundle_sha256")
        if isinstance(configuration, Mapping)
        else None
    )
    expected_environment = (
        configuration.get("environment_marker_sha256")
        if isinstance(configuration, Mapping)
        else None
    )
    if (
        not isinstance(root, str)
        or not isinstance(expected_source, str)
        or not isinstance(expected_environment, str)
    ):
        raise OpsError("record has no source or environment identity")
    environment_marker = pathlib.Path(root) / "venv" / ".oprd-env-ready.json"
    if _sha256_file(environment_marker) != expected_environment:
        raise OpsError("environment-ready marker changed after submission")
    actual_source = compute_source_bundle_sha256(root)
    if actual_source != expected_source:
        raise OpsError("bridge source bundle changed after submission")
    return {
        "ok": True,
        "artifact_dir": os.fspath(artifact),
        "record_path": os.fspath(record),
        "claim_dir": os.fspath(claim),
        "checked_paths": checked,
    }


def release_artifact_claim(
    *,
    scratch: os.PathLike[str] | str,
    artifact_dir: os.PathLike[str] | str,
    run_id: str,
    record_path: os.PathLike[str] | str,
) -> dict[str, Any]:
    """Explicitly release a claim only for the matching run and record."""

    artifact = canonical_path_under_scratch(scratch, artifact_dir)
    record = canonical_path_under_scratch(scratch, record_path)
    claim = canonical_path_under_scratch(
        scratch, claim_directory(artifact), must_exist=True
    )
    marker = claim / CLAIM_FILENAME
    existing = read_json(marker)
    identity = _claim_identity(artifact, run_id, record)
    if not isinstance(existing, Mapping) or any(
        existing.get(key) != value for key, value in identity.items()
    ):
        raise ClaimConflictError(f"claim identity does not match: {claim}")
    recorded = _load_record(record)
    if recorded.get("run_id") != run_id or recorded.get("artifact_dir") != os.fspath(
        artifact
    ):
        raise ClaimConflictError("record identity does not match the artifact claim")
    pending = [
        phase
        for phase in recorded["phase_order"]
        if recorded["phases"][phase].get("pending_submission") is not None
    ]
    if pending:
        raise ActiveJobsError(
            "unresolved submission intents prevent claim release: "
            + ", ".join(pending)
        )
    unresolved = [
        f"{phase}:{attempt['job_id']}"
        for phase, attempt in _latest_attempts(recorded)
        if attempt is not None
        and attempt.get("classification") not in {"succeeded", "failed"}
    ]
    if unresolved:
        raise ActiveJobsError(
            "reconcile every submitted job before releasing the claim: "
            + ", ".join(unresolved)
        )
    entries = list(claim.iterdir())
    if entries != [marker]:
        raise ClaimConflictError(f"claim directory contains unexpected files: {claim}")
    marker.unlink()
    claim.rmdir()
    _fsync_directory(artifact.parent)
    return {"ok": True, "released": os.fspath(claim)}


@contextlib.contextmanager
def _record_lock(
    record_path: pathlib.Path, *, create_parent: bool = False
) -> Iterable[None]:
    if not record_path.is_absolute():
        raise OpsError(f"record path must be absolute: {record_path}")
    if create_parent:
        record_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        if not record_path.is_file() or record_path.is_symlink():
            raise OpsError(f"record is missing or unsafe: {record_path}")
        preliminary = _validate_record(read_json(record_path), record_path)
        canonical = canonical_path_under_scratch(
            preliminary.get("scratch", ""), record_path, must_exist=True
        )
        if canonical != record_path.resolve(strict=True):
            raise OpsError(f"record path is not canonical: {record_path}")
        if not record_path.parent.is_dir():
            raise OpsError(f"record parent is missing: {record_path.parent}")
    lock_path = record_path.parent / f".{record_path.name}.lock"
    with lock_path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_record(record: Any, path: pathlib.Path | None = None) -> dict[str, Any]:
    label = path or "record"
    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise OpsError(f"invalid record schema: {label}")
    phase_order = record.get("phase_order")
    phases = record.get("phases")
    if (
        not isinstance(phase_order, list)
        or not phase_order
        or len(set(phase_order)) != len(phase_order)
        or not all(isinstance(phase, str) and phase for phase in phase_order)
        or not isinstance(phases, dict)
        or set(phases) != set(phase_order)
    ):
        raise OpsError(f"invalid phase structure: {label}")
    configuration = record.get("configuration", {})
    if not isinstance(configuration, dict) or not all(
        isinstance(key, str) and key and isinstance(value, str)
        for key, value in configuration.items()
    ):
        raise OpsError(f"invalid configuration: {label}")
    for phase in phase_order:
        phase_entry = phases[phase] if isinstance(phases[phase], dict) else None
        attempts = phase_entry.get("attempts") if isinstance(phase_entry, dict) else None
        if not isinstance(attempts, list):
            raise OpsError(f"invalid attempts for phase {phase}: {label}")
        pending = phase_entry.get("pending_submission")
        if pending is not None and (
            not isinstance(pending, dict)
            or not isinstance(pending.get("started_at"), str)
            or not isinstance(pending.get("script"), str)
            or not pending.get("script")
        ):
            raise OpsError(f"invalid pending submission for phase {phase}: {label}")
        if pending is not None:
            dependency = pending.get("dependency_job_id")
            if dependency is not None and (
                not isinstance(dependency, str)
                or JOB_ID_RE.fullmatch(dependency) is None
            ):
                raise OpsError(
                    f"invalid pending dependency for phase {phase}: {label}"
                )
        for index, attempt in enumerate(attempts, 1):
            if (
                not isinstance(attempt, dict)
                or attempt.get("attempt") != index
                or not isinstance(attempt.get("job_id"), str)
                or JOB_ID_RE.fullmatch(attempt["job_id"]) is None
            ):
                raise OpsError(f"invalid attempt {index} for phase {phase}: {label}")
            dependency = attempt.get("dependency_job_id")
            if dependency is not None and (
                not isinstance(dependency, str)
                or JOB_ID_RE.fullmatch(dependency) is None
            ):
                raise OpsError(
                    f"invalid dependency for attempt {index} of phase {phase}: {label}"
                )
            script = attempt.get("script")
            if script is not None and (not isinstance(script, str) or not script):
                raise OpsError(
                    f"invalid script for attempt {index} of phase {phase}: {label}"
                )
    if record.get("submission_status") not in {
        "initializing",
        "submitting",
        "submitted",
        "failed",
    }:
        raise OpsError(f"invalid submission_status: {label}")
    if record.get("dag_status") not in {
        "not_submitted",
        "incomplete",
        "active",
        "unresolved",
        "failed",
        "succeeded",
    }:
        raise OpsError(f"invalid dag_status: {label}")
    return record


def initialize_record(
    *,
    scratch: os.PathLike[str] | str,
    record_path: os.PathLike[str] | str,
    run_id: str,
    build_dir: os.PathLike[str] | str,
    artifact_dir: os.PathLike[str] | str,
    phase_order: Sequence[str] = DAG_PHASES,
    configuration: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Atomically initialize an append-only job-attempt record."""

    if not run_id or not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise OpsError(f"unsafe run ID: {run_id!r}")
    record_path = canonical_path_under_scratch(scratch, record_path)
    build_dir = canonical_path_under_scratch(scratch, build_dir)
    artifact_dir = canonical_path_under_scratch(scratch, artifact_dir)
    ordered = list(phase_order)
    if not ordered or len(set(ordered)) != len(ordered):
        raise OpsError("phase order must be nonempty and unique")
    now = utc_now()
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scratch": os.fspath(pathlib.Path(scratch).resolve()),
        "record_path": os.fspath(record_path),
        "build_dir": os.fspath(build_dir),
        "artifact_dir": os.fspath(artifact_dir),
        "claim_dir": os.fspath(claim_directory(artifact_dir)),
        "configuration": dict(configuration or {}),
        "created_at": now,
        "updated_at": now,
        "submission_status": "initializing",
        "dag_status": "not_submitted",
        "phase_order": ordered,
        "phases": {phase: {"attempts": []} for phase in ordered},
    }
    _validate_record(record)
    with _record_lock(record_path, create_parent=True):
        if record_path.exists() or record_path.is_symlink():
            raise OpsError(f"record already exists: {record_path}")
        atomic_write_json(record_path, record)
    return record


def _load_record(path: pathlib.Path) -> dict[str, Any]:
    record = _validate_record(read_json(path), path)
    if not path.is_absolute():
        raise OpsError(f"record path must be absolute: {path}")
    try:
        actual = path.resolve(strict=True)
        recorded = pathlib.Path(record.get("record_path", "")).resolve(strict=True)
        canonical_path_under_scratch(
            record.get("scratch", ""), actual, must_exist=True
        )
    except (OSError, RuntimeError, OpsError) as exc:
        raise OpsError(f"record path safety validation failed: {path}: {exc}") from exc
    if actual != recorded:
        raise OpsError(
            f"record was opened through a path that does not match record_path: {path}"
        )
    for key in ("build_dir", "artifact_dir", "claim_dir"):
        raw_value = record.get(key)
        if not isinstance(raw_value, str):
            raise OpsError(f"record has no valid {key}: {path}")
        canonical = canonical_path_under_scratch(record["scratch"], raw_value)
        if os.fspath(canonical) != raw_value:
            raise OpsError(f"record {key} is not canonical: {raw_value}")
    return record


def begin_job_submission(
    record_path: os.PathLike[str] | str,
    *,
    phase: str,
    dependency_job_id: str | None,
    script: str,
) -> dict[str, Any]:
    if dependency_job_id is not None and JOB_ID_RE.fullmatch(
        dependency_job_id
    ) is None:
        raise OpsError(f"invalid Slurm dependency job ID: {dependency_job_id!r}")
    if not script:
        raise OpsError("submission script must be nonempty")
    path = pathlib.Path(record_path)
    with _record_lock(path):
        record = _load_record(path)
        if phase not in record["phases"]:
            raise OpsError(f"unknown phase: {phase}")
        phase_entry = record["phases"][phase]
        if phase_entry.get("pending_submission") is not None:
            raise OpsError(f"phase already has an unresolved submission intent: {phase}")
        pending = {"started_at": utc_now(), "script": script}
        if dependency_job_id is not None:
            pending["dependency_job_id"] = dependency_job_id
        phase_entry["pending_submission"] = pending
        record["submission_status"] = "submitting"
        record["dag_status"] = "unresolved"
        record["updated_at"] = utc_now()
        atomic_write_json(path, record)
        return record


def record_job(
    record_path: os.PathLike[str] | str,
    *,
    phase: str,
    job_id: str,
    dependency_job_id: str | None = None,
    script: str | None = None,
) -> dict[str, Any]:
    """Append a Slurm job attempt while preserving every previous attempt."""

    path = pathlib.Path(record_path)
    if JOB_ID_RE.fullmatch(job_id) is None:
        raise OpsError(f"invalid Slurm job ID: {job_id!r}")
    if dependency_job_id is not None and JOB_ID_RE.fullmatch(
        dependency_job_id
    ) is None:
        raise OpsError(f"invalid Slurm dependency job ID: {dependency_job_id!r}")
    if script is not None and not script:
        raise OpsError("recorded script must be nonempty")
    with _record_lock(path):
        record = _load_record(path)
        if phase not in record["phases"]:
            raise OpsError(f"unknown phase: {phase}")
        phase_entry = record["phases"][phase]
        pending = phase_entry.get("pending_submission")
        if pending is not None:
            pending_dependency = pending.get("dependency_job_id")
            pending_script = pending.get("script")
            if dependency_job_id is None:
                dependency_job_id = pending_dependency
            if script is None:
                script = pending_script
            if (
                dependency_job_id != pending_dependency
                or script != pending_script
            ):
                raise OpsError(
                    f"recorded job does not match pending submission intent: {phase}"
                )
        for recorded_phase in record["phase_order"]:
            for attempt in record["phases"][recorded_phase]["attempts"]:
                if attempt["job_id"] == job_id:
                    if recorded_phase == phase and attempt is record["phases"][phase]["attempts"][-1]:
                        return record
                    raise OpsError(
                        f"job ID {job_id} is already recorded for {recorded_phase}"
                    )
        attempts = record["phases"][phase]["attempts"]
        attempt = {
            "attempt": len(attempts) + 1,
            "job_id": job_id,
            "recorded_at": utc_now(),
        }
        if dependency_job_id is not None:
            attempt["dependency_job_id"] = dependency_job_id
        if script is not None:
            attempt["script"] = script
        attempts.append(attempt)
        phase_entry.pop("pending_submission", None)
        record["submission_status"] = "submitting"
        record["dag_status"] = "incomplete"
        record["updated_at"] = utc_now()
        atomic_write_json(path, record)
        return record


def mark_submission_failed(
    record_path: os.PathLike[str] | str, *, error: str
) -> dict[str, Any]:
    path = pathlib.Path(record_path)
    with _record_lock(path):
        record = _load_record(path)
        record["submission_status"] = "failed"
        if record["dag_status"] == "not_submitted":
            record["dag_status"] = "incomplete"
        record["submission_error"] = error
        record["updated_at"] = utc_now()
        atomic_write_json(path, record)
        return record


def mark_submitted(record_path: os.PathLike[str] | str) -> dict[str, Any]:
    path = pathlib.Path(record_path)
    with _record_lock(path):
        record = _load_record(path)
        pending = [
            phase
            for phase in record["phase_order"]
            if record["phases"][phase].get("pending_submission") is not None
        ]
        if pending:
            raise OpsError(
                "cannot mark submitted with unresolved submission intents: "
                + ", ".join(pending)
            )
        missing = [
            phase
            for phase in record["phase_order"]
            if not record["phases"][phase]["attempts"]
        ]
        if missing:
            raise OpsError(
                "cannot mark submitted with phases lacking jobs: " + ", ".join(missing)
            )
        record["submission_status"] = "submitted"
        record["dag_status"] = "active"
        record["submitted_at"] = record.get("submitted_at", utc_now())
        record["updated_at"] = utc_now()
        atomic_write_json(path, record)
        return record


def _normalize_slurm_state(state: str) -> str:
    token = state.strip().upper().split(maxsplit=1)[0] if state.strip() else ""
    return token.rstrip("+")


def classify_slurm_state(state: str, exit_code: str) -> dict[str, str]:
    """Classify a Slurm state, failing closed on unknown or malformed values."""

    normalized = _normalize_slurm_state(state)
    if normalized in ACTIVE_STATES:
        return {
            "classification": "active",
            "state": normalized,
            "exit_code": exit_code,
            "reason": "job is active",
        }
    if normalized == "COMPLETED":
        if exit_code == "0:0":
            return {
                "classification": "succeeded",
                "state": normalized,
                "exit_code": exit_code,
                "reason": "completed with zero exit code",
            }
        return {
            "classification": "failed",
            "state": normalized,
            "exit_code": exit_code,
            "reason": "COMPLETED job has a nonzero or malformed exit code",
        }
    if normalized in FAILURE_STATES:
        return {
            "classification": "failed",
            "state": normalized,
            "exit_code": exit_code,
            "reason": f"terminal failure state {normalized}",
        }
    return {
        "classification": "unresolved",
        "state": normalized or "UNKNOWN",
        "exit_code": exit_code,
        "reason": "missing or unknown Slurm state requires later reconciliation",
    }


def parse_sacct_output(
    output: str, expected_job_ids: Sequence[str]
) -> dict[str, dict[str, str]]:
    """Parse pipe-delimited JobIDRaw,State,ExitCode for exact job IDs only."""

    expected = list(expected_job_ids)
    if len(set(expected)) != len(expected):
        raise OpsError("expected Slurm job IDs must be unique")
    rows: dict[str, tuple[str, str]] = {}
    expected_set = set(expected)
    for line_number, line in enumerate(output.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split("|")
        if len(fields) < 3:
            raise OpsError(f"malformed sacct line {line_number}: {line!r}")
        job_id, state, exit_code = (field.strip() for field in fields[:3])
        if job_id not in expected_set:
            continue
        if job_id in rows:
            raise OpsError(f"duplicate exact sacct row for job ID {job_id}")
        rows[job_id] = (state, exit_code)

    parsed: dict[str, dict[str, str]] = {}
    for job_id in expected:
        if job_id not in rows:
            parsed[job_id] = {
                "classification": "unresolved",
                "state": "MISSING",
                "exit_code": "",
                "reason": "job ID is missing from sacct output; accounting may be delayed",
            }
        else:
            parsed[job_id] = classify_slurm_state(*rows[job_id])
    return parsed


def query_sacct(
    job_ids: Sequence[str],
    *,
    sacct_command: str = "sacct",
    runner: Runner = subprocess.run,
) -> dict[str, dict[str, str]]:
    if not job_ids:
        return {}
    completed = _run_text(
        runner,
        [
            sacct_command,
            "-n",
            "-P",
            "-X",
            "-j",
            ",".join(job_ids),
            "--format=JobIDRaw,State,ExitCode",
        ],
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise OpsError(f"sacct failed with exit {completed.returncode}: {detail}")
    return parse_sacct_output(completed.stdout, job_ids)


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise OpsError(f"cannot hash required file {path}: {exc}") from exc
    return digest.hexdigest()


def _sha256_tree(path: pathlib.Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir() or path.is_symlink():
        raise OpsError(f"cannot checksum missing or unsafe output: {path}")
    digest = hashlib.sha256()
    for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


def compute_source_bundle_sha256(
    root: os.PathLike[str] | str,
    relative_paths: Sequence[str] = SOURCE_BUNDLE_PATHS,
) -> str:
    repository = pathlib.Path(root).resolve(strict=True)
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = (repository / relative).resolve(strict=True)
        try:
            path.relative_to(repository)
        except ValueError as exc:
            raise OpsError(f"source bundle path escapes repository: {path}") from exc
        name = path.name.encode()
        payload = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str, *, resume_phase: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SuccessValidationError(
            f"{label} is not a JSON object", resume_phase=resume_phase
        )
    return value


def _read_success_json(
    path: pathlib.Path, label: str, *, resume_phase: str
) -> Mapping[str, Any]:
    try:
        value = read_json(path)
    except OpsError as exc:
        raise SuccessValidationError(
            f"cannot read {label}: {exc}", resume_phase=resume_phase
        ) from exc
    return _require_mapping(value, label, resume_phase=resume_phase)


def _hash_success_file(path: pathlib.Path, *, resume_phase: str) -> str:
    try:
        return _sha256_file(path)
    except OpsError as exc:
        raise SuccessValidationError(str(exc), resume_phase=resume_phase) from exc


def _smoke_hash(marker: Mapping[str, Any], flat_key: str, filename: str) -> Any:
    if flat_key in marker:
        return marker.get(flat_key)
    files = marker.get("files")
    if isinstance(files, Mapping):
        entry = files.get(filename)
        if isinstance(entry, Mapping):
            return entry.get("sha256")
        if isinstance(entry, str):
            return entry
    return None


def _validate_success_evidence(
    build_dir: os.PathLike[str] | str,
    artifact_dir: os.PathLike[str] | str,
    *,
    required_builder_phases: Sequence[str] = BUILDER_REQUIRED_PHASES,
    include_smoke: bool,
) -> dict[str, Any]:
    """Validate builder and artifact evidence, optionally including Stage 2."""

    build = pathlib.Path(build_dir)
    artifact = pathlib.Path(artifact_dir)
    state = _read_success_json(
        build / "state.json", "builder state", resume_phase="corpus"
    )
    phases = _require_mapping(
        state.get("phases"), "builder phases", resume_phase="corpus"
    )
    for phase in required_builder_phases:
        if phase in {"init", "generate-corpus"}:
            resume_phase = "corpus"
        elif phase in {"fit-teacher-pca", "materialize-teacher-targets"}:
            resume_phase = "teacher"
        elif phase.startswith("train-student-group-"):
            resume_phase = phase.removeprefix("train-")
        else:
            resume_phase = "publish"
        entry = phases.get(phase)
        if not isinstance(entry, Mapping) or entry.get("status") != "complete":
            raise SuccessValidationError(
                f"required builder phase is not complete: {phase}",
                resume_phase=resume_phase,
            )
        outputs = entry.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise SuccessValidationError(
                f"required builder phase has no committed outputs: {phase}",
                resume_phase=resume_phase,
            )
        for output in outputs:
            if not isinstance(output, Mapping):
                raise SuccessValidationError(
                    f"builder phase has an invalid output entry: {phase}",
                    resume_phase=resume_phase,
                )
            raw_path = output.get("path")
            expected_hash = output.get("sha256")
            if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
                raise SuccessValidationError(
                    f"builder phase has incomplete output metadata: {phase}",
                    resume_phase=resume_phase,
                )
            candidate = pathlib.Path(raw_path)
            if not candidate.is_absolute():
                candidate = build / candidate
            try:
                actual_hash = _sha256_tree(candidate)
            except OpsError as exc:
                raise SuccessValidationError(
                    f"builder phase output is unavailable: {phase}: {exc}",
                    resume_phase=resume_phase,
                ) from exc
            if actual_hash != expected_hash:
                raise SuccessValidationError(
                    f"builder phase output checksum failed: {phase}: {candidate}",
                    resume_phase=resume_phase,
                )

    verify = _read_success_json(
        build / "verify.json", "verify.json", resume_phase="publish"
    )
    if verify.get("status") != "ok":
        raise SuccessValidationError(
            "verify.json does not report status=ok", resume_phase="publish"
        )
    digest = verify.get("artifact_manifest_sha256")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise SuccessValidationError(
            "verify.json has no valid artifact manifest SHA-256",
            resume_phase="publish",
        )
    checked = verify.get("checked_phases")
    expected_checked = [
        phase for phase in required_builder_phases if phase != "verify"
    ]
    if checked != expected_checked:
        raise SuccessValidationError(
            "verify.json checked_phases does not match required builder phases",
            resume_phase="publish",
        )
    try:
        verified_artifact = pathlib.Path(str(verify.get("artifact_dir"))).resolve(
            strict=True
        )
        expected_artifact = artifact.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SuccessValidationError(
            f"artifact directory is unavailable: {exc}", resume_phase="publish"
        ) from exc
    if verified_artifact != expected_artifact:
        raise SuccessValidationError(
            "verify.json artifact_dir does not match the recorded artifact",
            resume_phase="publish",
        )
    if artifact.is_symlink() or not artifact.is_dir():
        raise SuccessValidationError(
            "artifact directory is missing or is a symlink", resume_phase="publish"
        )

    complete_path = artifact / "COMPLETE"
    try:
        complete_parts = complete_path.read_text(encoding="ascii").strip().split()
    except (OSError, UnicodeError) as exc:
        raise SuccessValidationError(
            f"cannot read artifact COMPLETE: {exc}", resume_phase="publish"
        ) from exc
    if complete_parts != ["manifest_sha256", digest]:
        raise SuccessValidationError(
            "artifact COMPLETE does not match verify.json digest",
            resume_phase="publish",
        )
    if _hash_success_file(artifact / "manifest.json", resume_phase="publish") != digest:
        raise SuccessValidationError(
            "artifact manifest.json does not match the verified digest",
            resume_phase="publish",
        )
    published_result = {
        "ok": True,
        "artifact_manifest_sha256": digest,
        "builder_phases": list(required_builder_phases),
    }
    if not include_smoke:
        return published_result

    smoke = build / SMOKE_DIRECTORY
    marker = _read_success_json(
        smoke / SMOKE_COMPLETE_FILENAME,
        SMOKE_COMPLETE_FILENAME,
        resume_phase="stage2-smoke",
    )
    if marker.get("schema_version") != 2 or marker.get("status") != "ok":
        raise SuccessValidationError(
            "SMOKE_COMPLETE.json does not report the current validated schema",
            resume_phase="stage2-smoke",
        )
    if (
        marker.get("fresh_uninterrupted") is not True
        or marker.get("student_tensors_finite") is not True
        or marker.get("state_step") != 2
    ):
        raise SuccessValidationError(
            "SMOKE_COMPLETE.json lacks fresh finite step-2 validation",
            resume_phase="stage2-smoke",
        )
    if marker.get("artifact_manifest_sha256") != digest:
        raise SuccessValidationError(
            "SMOKE_COMPLETE.json has the wrong artifact digest",
            resume_phase="stage2-smoke",
        )
    results_path = smoke / "results.json"
    checkpoint_path = smoke / "train_state_last.pt"
    expected_results_hash = _smoke_hash(marker, "results_sha256", "results.json")
    expected_checkpoint_hash = _smoke_hash(
        marker, "train_state_last_sha256", "train_state_last.pt"
    )
    if expected_checkpoint_hash is None:
        expected_checkpoint_hash = marker.get("checkpoint_sha256")
    if expected_results_hash != _hash_success_file(
        results_path, resume_phase="stage2-smoke"
    ):
        raise SuccessValidationError(
            "SMOKE_COMPLETE.json results hash does not match results.json",
            resume_phase="stage2-smoke",
        )
    if expected_checkpoint_hash != _hash_success_file(
        checkpoint_path, resume_phase="stage2-smoke"
    ):
        raise SuccessValidationError(
            "SMOKE_COMPLETE.json checkpoint hash does not match train_state_last.pt",
            resume_phase="stage2-smoke",
        )
    results = _read_success_json(
        results_path, "Stage-2 results.json", resume_phase="stage2-smoke"
    )
    bridge_reference = results.get("bridge_reference")
    if (
        not isinstance(bridge_reference, Mapping)
        or bridge_reference.get("manifest_sha256") != digest
    ):
        raise SuccessValidationError(
            "Stage-2 results.json has the wrong bridge manifest identity",
            resume_phase="stage2-smoke",
        )
    losses = results.get("losses")
    if (
        not isinstance(losses, list)
        or len(losses) != 2
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in losses
        )
    ):
        raise SuccessValidationError(
            "Stage-2 results.json lacks exactly two finite losses",
            resume_phase="stage2-smoke",
        )
    if results.get("start_step") != 0 or results.get("resumed_from") is not None:
        raise SuccessValidationError(
            "Stage-2 results.json is not a fresh uninterrupted smoke",
            resume_phase="stage2-smoke",
        )
    stage2_identity = bridge_reference.get("stage2_identity")
    if not isinstance(stage2_identity, Mapping):
        raise SuccessValidationError(
            "Stage-2 results.json has no Stage-2 identity",
            resume_phase="stage2-smoke",
        )
    dataset_revision = stage2_identity.get("dataset_revision")
    expected_stage2 = {
        "prompts": "BytedTsinghua-SIA/DAPO-Math-17k",
        "dataset_revision": dataset_revision,
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
    if (
        not isinstance(dataset_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", dataset_revision) is None
        or dict(stage2_identity) != expected_stage2
    ):
        raise SuccessValidationError(
            "Stage-2 results.json has the wrong smoke identity",
            resume_phase="stage2-smoke",
        )
    identity_digest = hashlib.sha256(
        json.dumps(
            dict(stage2_identity), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if marker.get("stage2_identity_sha256") != identity_digest:
        raise SuccessValidationError(
            "SMOKE_COMPLETE.json has the wrong Stage-2 identity digest",
            resume_phase="stage2-smoke",
        )
    return {
        **published_result,
        "results_sha256": expected_results_hash,
        "train_state_last_sha256": expected_checkpoint_hash,
    }


def validate_published_artifact(
    build_dir: os.PathLike[str] | str,
    artifact_dir: os.PathLike[str] | str,
) -> dict[str, Any]:
    return _validate_success_evidence(
        build_dir, artifact_dir, include_smoke=False
    )


def validate_success(
    build_dir: os.PathLike[str] | str,
    artifact_dir: os.PathLike[str] | str,
) -> dict[str, Any]:
    return _validate_success_evidence(
        build_dir, artifact_dir, include_smoke=True
    )


def _latest_attempts(record: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any] | None]]:
    latest: list[tuple[str, Mapping[str, Any] | None]] = []
    for phase in record["phase_order"]:
        attempts = record["phases"][phase]["attempts"]
        latest.append((phase, attempts[-1] if attempts else None))
    return latest


def determine_resume_start_index(record: Mapping[str, Any]) -> int:
    """Return the first phase to retry, refusing any currently active job."""

    validated = _validate_record(dict(record))
    pending = [
        phase
        for phase in validated["phase_order"]
        if validated["phases"][phase].get("pending_submission") is not None
    ]
    if pending:
        raise ActiveJobsError(
            "unresolved submission intents prevent resume: " + ", ".join(pending)
        )
    latest = _latest_attempts(validated)
    active = [
        f"{phase}:{attempt['job_id']}:{attempt.get('classification')}"
        for phase, attempt in latest
        if attempt is not None
        and attempt.get("classification") in {"active", "unresolved"}
    ]
    if active:
        raise ActiveJobsError(
            "active or unresolved jobs prevent resume: " + ", ".join(active)
        )
    for index, (_, attempt) in enumerate(latest):
        if attempt is None or attempt.get("classification") != "succeeded":
            return index
    validation = validated.get("success_validation")
    if isinstance(validation, Mapping) and validation.get("ok") is False:
        resume_phase = validation.get("resume_phase")
        if resume_phase in validated["phase_order"]:
            return validated["phase_order"].index(resume_phase)
        return len(validated["phase_order"]) - 1
    return len(validated["phase_order"])


def reconcile_record(
    record_path: os.PathLike[str] | str,
    *,
    sacct_command: str = "sacct",
    runner: Runner = subprocess.run,
    success_validator: Callable[[os.PathLike[str] | str, os.PathLike[str] | str], dict[str, Any]] = validate_success,
) -> dict[str, Any]:
    """Query exact recorded IDs, update attempts, and reconcile the DAG status."""

    path = pathlib.Path(record_path)
    with _record_lock(path):
        record = _load_record(path)
        job_ids = [
            attempt["job_id"]
            for phase in record["phase_order"]
            for attempt in record["phases"][phase]["attempts"]
        ]
        statuses = query_sacct(
            job_ids, sacct_command=sacct_command, runner=runner
        )
        checked_at = utc_now()
        for phase in record["phase_order"]:
            for attempt in record["phases"][phase]["attempts"]:
                status = statuses[attempt["job_id"]]
                attempt.update(
                    {
                        "classification": status["classification"],
                        "slurm_state": status["state"],
                        "exit_code": status["exit_code"],
                        "status_reason": status["reason"],
                        "status_checked_at": checked_at,
                    }
                )

        latest = _latest_attempts(record)
        missing_phase = any(attempt is None for _, attempt in latest)
        classifications = [
            attempt.get("classification")
            for _, attempt in latest
            if attempt is not None
        ]
        record.pop("success_validation", None)
        has_pending_submission = any(
            record["phases"][phase].get("pending_submission") is not None
            for phase in record["phase_order"]
        )
        if has_pending_submission:
            record["dag_status"] = "unresolved"
        elif missing_phase:
            record["dag_status"] = (
                "failed"
                if record["submission_status"] == "submitted"
                else "incomplete"
            )
        elif "failed" in classifications:
            record["dag_status"] = "failed"
        elif "unresolved" in classifications:
            record["dag_status"] = "unresolved"
        elif "active" in classifications:
            record["dag_status"] = "active"
        elif classifications and all(value == "succeeded" for value in classifications):
            try:
                validation = success_validator(
                    record["build_dir"], record["artifact_dir"]
                )
            except SuccessValidationError as exc:
                record["dag_status"] = "failed"
                record["success_validation"] = {
                    "ok": False,
                    "error": str(exc),
                    "resume_phase": exc.resume_phase,
                }
            except OpsError as exc:
                record["dag_status"] = "failed"
                record["success_validation"] = {
                    "ok": False,
                    "error": str(exc),
                    "resume_phase": "publish",
                }
            else:
                record["dag_status"] = "succeeded"
                record["success_validation"] = validation
        else:
            record["dag_status"] = "failed"
        record["updated_at"] = utc_now()
        atomic_write_json(path, record)
        return record


def _json_print(value: Any, *, stream: Any = sys.stdout) -> None:
    print(json.dumps(value, sort_keys=True, separators=(",", ":")), file=stream)


def _parse_config_values(values: Sequence[str]) -> dict[str, str]:
    configuration: dict[str, str] = {}
    for raw in values:
        key, separator, value = raw.partition("=")
        if not separator or not key or key in configuration:
            raise OpsError(f"invalid or duplicate configuration entry: {raw!r}")
        configuration[key] = value
    return configuration


def _add_scratch(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scratch", required=True)


def _add_cuda_runtime_arguments(
    parser: argparse.ArgumentParser, *, include_flashinfer: bool
) -> None:
    parser.add_argument("--runtime-path", required=True)
    parser.add_argument("--cuda-home", required=True)
    parser.add_argument("--cuda-nvcc", required=True)
    parser.add_argument("--runtime-ld-library-path", required=True)
    if include_flashinfer:
        parser.add_argument("--flashinfer-workspace-base", required=True)
    parser.add_argument("--readelf-command", default="readelf")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    _add_scratch(preflight)
    preflight.add_argument("--hf-home", required=True)
    preflight.add_argument("--publication-parent", required=True)
    preflight.add_argument("--python", required=True)
    _add_cuda_runtime_arguments(preflight, include_flashinfer=True)
    preflight.add_argument("--import", dest="required_imports", action="append", default=[])
    preflight.add_argument("--script", dest="required_scripts", action="append", default=[])
    preflight.add_argument("--writable-path", action="append", default=[])
    preflight.add_argument("--environment-marker")
    preflight.add_argument("--requirements-file")
    preflight.add_argument("--min-free-blocks", type=int, default=0)
    preflight.add_argument("--min-free-inodes", type=int, default=0)
    preflight.add_argument("--min-free-bytes", type=int, default=0)
    preflight.add_argument("--sbatch-command", default="sbatch")
    preflight.add_argument("--sacct-command", default="sacct")
    preflight.add_argument("--ffmpeg-command", default="ffmpeg")
    preflight.add_argument("--compiler-command", default="gcc")
    preflight.add_argument("--linker-command", default="ld")
    preflight.add_argument("--quota-user")
    preflight.add_argument("--allow-quota-unavailable", action="store_true")

    source_bundle = subparsers.add_parser("source-bundle-sha256")
    source_bundle.add_argument("--root", required=True)

    paths = subparsers.add_parser("check-paths")
    _add_scratch(paths)
    paths.add_argument("--path", action="append", required=True)

    toolchain = subparsers.add_parser("check-toolchain")
    _add_scratch(toolchain)
    toolchain.add_argument("--compiler-command", default="gcc")
    toolchain.add_argument("--linker-command", default="ld")

    cuda_runtime = subparsers.add_parser("check-cuda-runtime")
    _add_scratch(cuda_runtime)
    cuda_runtime.add_argument("--python", required=True)
    _add_cuda_runtime_arguments(cuda_runtime, include_flashinfer=False)

    claim = subparsers.add_parser("claim")
    _add_scratch(claim)
    claim.add_argument("--artifact", required=True)
    claim.add_argument("--run-id", required=True)
    claim.add_argument("--record", required=True)
    claim.add_argument("--allow-existing-artifact", action="store_true")
    claim.add_argument("--build-dir")

    environment = subparsers.add_parser("check-environment")
    _add_scratch(environment)
    environment.add_argument("--marker", required=True)
    environment.add_argument("--requirements-file", required=True)
    environment.add_argument("--python", required=True)
    _add_cuda_runtime_arguments(environment, include_flashinfer=True)
    environment.add_argument("--ffmpeg-command", default="ffmpeg")

    runtime = subparsers.add_parser("check-runtime")
    _add_scratch(runtime)
    runtime.add_argument("--path", action="append", default=[])
    runtime.add_argument("--artifact", required=True)
    runtime.add_argument("--run-id", required=True)
    runtime.add_argument("--record", required=True)
    runtime.add_argument("--cuda-home")
    runtime.add_argument("--cuda-nvcc")
    runtime.add_argument("--runtime-ld-library-path")
    runtime.add_argument("--flashinfer-workspace-base")

    initialize = subparsers.add_parser("init-record")
    _add_scratch(initialize)
    initialize.add_argument("--record", required=True)
    initialize.add_argument("--run-id", required=True)
    initialize.add_argument("--build-dir", required=True)
    initialize.add_argument("--artifact", required=True)
    initialize.add_argument("--phase", action="append")
    initialize.add_argument("--config", action="append", default=[])

    begin_job = subparsers.add_parser("begin-job")
    begin_job.add_argument("--record", required=True)
    begin_job.add_argument("--phase", required=True)
    begin_job.add_argument("--dependency-job-id")
    begin_job.add_argument("--script", required=True)

    job = subparsers.add_parser("record-job")
    job.add_argument("--record", required=True)
    job.add_argument("--phase", required=True)
    job.add_argument("--job-id", required=True)
    job.add_argument("--dependency-job-id")
    job.add_argument("--script")

    submission_failed = subparsers.add_parser("mark-submission-failed")
    submission_failed.add_argument("--record", required=True)
    submission_failed.add_argument("--error", required=True)

    submitted = subparsers.add_parser("mark-submitted")
    submitted.add_argument("--record", required=True)

    validate_published = subparsers.add_parser("validate-published")
    validate_published.add_argument("--build-dir", required=True)
    validate_published.add_argument("--artifact", required=True)

    validate = subparsers.add_parser("validate-success")
    validate.add_argument("--build-dir", required=True)
    validate.add_argument("--artifact", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--record", required=True)
    status.add_argument("--sacct-command", default="sacct")

    resume = subparsers.add_parser("resume-index")
    resume.add_argument("--record", required=True)
    resume.add_argument("--sacct-command", default="sacct")
    resume.add_argument("--no-refresh", action="store_true")

    release = subparsers.add_parser("release-claim")
    _add_scratch(release)
    release.add_argument("--artifact", required=True)
    release.add_argument("--run-id", required=True)
    release.add_argument("--record", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = run_preflight(
                scratch=args.scratch,
                hf_home=args.hf_home,
                publication_parent=args.publication_parent,
                python=args.python,
                required_imports=args.required_imports,
                required_scripts=args.required_scripts,
                runtime_path=args.runtime_path,
                cuda_home=args.cuda_home,
                cuda_nvcc=args.cuda_nvcc,
                runtime_ld_library_path=args.runtime_ld_library_path,
                flashinfer_workspace_base=args.flashinfer_workspace_base,
                writable_paths=args.writable_path,
                environment_marker=args.environment_marker,
                requirements_file=args.requirements_file,
                min_free_blocks=args.min_free_blocks,
                min_free_inodes=args.min_free_inodes,
                min_free_bytes=args.min_free_bytes,
                sbatch_command=args.sbatch_command,
                sacct_command=args.sacct_command,
                ffmpeg_command=args.ffmpeg_command,
                compiler_command=args.compiler_command,
                linker_command=args.linker_command,
                readelf_command=args.readelf_command,
                quota_user=args.quota_user,
                allow_quota_unavailable=args.allow_quota_unavailable,
            )
            _json_print(result)
            return 0 if result["ok"] else 1
        if args.command == "source-bundle-sha256":
            print(compute_source_bundle_sha256(args.root))
            return 0
        if args.command == "check-paths":
            _json_print(validate_paths_under_scratch(args.scratch, args.path))
            return 0
        if args.command == "check-toolchain":
            _json_print(
                probe_native_toolchain(
                    args.scratch,
                    compiler_command=args.compiler_command,
                    linker_command=args.linker_command,
                )
            )
            return 0
        if args.command == "check-cuda-runtime":
            _json_print(
                probe_cuda_runtime(
                    args.scratch,
                    args.python,
                    args.runtime_path,
                    args.cuda_home,
                    args.cuda_nvcc,
                    args.runtime_ld_library_path,
                    readelf_command=args.readelf_command,
                )
            )
            return 0
        if args.command == "claim":
            result = claim_artifact(
                scratch=args.scratch,
                artifact_dir=args.artifact,
                run_id=args.run_id,
                record_path=args.record,
                allow_existing_artifact=args.allow_existing_artifact,
                build_dir=args.build_dir,
            )
        elif args.command == "check-environment":
            result = validate_current_environment(
                args.marker,
                args.requirements_file,
                args.python,
                scratch=args.scratch,
                runtime_path=args.runtime_path,
                cuda_home=args.cuda_home,
                cuda_nvcc=args.cuda_nvcc,
                runtime_ld_library_path=args.runtime_ld_library_path,
                flashinfer_workspace_base=args.flashinfer_workspace_base,
                ffmpeg_command=args.ffmpeg_command,
                readelf_command=args.readelf_command,
            )
        elif args.command == "check-runtime":
            result = validate_runtime_claim(
                scratch=args.scratch,
                paths=args.path,
                artifact_dir=args.artifact,
                run_id=args.run_id,
                record_path=args.record,
                cuda_home=args.cuda_home,
                cuda_nvcc=args.cuda_nvcc,
                runtime_ld_library_path=args.runtime_ld_library_path,
                flashinfer_workspace_base=args.flashinfer_workspace_base,
            )
        elif args.command == "init-record":
            result = initialize_record(
                scratch=args.scratch,
                record_path=args.record,
                run_id=args.run_id,
                build_dir=args.build_dir,
                artifact_dir=args.artifact,
                phase_order=args.phase or DAG_PHASES,
                configuration=_parse_config_values(args.config),
            )
        elif args.command == "begin-job":
            result = begin_job_submission(
                args.record,
                phase=args.phase,
                dependency_job_id=args.dependency_job_id,
                script=args.script,
            )
        elif args.command == "record-job":
            result = record_job(
                args.record,
                phase=args.phase,
                job_id=args.job_id,
                dependency_job_id=args.dependency_job_id,
                script=args.script,
            )
        elif args.command == "mark-submission-failed":
            result = mark_submission_failed(args.record, error=args.error)
        elif args.command == "mark-submitted":
            result = mark_submitted(args.record)
        elif args.command == "validate-published":
            result = validate_published_artifact(args.build_dir, args.artifact)
        elif args.command == "validate-success":
            result = validate_success(args.build_dir, args.artifact)
        elif args.command == "status":
            result = reconcile_record(
                args.record, sacct_command=args.sacct_command
            )
        elif args.command == "resume-index":
            record = (
                _load_record(pathlib.Path(args.record))
                if args.no_refresh
                else reconcile_record(args.record, sacct_command=args.sacct_command)
            )
            index = determine_resume_start_index(record)
            result = {
                "ok": True,
                "resume_index": index,
                "complete": index == len(record["phase_order"]),
                "phase": (
                    record["phase_order"][index]
                    if index < len(record["phase_order"])
                    else None
                ),
            }
        elif args.command == "release-claim":
            result = release_artifact_claim(
                scratch=args.scratch,
                artifact_dir=args.artifact,
                run_id=args.run_id,
                record_path=args.record,
            )
        else:
            raise AssertionError(args.command)
    except OpsError as exc:
        _json_print(
            {"ok": False, "error": str(exc), "type": type(exc).__name__},
            stream=sys.stderr,
        )
        return 1
    _json_print(result)
    if args.command == "status" and result.get("dag_status") in {
        "failed",
        "incomplete",
        "unresolved",
    }:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
