#!/usr/bin/env python3
"""Preflight and acceptance checks for the official OPRD Stage-2 task 0."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

from contract import contract_sha256, load_contract, validate_resolved_config


MODEL_FILES_PATH = pathlib.Path(__file__).with_name("model_files.json")


class Task0ValidationError(RuntimeError):
    """Official task 0 failed a provenance, protocol, or completion check."""


def _load(path: pathlib.Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Task0ValidationError(f"cannot read JSON file {path}: {exc}") from exc


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Task0ValidationError(message)


def _sha256_symlink(path: pathlib.Path) -> str:
    return hashlib.sha256(os.fsencode(os.readlink(path))).hexdigest()


def _verify_file_records(root: pathlib.Path, records: list[dict[str, Any]]) -> None:
    expected_paths = {record["path"] for record in records}
    for record in records:
        path = root / record["path"]
        if record.get("mode") == "symlink":
            _require(path.is_symlink(), f"authenticated symlink is missing: {path}")
            actual_digest = _sha256_symlink(path)
            actual_size = len(os.fsencode(os.readlink(path)))
        else:
            _require(path.is_file() and not path.is_symlink(), f"authenticated file is missing: {path}")
            actual_digest = _sha256_file(path)
            actual_size = path.stat().st_size
        _require(actual_size == record["size"], f"authenticated file size changed: {path}")
        _require(actual_digest == record["sha256"], f"authenticated file digest changed: {path}")
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if (path.is_file() or path.is_symlink()) and ".git" not in path.relative_to(root).parts
    }
    _require(actual_paths == expected_paths, "authenticated tree contains missing or extra files")


def _git_blob_sha1(path: pathlib.Path, size: int) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_files(models_root: pathlib.Path, contract: dict[str, Any]) -> dict[str, Any]:
    models_root = models_root.resolve(strict=True)
    pinned = _load(MODEL_FILES_PATH)
    summaries = {}
    for directory_name, model_contract in (
        ("Qwen3-1.7B-Base", contract["models"]["student"]),
        ("Qwen3-4B", contract["models"]["teacher"]),
    ):
        specification = pinned["models"][directory_name]
        _require(specification["model_id"] == model_contract["id"], "pinned model ID mismatch")
        _require(specification["revision"] == model_contract["revision"], "pinned model revision mismatch")
        root = models_root / directory_name
        expected_paths = {record["path"] for record in specification["files"]}
        actual_paths = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        _require(actual_paths == expected_paths, f"model snapshot file set mismatch: {directory_name}")
        for record in specification["files"]:
            path = root / record["path"]
            _require(path.is_file() and not path.is_symlink(), f"model file is missing: {path}")
            size = path.stat().st_size
            _require(size == record["size"], f"model file size mismatch: {path}")
            if record["oid_type"] == "sha256":
                actual_oid = _sha256_file(path)
            elif record["oid_type"] == "git_blob_sha1":
                actual_oid = _git_blob_sha1(path, size)
            else:
                raise Task0ValidationError(f"unsupported model OID type: {record['oid_type']}")
            _require(actual_oid == record["oid"], f"model content identity mismatch: {path}")
        summaries[directory_name] = {
            "model_id": specification["model_id"],
            "revision": specification["revision"],
            "files": len(specification["files"]),
        }
    return {
        "status": "models_ok",
        "model_files_contract_sha256": _sha256_file(MODEL_FILES_PATH),
        "models": summaries,
    }


def _tree_manifest(root: pathlib.Path) -> tuple[str, list[dict[str, Any]]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    digest = hashlib.sha256(
        b"".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
            + b"\n"
            for record in records
        )
    ).hexdigest()
    return digest, records


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([value for value in visible.split(",") if value.strip()])
    try:
        import torch
    except ImportError as exc:
        raise Task0ValidationError("cannot determine visible GPU count") from exc
    return torch.cuda.device_count()


def _validate_project_trust(project_root: pathlib.Path, project_revision: str) -> None:
    actual_project_revision = subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"], text=True
    ).strip()
    _require(actual_project_revision == project_revision, "project revision mismatch")
    trust_paths = [
        "cluster/official_stage2",
        "cluster/oprd_bridge.py",
        "cluster/oprd_atomic_directory.py",
        "cluster/nexus/task0_official_stage2.sbatch",
        "cluster/nexus/evaluate_official_stage2.sbatch",
    ]
    trust_status = subprocess.check_output(
        [
            "git", "-C", str(project_root), "status", "--porcelain",
            "--untracked-files=all", "--", *trust_paths
        ],
        text=True,
    )
    _require(not trust_status, "project-side official trust files are dirty or untracked")


def preflight(
    official_root: pathlib.Path,
    project_root: pathlib.Path,
    project_revision: str,
    stage1_artifact: pathlib.Path,
    resolved_path: pathlib.Path,
    result_directory: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    official_root = official_root.resolve(strict=True)
    project_root = project_root.resolve(strict=True)
    _validate_project_trust(project_root, project_revision)
    actual_project_revision = project_revision
    result_directory = result_directory.resolve(strict=False)
    _require(not result_directory.exists(), f"result directory already exists: {result_directory}")
    _require(
        result_directory.parent.resolve(strict=True)
        == (official_root / "runs").resolve(strict=True),
        "result directory must be an immediate child of the official runs directory",
    )

    source = official_root / "source" / "OPRD"
    source_manifest_path = official_root / "source" / "manifest.json"
    data_manifest_path = official_root / "source" / "dapo-manifest.json"
    models_root = official_root / "models"
    models_manifest_path = models_root / "manifest.json"
    adapter = official_root / "bridge" / "rank8_all_layers"
    adapter_manifest_path = adapter / "manifest.json"
    ready_path = official_root / "env-a63236c" / ".oprd-official-ready.json"
    for path in (
        source,
        source_manifest_path,
        data_manifest_path,
        models_manifest_path,
        adapter_manifest_path,
        ready_path,
    ):
        _require(path.exists(), f"required official input is missing: {path}")

    source_manifest = _load(source_manifest_path)
    expected_commit = contract["official_source"]["commit"]
    _require(source_manifest["commit"] == expected_commit, "official source manifest commit mismatch")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    _require(actual_commit == expected_commit, "official source checkout commit mismatch")
    source_status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    _require(not source_status, "official source checkout is dirty")
    _verify_file_records(source, source_manifest["files"])

    models_validation = validate_model_files(models_root, contract)
    models_manifest = _load(models_manifest_path)
    for directory_name, model_contract in (
        ("Qwen3-1.7B-Base", contract["models"]["student"]),
        ("Qwen3-4B", contract["models"]["teacher"]),
    ):
        model_manifest = models_manifest["models"][directory_name]
        _require(model_manifest["model_id"] == model_contract["id"], "model ID mismatch")
        _require(model_manifest["revision"] == model_contract["revision"], "model revision mismatch")
        _verify_file_records(models_root / directory_name, model_manifest["files"])

    data_manifest = _load(data_manifest_path)
    data = contract["training_data"]
    _require(data_manifest["rows"] == data["rows"], "DAPO row count mismatch")
    _require(
        data_manifest["official_git_blob"] == data["official_git_blob"],
        "DAPO Git blob mismatch",
    )
    _require(
        data_manifest["prompt_tokens"]["post_filter_rows"] == data["post_filter_rows"],
        "DAPO post-filter row count mismatch",
    )
    _require(
        data_manifest["prompt_tokens"]["overlong_rows"] == data["overlong_prompt_rows"],
        "DAPO overlong prompt count mismatch",
    )
    _require(
        data_manifest["expected_updates"] == data["expected_updates"],
        "DAPO update count mismatch",
    )

    adapter_manifest = _load(adapter_manifest_path)
    complete_parts = (adapter / "COMPLETE").read_text(encoding="ascii").strip().split()
    _require(
        complete_parts == ["manifest_sha256", _sha256_file(adapter_manifest_path)],
        "bridge adapter COMPLETE marker does not authenticate its manifest",
    )
    for filename, metadata in adapter_manifest["files"].items():
        path = adapter / filename
        _require(path.is_file() and not path.is_symlink(), f"bridge adapter file is missing: {filename}")
        _require(path.stat().st_size == metadata["size"], f"bridge adapter file size mismatch: {filename}")
        _require(_sha256_file(path) == metadata["sha256"], f"bridge adapter checksum mismatch: {filename}")
    _require(
        adapter_manifest["contract_sha256"] == contract_sha256(contract),
        "bridge adapter contract digest mismatch",
    )
    _require(adapter_manifest["parity_status"] == "ok", "bridge parity did not pass")
    _require(
        adapter_manifest["source_bridge"]["artifact_manifest_sha256"]
        == contract["stage1_bridge"]["artifact_manifest_sha256"],
        "bridge adapter source manifest mismatch",
    )
    bank_path = adapter / "ps_bank.pt"
    _require(bank_path.is_file(), "official projector bank is missing")
    _require(
        _sha256_file(bank_path)
        == adapter_manifest["files"]["ps_bank.pt"]["sha256"],
        "official projector bank checksum mismatch",
    )
    from convert_bridge_to_official import verify_converted_bridge
    adapter_validation = verify_converted_bridge(
        stage1_artifact.resolve(strict=True), adapter, contract
    )

    ready = _load(ready_path)
    _require(ready.get("status") == "ready", "official environment is not ready")
    _require(
        pathlib.Path(sys.executable).resolve(strict=True)
        == pathlib.Path(ready["python"]["executable"]).resolve(strict=True),
        "runtime Python does not match the authenticated environment",
    )
    _require(
        ready["official_source_manifest"]["sha256"]
        == _sha256_file(source_manifest_path),
        "environment source manifest mismatch",
    )
    for field in ("lock", "pip_freeze"):
        authenticated_path = pathlib.Path(ready[field]["path"]).resolve(strict=True)
        _require(
            _sha256_file(authenticated_path) == ready[field]["sha256"],
            f"authenticated environment {field} digest mismatch",
        )
    for distribution, expected_version in ready["packages"].items():
        _require(
            importlib.metadata.version(distribution) == expected_version,
            f"installed package changed: {distribution}",
        )
    current_freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    ).splitlines()
    recorded_freeze = pathlib.Path(ready["pip_freeze"]["path"]).read_text(encoding="utf-8").splitlines()
    _require(sorted(current_freeze) == recorded_freeze, "installed package freeze changed")
    resolved = _load(resolved_path)
    validate_resolved_config(resolved, contract)
    _require(_visible_gpu_count() == 8, "official task 0 requires exactly eight visible GPUs")
    _require(os.environ.get("SLURM_JOB_NUM_NODES") == "1", "official task 0 requires one node")
    return {
        "status": "preflight_ok",
        "project_revision": actual_project_revision,
        "contract_sha256": contract_sha256(contract),
        "source_tree_sha256": source_manifest["tree_sha256"],
        "data_sha256": data_manifest["sha256"],
        "data_row_order_sha256": data_manifest["row_order_sha256"],
        "models_manifest_sha256": _sha256_file(models_manifest_path),
        "model_files_contract_sha256": models_validation["model_files_contract_sha256"],
        "adapter_manifest_sha256": _sha256_file(adapter_manifest_path),
        "adapter_bank_sha256": adapter_validation["bank_sha256"],
        "adapter_audit_sha256": adapter_validation["audit_sha256"],
        "environment_marker_sha256": _sha256_file(ready_path),
        "resolved_config_sha256": _sha256_file(resolved_path),
        "expected_updates": data["expected_updates"],
    }


def accept(
    run_directory: pathlib.Path,
    project_root: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    run_directory = run_directory.resolve(strict=True)
    preflight_path = run_directory / "preflight.json"
    _require(preflight_path.is_file(), "run has no authenticated preflight report")
    preflight_report = _load(preflight_path)
    _require(preflight_report.get("status") == "preflight_ok", "preflight did not pass")
    _require(
        preflight_report["contract_sha256"] == contract_sha256(contract),
        "run contract digest mismatch",
    )
    _validate_project_trust(
        project_root.resolve(strict=True), preflight_report["project_revision"]
    )
    hydra_path = run_directory / "actual-hydra.yaml"
    argv_path = run_directory / "actual-python-argv.json"
    _require(hydra_path.is_file() and argv_path.is_file(), "actual Hydra/argv capture is missing")
    argv = _load(argv_path)
    _require(isinstance(argv, list) and all(isinstance(value, str) for value in argv), "captured Python argv must be a string array")
    _require(argv[:2] == ["-m", "verl.trainer.main_ppo"], "captured Python argv is not the official verl trainer")
    _require("--cfg" not in argv, "captured training argv unexpectedly requests config-only mode")
    try:
        import yaml
    except ImportError as exc:
        raise Task0ValidationError("PyYAML is required for task-0 acceptance") from exc
    from contract import validate_hydra_config
    validate_hydra_config(
        yaml.safe_load(hydra_path.read_text(encoding="utf-8")),
        official_root=run_directory.parent.parent,
    )

    expected_step = contract["training_data"]["expected_updates"]
    checkpoints = sorted(run_directory.rglob(f"global_step_{expected_step}"))
    _require(len(checkpoints) == 1, f"expected one global_step_{expected_step} checkpoint")
    later = [
        path
        for path in run_directory.rglob("global_step_*")
        if path.is_dir()
        and path.name.removeprefix("global_step_").isdigit()
        and int(path.name.removeprefix("global_step_")) > expected_step
    ]
    _require(not later, "run contains optimizer steps beyond the one-epoch contract")
    log_paths = sorted((run_directory / "logs" / "terminal").glob("*.log"))
    _require(len(log_paths) == 1, "run must contain exactly one official terminal log")
    log_text = log_paths[0].read_text(encoding="utf-8", errors="replace")
    for failure in ("Traceback", "RayActorError", "CUDA out of memory", "NCCL error"):
        _require(failure not in log_text, f"terminal log contains failure marker: {failure}")
    checkpoint_tree_sha256, checkpoint_files = _tree_manifest(checkpoints[0])
    merged_checkpoint = run_directory / "merged-hf"
    _require(merged_checkpoint.is_dir() and not merged_checkpoint.is_symlink(), "merged Hugging Face checkpoint is missing")
    merged_tree_sha256, merged_files = _tree_manifest(merged_checkpoint)
    _require(merged_files, "merged Hugging Face checkpoint is empty")
    return {
        "status": "ACCEPTED_OFFICIAL_TASK0",
        "project_revision": preflight_report["project_revision"],
        "contract_sha256": contract_sha256(contract),
        "global_step": expected_step,
        "checkpoint": str(checkpoints[0]),
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "checkpoint_files": checkpoint_files,
        "merged_checkpoint": str(merged_checkpoint),
        "merged_checkpoint_tree_sha256": merged_tree_sha256,
        "merged_checkpoint_files": merged_files,
        "actual_hydra_sha256": _sha256_file(hydra_path),
        "actual_argv_sha256": _sha256_file(argv_path),
        "terminal_log": str(log_paths[0]),
        "terminal_log_sha256": _sha256_file(log_paths[0]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=pathlib.Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--official-root", type=pathlib.Path, required=True)
    preflight_parser.add_argument("--project-root", type=pathlib.Path, required=True)
    preflight_parser.add_argument("--project-revision", required=True)
    preflight_parser.add_argument("--stage1-artifact", type=pathlib.Path, required=True)
    preflight_parser.add_argument("--resolved", type=pathlib.Path, required=True)
    preflight_parser.add_argument("--result-directory", type=pathlib.Path, required=True)
    accept_parser = subparsers.add_parser("accept")
    accept_parser.add_argument("--run-directory", type=pathlib.Path, required=True)
    accept_parser.add_argument("--project-root", type=pathlib.Path, required=True)
    models_parser = subparsers.add_parser("models")
    models_parser.add_argument("--models-root", type=pathlib.Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    contract = load_contract(arguments.contract) if arguments.contract else load_contract()
    if arguments.command == "preflight":
        result = preflight(
            arguments.official_root,
            arguments.project_root,
            arguments.project_revision,
            arguments.stage1_artifact,
            arguments.resolved,
            arguments.result_directory,
            contract,
        )
    elif arguments.command == "accept":
        result = accept(arguments.run_directory, arguments.project_root, contract)
    else:
        result = validate_model_files(arguments.models_root, contract)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
