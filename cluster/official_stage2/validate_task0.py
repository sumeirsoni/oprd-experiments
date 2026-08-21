#!/usr/bin/env python3
"""Preflight and acceptance checks for the official OPRD Stage-2 task 0."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Any

from contract import (
    contract_sha256,
    effective_execution_topology,
    effective_resume_authorization,
    execution_manifest_sha256,
    load_contract,
    load_execution_manifest,
    load_resume_manifest,
    resume_manifest_sha256,
    validate_resolved_config,
)


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
    _require(root.is_dir() and not root.is_symlink(), f"authenticated tree root is invalid: {root}")
    records = []
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), f"authenticated tree contains a symlink: {path}")
        if not path.is_file():
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


def validate_fsdp_shards(actor: pathlib.Path, world_size: int) -> list[dict[str, Any]]:
    _require(actor.is_dir() and not actor.is_symlink(), "actor checkpoint directory is missing")
    expected = {
        f"{kind}_world_size_{world_size}_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state")
        for rank in range(world_size)
    }
    actual = {
        path.name
        for path in actor.glob("*_world_size_*_rank_*.pt")
        if path.is_file() or path.is_symlink()
    }
    _require(actual == expected, "FSDP checkpoint shard topology mismatch")
    records = []
    for name in sorted(expected):
        path = actor / name
        _require(path.is_file() and not path.is_symlink(), f"invalid FSDP shard: {path}")
        _require(path.stat().st_size > 0, f"empty FSDP shard: {path}")
        records.append({
            "path": name,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        })
    _require(
        (actor / "huggingface" / "config.json").is_file(),
        "checkpoint Hugging Face config is missing",
    )
    return records


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([value for value in visible.split(",") if value.strip()])
    try:
        import torch
    except ImportError as exc:
        raise Task0ValidationError("cannot determine visible GPU count") from exc
    return torch.cuda.device_count()


def _validate_project_trust(
    project_root: pathlib.Path, project_revision: str, *, include_resume: bool = False
) -> None:
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
    if include_resume:
        trust_paths.append("cluster/nexus/task0_resume_7258005.sbatch")
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
    resolved_execution_path: pathlib.Path,
    execution_path: pathlib.Path,
    result_directory: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    official_root = official_root.resolve(strict=True)
    project_root = project_root.resolve(strict=True)
    _validate_project_trust(project_root, project_revision)
    execution_manifest = load_execution_manifest(contract, execution_path)
    execution_digest = execution_manifest_sha256(execution_manifest, contract)
    effective_topology = effective_execution_topology(execution_manifest, contract)
    _require(
        _load(resolved_execution_path) == effective_topology,
        "resolved execution topology mismatch",
    )
    world_size = execution_manifest["approved_topology"]["world_size"]
    actual_project_revision = project_revision
    result_directory = result_directory.resolve(strict=False)
    _require(
        result_directory.name == execution_manifest["scope"]["authorized_result_namespace"],
        "result namespace is not authorized by the execution manifest",
    )
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
    _require(
        _visible_gpu_count() == world_size,
        f"approved task 0 requires exactly {world_size} visible GPUs",
    )
    _require(os.environ.get("SLURM_JOB_NUM_NODES") == "1", "official task 0 requires one node")
    return {
        "status": "preflight_ok",
        "project_revision": actual_project_revision,
        "contract_sha256": contract_sha256(contract),
        "execution_manifest_sha256": execution_digest,
        "historical_world_size": execution_manifest["historical_topology"]["world_size"],
        "effective_world_size": world_size,
        "canary_evidence": execution_manifest["canary_evidence"],
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
        "resolved_execution_sha256": _sha256_file(resolved_execution_path),
        "expected_updates": data["expected_updates"],
    }


def accept(
    run_directory: pathlib.Path,
    project_root: pathlib.Path,
    execution_path: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    run_directory = run_directory.resolve(strict=True)
    preflight_path = run_directory / "preflight.json"
    _require(preflight_path.is_file(), "run has no authenticated preflight report")
    preflight_report = _load(preflight_path)
    _require(preflight_report.get("status") == "preflight_ok", "preflight did not pass")
    execution_manifest = load_execution_manifest(contract, execution_path)
    execution_digest = execution_manifest_sha256(execution_manifest, contract)
    world_size = execution_manifest["approved_topology"]["world_size"]
    _require(
        run_directory.name == execution_manifest["scope"]["authorized_result_namespace"],
        "run namespace is not authorized by the execution manifest",
    )
    _require(
        preflight_report["contract_sha256"] == contract_sha256(contract),
        "run contract digest mismatch",
    )
    _require(
        preflight_report["execution_manifest_sha256"] == execution_digest,
        "run execution manifest digest mismatch",
    )
    _require(
        preflight_report["effective_world_size"] == world_size,
        "run effective world size mismatch",
    )
    _validate_project_trust(
        project_root.resolve(strict=True), preflight_report["project_revision"]
    )
    resolved_execution_path = run_directory / "resolved-execution-topology.json"
    _require(resolved_execution_path.is_file(), "resolved execution topology is missing")
    _require(
        _sha256_file(resolved_execution_path)
        == preflight_report["resolved_execution_sha256"],
        "resolved execution topology digest mismatch",
    )
    _require(
        _load(resolved_execution_path)
        == effective_execution_topology(execution_manifest, contract),
        "resolved execution topology content mismatch",
    )
    hydra_path = run_directory / "actual-hydra.yaml"
    argv_path = run_directory / "actual-python-argv.json"
    _require(hydra_path.is_file() and argv_path.is_file(), "actual Hydra/argv capture is missing")
    argv = _load(argv_path)
    _require(isinstance(argv, list) and all(isinstance(value, str) for value in argv), "captured Python argv must be a string array")
    _require(argv[:2] == ["-m", "verl.trainer.main_ppo"], "captured Python argv is not the official verl trainer")
    _require("--cfg" not in argv, "captured training argv unexpectedly requests config-only mode")
    topology_arguments = [
        value for value in argv if value.startswith("trainer.n_gpus_per_node=")
    ]
    _require(
        topology_arguments == [f"trainer.n_gpus_per_node={world_size}"],
        "captured training argv has an invalid topology override",
    )
    retention_arguments = [
        value for value in argv if value.startswith("trainer.max_actor_ckpt_to_keep=")
    ]
    _require(
        retention_arguments
        == [
            "trainer.max_actor_ckpt_to_keep="
            f"{execution_manifest['checkpoint_policy']['max_actor_checkpoints_to_keep']}"
        ],
        "captured training argv has an invalid checkpoint retention override",
    )
    hydra_run_arguments = [
        value for value in argv if value.startswith("hydra.run.dir=")
    ]
    _require(
        hydra_run_arguments == [f"hydra.run.dir={run_directory}/hydra-run"],
        "captured training argv has an invalid Hydra run directory",
    )
    try:
        import yaml
    except ImportError as exc:
        raise Task0ValidationError("PyYAML is required for task-0 acceptance") from exc
    from contract import validate_hydra_config
    validate_hydra_config(
        yaml.safe_load(hydra_path.read_text(encoding="utf-8")),
        official_root=run_directory.parent.parent,
        execution_manifest=execution_manifest,
        contract=contract,
    )

    expected_step = contract["training_data"]["expected_updates"]
    checkpoints = sorted(run_directory.rglob(f"global_step_{expected_step}"))
    _require(len(checkpoints) == 1, f"expected one global_step_{expected_step} checkpoint")
    _require(
        checkpoints[0].is_dir() and not checkpoints[0].is_symlink(),
        "final checkpoint must be a real directory",
    )
    checkpoint_resolved = checkpoints[0].resolve(strict=True)
    _require(
        checkpoint_resolved.is_relative_to(run_directory),
        "final checkpoint resolves outside the run directory",
    )
    later = [
        path
        for path in run_directory.rglob("global_step_*")
        if path.is_dir()
        and path.name.removeprefix("global_step_").isdigit()
        and int(path.name.removeprefix("global_step_")) > expected_step
    ]
    _require(not later, "run contains optimizer steps beyond the one-epoch contract")
    fsdp_shards = validate_fsdp_shards(checkpoints[0] / "actor", world_size)
    log_paths = sorted((run_directory / "logs" / "terminal").glob("*.log"))
    _require(len(log_paths) == 1, "run must contain exactly one official terminal log")
    log_text = log_paths[0].read_text(encoding="utf-8", errors="replace")
    for failure in ("Traceback", "RayActorError", "CUDA out of memory", "NCCL error"):
        _require(failure not in log_text, f"terminal log contains failure marker: {failure}")
    runtime_path = run_directory / "official-runtime.log"
    _require(runtime_path.is_file(), "official runtime log is missing")
    runtime_text = runtime_path.read_text(encoding="utf-8", errors="replace")
    _require(
        re.search(rf"training/global_step:{expected_step}(?![0-9])", runtime_text)
        is not None,
        "runtime log is missing the exact final global step",
    )
    for failure in ("RayActorError", "ActorDiedError", "CUDA out of memory", "NCCL error"):
        _require(failure not in runtime_text, f"runtime log contains failure marker: {failure}")
    cleanup_warning_count = sum(
        "RuntimeError: DataLoader worker (" in line
        and "is killed by signal: Killed" in line
        for line in runtime_text.splitlines()
    )
    _require(
        runtime_text.count("Traceback (most recent call last):") == cleanup_warning_count,
        "runtime contains an unexplained traceback",
    )
    checkpoint_tree_sha256, checkpoint_files = _tree_manifest(checkpoints[0])
    merged_checkpoint = run_directory / "merged-hf"
    _require(merged_checkpoint.is_dir() and not merged_checkpoint.is_symlink(), "merged Hugging Face checkpoint is missing")
    merged_tree_sha256, merged_files = _tree_manifest(merged_checkpoint)
    _require(merged_files, "merged Hugging Face checkpoint is empty")
    return {
        "status": "ACCEPTED_OFFICIAL_TASK0",
        "project_revision": preflight_report["project_revision"],
        "contract_sha256": contract_sha256(contract),
        "execution_manifest_sha256": execution_digest,
        "historical_world_size": execution_manifest["historical_topology"]["world_size"],
        "effective_world_size": world_size,
        "topology_deviation_from_historical_official": True,
        "global_step": expected_step,
        "checkpoint": str(checkpoints[0]),
        "checkpoint_tree_sha256": checkpoint_tree_sha256,
        "checkpoint_files": checkpoint_files,
        "fsdp_shards": fsdp_shards,
        "merged_checkpoint": str(merged_checkpoint),
        "merged_checkpoint_tree_sha256": merged_tree_sha256,
        "merged_checkpoint_files": merged_files,
        "actual_hydra_sha256": _sha256_file(hydra_path),
        "actual_argv_sha256": _sha256_file(argv_path),
        "terminal_log": str(log_paths[0]),
        "terminal_log_sha256": _sha256_file(log_paths[0]),
        "runtime_log": str(runtime_path),
        "runtime_log_sha256": _sha256_file(runtime_path),
        "cleanup_warning_count": cleanup_warning_count,
    }


def _require_real_descendant(path: pathlib.Path, root: pathlib.Path, description: str) -> pathlib.Path:
    root = root.resolve(strict=True)
    _require(root.is_dir() and not root.is_symlink(), f"{description} root is invalid")
    _require(path.is_absolute(), f"{description} path must be absolute")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise Task0ValidationError(
            f"{description} path is outside its authorized root"
        ) from exc
    current = root
    for part in relative.parts:
        current = current / part
        _require(not current.is_symlink(), f"{description} path contains a symlink: {current}")
    resolved = path.resolve(strict=True)
    _require(resolved.is_relative_to(root), f"{description} resolves outside its authorized root")
    return resolved


def _verify_record(path: pathlib.Path, record: dict[str, Any], description: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"{description} is missing or not a regular file: {path}")
    _require(path.stat().st_size == record["size"], f"{description} size changed: {path}")
    digest = _sha256_file(path)
    _require(digest == record["sha256"], f"{description} digest changed: {path}")
    return {"path": str(path), "size": record["size"], "sha256": digest}


def _load_dataloader_state(path: pathlib.Path) -> dict[str, Any]:
    try:
        import torch
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Task0ValidationError(f"cannot authenticate checkpoint dataloader state: {exc}") from exc
    _require(isinstance(state, dict), "checkpoint data.pt must contain a state dictionary")
    expected_keys = {
        "snapshot_step": "_snapshot_step",
        "samples_yielded": "samples_yielded",
        "sampler_iter_yielded": "_sampler_iter_yielded",
        "num_workers": "_num_workers",
        "iterator_finished": "_iterator_finished",
    }
    _require(all(key in state for key in expected_keys.values()), "checkpoint data.pt is missing required dataloader fields")
    return {name: state[key] for name, key in expected_keys.items()}


def _resume_context(
    run_directory: pathlib.Path,
    project_root: pathlib.Path,
    project_revision: str,
    execution_path: pathlib.Path,
    resume_path: pathlib.Path,
    contract: dict[str, Any],
) -> tuple[pathlib.Path, pathlib.Path, dict[str, Any], dict[str, Any], str]:
    run_directory = run_directory.resolve(strict=True)
    _require(run_directory.is_dir() and not run_directory.is_symlink(), "original run directory must be real")
    execution = load_execution_manifest(contract, execution_path)
    resume = load_resume_manifest(contract, execution, resume_path)
    _require(run_directory.name == resume["scope"]["authorized_result_namespace"], "resume namespace is not authorized")
    _validate_project_trust(
        project_root.resolve(strict=True), project_revision, include_resume=True
    )
    continuation = run_directory / resume["scope"]["continuation_subdirectory"]
    _require(continuation.parent == run_directory / "continuations", "continuation path is not canonical")
    return run_directory, continuation, execution, resume, resume_manifest_sha256(resume, contract, execution)


def _authenticate_original(run_directory: pathlib.Path, resume: dict[str, Any]) -> dict[str, Any]:
    _require(not (run_directory / "COMPLETE").exists(), "original run is already complete")
    _require(not (run_directory / "acceptance.json").exists(), "original run is already accepted")
    _require(not (run_directory / "merged-hf").exists(), "original run has a root merged checkpoint")
    evidence = []
    for record in resume["source_evidence"]:
        raw = pathlib.Path(record["path"])
        path = raw if raw.is_absolute() else run_directory / raw
        if not raw.is_absolute():
            _require_real_descendant(path, run_directory, "original evidence")
        evidence.append(_verify_record(path, record, "original evidence"))
    source = _require_real_descendant(
        run_directory / resume["source_checkpoint"]["relative_path"], run_directory, "source checkpoint"
    )
    _require(source.is_dir() and not source.is_symlink(), "source checkpoint must be a real directory")
    inventory = []
    for record in resume["source_checkpoint"]["inventory"]:
        path = source / record["path"]
        _require_real_descendant(path, source, "source checkpoint file")
        inventory.append(_verify_record(path, record, "source checkpoint file"))
    actual_inventory = {
        path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file() or path.is_symlink()
    }
    expected_inventory = {record["path"] for record in resume["source_checkpoint"]["inventory"]}
    _require(actual_inventory == expected_inventory, "source checkpoint contains missing, extra, or symlinked files")
    validate_fsdp_shards(source / "actor", resume["source_checkpoint"]["world_size"])
    tracker_record = resume["source_checkpoint"]["tracker"]
    tracker = run_directory / tracker_record["relative_path"]
    _require_real_descendant(tracker, run_directory, "source checkpoint tracker")
    tracker_result = _verify_record(tracker, tracker_record, "source checkpoint tracker")
    _require(tracker.read_bytes() == str(tracker_record["value"]).encode("ascii"), "source checkpoint tracker value mismatch")
    dataloader = _load_dataloader_state(source / "data.pt")
    _require(dataloader == resume["source_checkpoint"]["dataloader_state"], "source checkpoint dataloader state mismatch")
    later = [
        path for path in run_directory.rglob("global_step_*")
        if path.is_dir() and path.name.removeprefix("global_step_").isdigit()
        and int(path.name.removeprefix("global_step_")) > resume["source_checkpoint"]["global_step"]
        and not path.is_relative_to(run_directory / "continuations")
    ]
    _require(not later, "original run contains a complete checkpoint later than the authorized source")
    return {"source_checkpoint": str(source), "source_evidence": evidence, "source_inventory": inventory, "tracker": tracker_result, "dataloader_state": dataloader}


def _authenticate_current_resume_inputs(
    run_directory: pathlib.Path,
    project_root: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    official_root = run_directory.parent.parent.resolve(strict=True)
    source = official_root / "source" / "OPRD"
    source_manifest_path = official_root / "source" / "manifest.json"
    models_root = official_root / "models"
    models_manifest_path = models_root / "manifest.json"
    adapter = official_root / "bridge" / "rank8_all_layers"
    adapter_manifest_path = adapter / "manifest.json"
    ready_path = official_root / "env-a63236c" / ".oprd-official-ready.json"
    for path in (
        source,
        source_manifest_path,
        models_manifest_path,
        adapter_manifest_path,
        ready_path,
    ):
        _require(path.exists() and not path.is_symlink(), f"current official input is missing or symlinked: {path}")

    source_manifest = _load(source_manifest_path)
    expected_commit = contract["official_source"]["commit"]
    actual_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    _require(source_manifest["commit"] == expected_commit == actual_commit, "current official source commit mismatch")
    source_status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    _require(not source_status, "current official source checkout is dirty")
    _verify_file_records(source, source_manifest["files"])

    models_validation = validate_model_files(models_root, contract)
    models_manifest = _load(models_manifest_path)
    for directory_name, model_contract in (
        ("Qwen3-1.7B-Base", contract["models"]["student"]),
        ("Qwen3-4B", contract["models"]["teacher"]),
    ):
        model_manifest = models_manifest["models"][directory_name]
        _require(model_manifest["model_id"] == model_contract["id"], "current model ID mismatch")
        _require(model_manifest["revision"] == model_contract["revision"], "current model revision mismatch")
        _verify_file_records(models_root / directory_name, model_manifest["files"])

    adapter_manifest = _load(adapter_manifest_path)
    complete_parts = (adapter / "COMPLETE").read_text(encoding="ascii").strip().split()
    _require(
        complete_parts == ["manifest_sha256", _sha256_file(adapter_manifest_path)],
        "current bridge COMPLETE marker mismatch",
    )
    for filename, metadata in adapter_manifest["files"].items():
        _verify_record(adapter / filename, {"path": filename, **metadata}, "current bridge file")
    _require(adapter_manifest["contract_sha256"] == contract_sha256(contract), "current bridge contract mismatch")
    _require(adapter_manifest["parity_status"] == "ok", "current bridge parity did not pass")

    ready = _load(ready_path)
    _require(ready.get("status") == "ready", "current official environment is not ready")
    _require(
        pathlib.Path(sys.executable).resolve(strict=True)
        == pathlib.Path(ready["python"]["executable"]).resolve(strict=True),
        "current runtime Python does not match the authenticated environment",
    )
    for field in ("lock", "pip_freeze"):
        authenticated_path = pathlib.Path(ready[field]["path"]).resolve(strict=True)
        _require(_sha256_file(authenticated_path) == ready[field]["sha256"], f"current environment {field} digest mismatch")
    for distribution, expected_version in ready["packages"].items():
        _require(importlib.metadata.version(distribution) == expected_version, f"current installed package changed: {distribution}")
    current_freeze = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True
    ).splitlines()
    recorded_freeze = pathlib.Path(ready["pip_freeze"]["path"]).read_text(encoding="utf-8").splitlines()
    _require(sorted(current_freeze) == recorded_freeze, "current installed package freeze changed")
    return {
        "source_tree_sha256": source_manifest["tree_sha256"],
        "models_manifest_sha256": _sha256_file(models_manifest_path),
        "model_files_contract_sha256": models_validation["model_files_contract_sha256"],
        "adapter_manifest_sha256": _sha256_file(adapter_manifest_path),
        "environment_marker_sha256": _sha256_file(ready_path),
    }


def resume_preflight(
    run_directory: pathlib.Path,
    project_root: pathlib.Path,
    project_revision: str,
    execution_path: pathlib.Path,
    resume_path: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    run, continuation, execution, resume, resume_digest = _resume_context(
        run_directory, project_root, project_revision, execution_path, resume_path, contract
    )
    _require(not continuation.exists(), f"continuation path already exists: {continuation}")
    _require(not continuation.parent.is_symlink(), "continuations directory must not be a symlink")
    original = _authenticate_original(run, resume)
    current_inputs = _authenticate_current_resume_inputs(run, project_root, contract)
    _require(_visible_gpu_count() == resume["approved_topology"]["world_size"], "resume requires exactly four visible GPUs")
    _require(os.environ.get("SLURM_JOB_NUM_NODES") == "1", "resume requires one node")
    return {
        "status": "resume_preflight_ok", "project_revision": project_revision,
        "historical_project_revision": resume["scope"]["original_project_revision"],
        "contract_sha256": contract_sha256(contract),
        "execution_manifest_sha256": execution_manifest_sha256(execution, contract),
        "resume_manifest_sha256": resume_digest, "continuation": str(continuation),
        "current_inputs": current_inputs, **original, **resume["resume_arithmetic"],
    }


def _required_resume_arguments(resume: dict[str, Any], source: pathlib.Path, continuation: pathlib.Path) -> dict[str, str]:
    return {
        "trainer.resume_mode": "resume_path", "trainer.resume_from_path": str(source),
        "trainer.default_local_dir": str(continuation / "checkpoints"),
        "trainer.output_log_path": str(continuation / "logs" / "terminal" / "resume.log"),
        "trainer.total_training_steps": str(resume["resume_arithmetic"]["target_step"]),
        "trainer.n_gpus_per_node": str(resume["approved_topology"]["world_size"]),
        "trainer.max_actor_ckpt_to_keep": "1", "hydra.run.dir": str(continuation / "hydra-run"),
    }


def _authenticate_dataloader_restoration_evidence(
    continuation: pathlib.Path,
    source: pathlib.Path,
    resume: dict[str, Any],
) -> dict[str, Any]:
    evidence_path = continuation / "dataloader-restoration.json"
    _require(
        evidence_path.is_file() and not evidence_path.is_symlink(),
        "continuation dataloader restoration evidence is missing",
    )
    evidence = _load(evidence_path)
    data_record = next(
        record
        for record in resume["source_checkpoint"]["inventory"]
        if record["path"] == "data.pt"
    )
    expected = {
        "status": "dataloader_state_restored",
        "data_path": str(source / "data.pt"),
        "data_sha256": data_record["sha256"],
        "state": resume["source_checkpoint"]["dataloader_state"],
    }
    _require(
        evidence == expected,
        "continuation dataloader restoration evidence does not match the authorized data.pt state",
    )
    return {
        **evidence,
        "evidence_path": str(evidence_path),
        "evidence_sha256": _sha256_file(evidence_path),
    }


def _require_positive_dataloader_restore(text: str) -> None:
    negative = re.compile(
        r"(?:fail(?:ed|ure)?|could not|cannot|unable|missing|not found|starting over|start(?:ing)? from scratch|fresh iterator|new iterator|reset)"
        r"[^\n]*(?:data(?:loader)?|sampler)|"
        r"(?:data(?:loader)?|sampler)[^\n]*(?:fail(?:ed|ure)?|could not|cannot|unable|missing|not found|starting over|start(?:ing)? from scratch|fresh iterator|new iterator|reset)",
        re.IGNORECASE,
    )
    _require(
        negative.search(text) is None,
        "runtime contains negative dataloader restoration evidence",
    )


def _require_only_known_tracebacks(text: str, description: str) -> int:
    starts = [match.start() for match in re.finditer(r"^Traceback \(most recent call last\):", text, re.MULTILINE)]
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        block = text[start:end]
        exception_lines = [
            line.strip()
            for line in block.splitlines()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):", line.strip())
        ]
        _require(
            len(exception_lines) == 1
            and re.fullmatch(
                r"RuntimeError: DataLoader worker \([^\n]*\) is killed by signal: Killed",
                exception_lines[0],
            ) is not None,
            f"{description} contains an unexplained traceback",
        )
    return len(starts)


def resume_accept(
    run_directory: pathlib.Path, project_root: pathlib.Path, project_revision: str,
    execution_path: pathlib.Path, resume_path: pathlib.Path, contract: dict[str, Any],
) -> dict[str, Any]:
    run, continuation, execution, resume, resume_digest = _resume_context(
        run_directory, project_root, project_revision, execution_path, resume_path, contract
    )
    continuation = _require_real_descendant(continuation, run, "continuation")
    _require(continuation.is_dir() and not continuation.is_symlink(), "continuation directory is invalid")
    original = _authenticate_original(run, resume)
    current_inputs = _authenticate_current_resume_inputs(run, project_root, contract)
    preflight_path = continuation / "resume-preflight.json"
    _require(preflight_path.is_file() and not preflight_path.is_symlink(), "resume preflight report is missing")
    preflight_report = _load(preflight_path)
    expected_preflight = {
        "status": "resume_preflight_ok",
        "project_revision": project_revision,
        "historical_project_revision": resume["scope"]["original_project_revision"],
        "contract_sha256": contract_sha256(contract),
        "execution_manifest_sha256": execution_manifest_sha256(execution, contract),
        "resume_manifest_sha256": resume_digest,
        "continuation": str(continuation),
        "current_inputs": current_inputs,
        **original,
        **resume["resume_arithmetic"],
    }
    _require(
        preflight_report == expected_preflight,
        "resume preflight report does not match fully reauthenticated evidence",
    )
    resolved_resume_path = continuation / "resolved-resume-authorization.json"
    _require(
        resolved_resume_path.is_file() and not resolved_resume_path.is_symlink(),
        "resolved resume authorization is missing or not a regular file",
    )
    resolved_resume = _load(resolved_resume_path)
    canonical_resolved_resume = effective_resume_authorization(
        resume, contract, execution
    )
    _require(
        resolved_resume == canonical_resolved_resume,
        "resolved resume authorization does not match the canonical authorization",
    )
    canonical_resolved_bytes = (
        json.dumps(
            canonical_resolved_resume,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _require(
        resolved_resume_path.read_bytes() == canonical_resolved_bytes,
        "resolved resume authorization is not the canonical byte representation",
    )
    argv_path, hydra_path = continuation / "actual-python-argv.json", continuation / "actual-hydra.yaml"
    argv = _load(argv_path)
    _require(isinstance(argv, list) and argv[:2] == ["-m", "verl.trainer.main_ppo"], "captured resume argv is invalid")
    _require("--cfg" not in argv, "captured resume argv unexpectedly requests config-only mode")
    source = pathlib.Path(original["source_checkpoint"])
    for key, value in _required_resume_arguments(resume, source, continuation).items():
        hits = [argument for argument in argv if argument.startswith(f"{key}=")]
        _require(hits == [f"{key}={value}"], f"captured resume argv has an invalid or duplicate {key} override")
    _require(hydra_path.is_file() and not hydra_path.is_symlink(), "captured resume Hydra config is missing")
    try:
        import yaml
    except ImportError as exc:
        raise Task0ValidationError(
            "PyYAML is required for task-0 resume acceptance"
        ) from exc
    from contract import validate_hydra_config
    validate_hydra_config(
        yaml.safe_load(hydra_path.read_text(encoding="utf-8")),
        official_root=run.parent.parent,
        execution_manifest=execution,
        resume_manifest=resume,
        official_namespace=run,
        continuation_dir=continuation,
        contract=contract,
    )
    target = resume["resume_arithmetic"]["target_step"]
    checkpoint = continuation / "checkpoints" / f"global_step_{target}"
    _require_real_descendant(checkpoint, continuation, "final continuation checkpoint")
    _require(checkpoint.is_dir() and not checkpoint.is_symlink(), "exact final continuation checkpoint is missing")
    later = [p for p in (continuation / "checkpoints").glob("global_step_*") if p.is_dir() and p.name[12:].isdigit() and int(p.name[12:]) > target]
    _require(not later, "continuation contains optimizer steps beyond the target")
    fsdp = validate_fsdp_shards(checkpoint / "actor", resume["approved_topology"]["world_size"])
    logs = sorted((continuation / "logs" / "terminal").glob("*.log"))
    _require(len(logs) == 1 and not logs[0].is_symlink(), "continuation must contain exactly one real terminal log")
    runtime = continuation / "official-runtime.log"
    _require(runtime.is_file() and not runtime.is_symlink(), "continuation runtime log is missing")
    text = runtime.read_text(encoding="utf-8", errors="replace")
    terminal_text = logs[0].read_text(encoding="utf-8", errors="replace")
    for failure in ("RayActorError", "ActorDiedError", "CUDA out of memory", "NCCL error", "Training from scratch", "will start from scratch"):
        _require(failure.lower() not in text.lower() and failure.lower() not in terminal_text.lower(), f"continuation log contains failure marker: {failure}")
    _require_only_known_tracebacks(text, "continuation runtime")
    _require_only_known_tracebacks(terminal_text, "continuation terminal log")
    start = resume["source_checkpoint"]["global_step"]
    first = resume["resume_arithmetic"]["first_resumed_step"]
    for pattern, message in (
        (rf"Load from checkpoint folder:\s*{re.escape(str(source))}(?:\s|$)", "source checkpoint load proof"),
        (rf"Setting global step to {start}(?![0-9])", "global-step restore proof"),
        (rf"Resuming from\s+{re.escape(str(source))}(?:\s|$)", "resume-path proof"),
        (rf"training/global_step:{first}(?![0-9])", "first resumed step proof"),
        (rf"training/global_step:{target}(?![0-9])", "exact final step proof"),
    ):
        _require(re.search(pattern, text) is not None, f"runtime is missing {message}")
    _require_positive_dataloader_restore(text)
    dataloader_restoration = _authenticate_dataloader_restoration_evidence(
        continuation, source, resume
    )
    logged_steps = {
        int(value)
        for value in re.findall(r"training/global_step:([0-9]+)", text)
    }
    expected_steps = set(range(first, target + 1))
    _require(
        expected_steps.issubset(logged_steps),
        "runtime does not prove every resumed step from 1601 through 2239",
    )
    _require(
        not any(step > target for step in logged_steps),
        "runtime contains optimizer steps beyond the target",
    )
    for rank in range(resume["approved_topology"]["world_size"]):
        for kind, label in (("model", "model"), ("optim", "optimizer"), ("extra_state", "rng"), ("extra_state", "lr_scheduler")):
            filename = f"{kind}_world_size_4_rank_{rank}.pt"
            _require(re.search(rf"Loaded {label} from .*?/actor/{re.escape(filename)}", text) is not None, f"runtime is missing {label} restore proof for rank {rank}")
    checkpoint_digest, checkpoint_files = _tree_manifest(checkpoint)
    merged = continuation / "merged-hf"
    _require_real_descendant(merged, continuation, "merged continuation checkpoint")
    _require(merged.is_dir() and not merged.is_symlink(), "merged continuation checkpoint is missing")
    merged_digest, merged_files = _tree_manifest(merged)
    _require(merged_files, "merged continuation checkpoint is empty")
    return {
        "status": "ACCEPTED_TASK0_CONTINUATION", "project_revision": project_revision,
        "historical_project_revision": resume["scope"]["original_project_revision"],
        "contract_sha256": contract_sha256(contract),
        "execution_manifest_sha256": execution_manifest_sha256(execution, contract),
        "resume_manifest_sha256": resume_digest,
        "resolved_resume_authorization": str(resolved_resume_path),
        "resolved_resume_authorization_sha256": _sha256_file(resolved_resume_path),
        "current_inputs_reauthenticated": current_inputs,
        "original_evidence_reauthenticated": original,
        "dataloader_restoration": dataloader_restoration,
        "global_step": target, "checkpoint": str(checkpoint), "checkpoint_tree_sha256": checkpoint_digest,
        "checkpoint_files": checkpoint_files, "fsdp_shards": fsdp, "merged_checkpoint": str(merged),
        "merged_checkpoint_tree_sha256": merged_digest, "merged_checkpoint_files": merged_files,
        "actual_hydra_sha256": _sha256_file(hydra_path), "actual_argv_sha256": _sha256_file(argv_path),
        "terminal_log": str(logs[0]), "terminal_log_sha256": _sha256_file(logs[0]),
        "runtime_log": str(runtime), "runtime_log_sha256": _sha256_file(runtime),
    }


def resume_composite_accept(
    run_directory: pathlib.Path, project_root: pathlib.Path, project_revision: str,
    execution_path: pathlib.Path, resume_path: pathlib.Path, contract: dict[str, Any],
) -> dict[str, Any]:
    run, continuation, _, _, _ = _resume_context(
        run_directory, project_root, project_revision, execution_path, resume_path, contract
    )
    acceptance_path = continuation / "acceptance.json"
    _require(acceptance_path.is_file() and not acceptance_path.is_symlink(), "continuation acceptance is missing")
    recorded = _load(acceptance_path)
    expected = resume_accept(run, project_root, project_revision, execution_path, resume_path, contract)
    _require(recorded == expected, "continuation acceptance does not match revalidated evidence")
    return {
        **expected, "status": "ACCEPTED_OFFICIAL_TASK0",
        "continuation_acceptance": str(acceptance_path),
        "continuation_acceptance_sha256": _sha256_file(acceptance_path),
    }


def validate_resume_reservation(
    run_directory: pathlib.Path,
    project_root: pathlib.Path,
    project_revision: str,
    execution_path: pathlib.Path,
    resume_path: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    run, continuation, _, _, resume_digest = _resume_context(
        run_directory, project_root, project_revision, execution_path, resume_path, contract
    )
    continuation = _require_real_descendant(continuation, run, "reserved continuation")
    _require(continuation.is_dir() and not continuation.is_symlink(), "reserved continuation is not a real directory")
    _require(not any(continuation.iterdir()), "reserved continuation must be empty before evidence publication")
    return {
        "status": "resume_reservation_ok",
        "continuation": str(continuation),
        "resume_manifest_sha256": resume_digest,
    }


def atomic_publish_file(source: pathlib.Path, destination: pathlib.Path) -> None:
    source = source.resolve(strict=True)
    destination = destination.absolute()
    _require(source.is_file() and not source.is_symlink(), "publication source must be a real file")
    _require(destination.parent.is_dir() and not destination.parent.is_symlink(), "publication directory is invalid")
    _require(not destination.exists() and not destination.is_symlink(), "publication destination already exists")
    temporary = destination.parent / f".{destination.name}.publish-{os.getpid()}"
    _require(not temporary.exists(), "publication temporary path already exists")
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(fd, "wb") as handle:
            with source.open("rb") as input_handle:
                for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def atomic_publish_complete(destination: pathlib.Path) -> None:
    destination = destination.absolute()
    staging = destination.parent / f".{destination.name}.content-{os.getpid()}"
    _require(not staging.exists(), "completion staging path already exists")
    staging.write_bytes(b"ACCEPTED_OFFICIAL_TASK0\n")
    try:
        atomic_publish_file(staging, destination)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


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
    preflight_parser.add_argument("--resolved-execution", type=pathlib.Path, required=True)
    preflight_parser.add_argument("--execution", type=pathlib.Path, required=True)
    preflight_parser.add_argument("--result-directory", type=pathlib.Path, required=True)
    accept_parser = subparsers.add_parser("accept")
    accept_parser.add_argument("--run-directory", type=pathlib.Path, required=True)
    accept_parser.add_argument("--project-root", type=pathlib.Path, required=True)
    accept_parser.add_argument("--execution", type=pathlib.Path, required=True)
    for command in (
        "resume-preflight",
        "resume-reservation",
        "resume-accept",
        "resume-composite-accept",
    ):
        resume_parser = subparsers.add_parser(command)
        resume_parser.add_argument("--run-directory", type=pathlib.Path, required=True)
        resume_parser.add_argument("--project-root", type=pathlib.Path, required=True)
        resume_parser.add_argument("--project-revision", required=True)
        resume_parser.add_argument("--execution", type=pathlib.Path, required=True)
        resume_parser.add_argument("--resume", type=pathlib.Path, required=True)
    publish_parser = subparsers.add_parser("publish-file")
    publish_parser.add_argument("--source", type=pathlib.Path, required=True)
    publish_parser.add_argument("--destination", type=pathlib.Path, required=True)
    complete_parser = subparsers.add_parser("publish-complete")
    complete_parser.add_argument("--destination", type=pathlib.Path, required=True)
    models_parser = subparsers.add_parser("models")
    models_parser.add_argument("--models-root", type=pathlib.Path, required=True)
    shards_parser = subparsers.add_parser("shards")
    shards_parser.add_argument("--actor", type=pathlib.Path, required=True)
    shards_parser.add_argument("--world-size", type=int, choices=(4, 8), required=True)
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
            arguments.resolved_execution,
            arguments.execution,
            arguments.result_directory,
            contract,
        )
    elif arguments.command == "accept":
        result = accept(
            arguments.run_directory, arguments.project_root, arguments.execution, contract
        )
    elif arguments.command == "resume-preflight":
        result = resume_preflight(
            arguments.run_directory, arguments.project_root, arguments.project_revision,
            arguments.execution, arguments.resume, contract,
        )
    elif arguments.command == "resume-reservation":
        result = validate_resume_reservation(
            arguments.run_directory, arguments.project_root, arguments.project_revision,
            arguments.execution, arguments.resume, contract,
        )
    elif arguments.command == "resume-accept":
        result = resume_accept(
            arguments.run_directory, arguments.project_root, arguments.project_revision,
            arguments.execution, arguments.resume, contract,
        )
    elif arguments.command == "resume-composite-accept":
        result = resume_composite_accept(
            arguments.run_directory, arguments.project_root, arguments.project_revision,
            arguments.execution, arguments.resume, contract,
        )
    elif arguments.command == "publish-file":
        atomic_publish_file(arguments.source, arguments.destination)
        result = {"status": "published", "destination": str(arguments.destination)}
    elif arguments.command == "publish-complete":
        atomic_publish_complete(arguments.destination)
        result = {"status": "published", "destination": str(arguments.destination)}
    elif arguments.command == "models":
        result = validate_model_files(arguments.models_root, contract)
    else:
        result = {
            "status": "shards_ok",
            "world_size": arguments.world_size,
            "shards": validate_fsdp_shards(arguments.actor, arguments.world_size),
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
