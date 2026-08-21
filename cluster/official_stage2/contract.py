#!/usr/bin/env python3
"""Canonical contract and fail-closed validation for official OPRD Stage 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
from typing import Any, Mapping


CONTRACT_PATH = pathlib.Path(__file__).with_name("task0_contract.json")
EXECUTION_PATH = pathlib.Path(__file__).with_name("task0_execution_4h200.json")
RESUME_PATH = pathlib.Path(__file__).with_name("task0_resume_7258005.json")
RESUME_MANIFEST_SHA256 = "7999d929ce351a7f44b591ab23dc9c145441b97e86807af0513c382082c4e143"
FORBIDDEN_SCIENTIFIC_ENVIRONMENT = frozenset(
    {
        "ATTENTION_DISTILLATION",
        "BATCH_SIZE",
        "ENABLE_THINKING",
        "EPOCHS",
        "GRAD_CLIP",
        "IGNORE_EOS",
        "KL_COEF",
        "LAST_K",
        "LEARNING_RATE",
        "LOG_PROB_TOP_K",
        "LOSS_AGG_MODE",
        "LR",
        "LR_SCHEDULER",
        "MAX_PROMPT_LENGTH",
        "MAX_RESPONSE_LENGTH",
        "MINI_BATCH_SIZE",
        "N_RESPONSES",
        "NUM_GPUS",
        "NUM_NODES",
        "REP_DISTILLATION_COEF",
        "REP_DISTILLATION_LAST_K",
        "REP_DISTILLATION_LAYERS",
        "REP_DISTILLATION_ONLY",
        "REP_FREEZE_PS",
        "REP_LOW_RANK",
        "REP_PROJECTOR_MODE",
        "RESPONSES_PER_PROMPT",
        "STEPS",
        "TEMPERATURE",
        "TOP_K",
        "TOP_P",
        "USE_REP_DISTILLATION",
        "WARMUP_RATIO",
        "WARMUP_STEPS",
        "WEIGHT_DECAY",
    }
)


class ContractError(RuntimeError):
    """The task-0 contract or a resolved runtime configuration is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_contract(path: os.PathLike[str] | str = CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def contract_sha256(contract: Mapping[str, Any]) -> str:
    validate_contract(contract)
    return sha256_bytes(canonical_json_bytes(contract))


def load_execution_manifest(
    contract: Mapping[str, Any], path: os.PathLike[str] | str = EXECUTION_PATH
) -> dict[str, Any]:
    manifest = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    validate_execution_manifest(manifest, contract)
    return manifest


def execution_manifest_sha256(
    manifest: Mapping[str, Any], contract: Mapping[str, Any]
) -> str:
    validate_execution_manifest(manifest, contract)
    return sha256_bytes(canonical_json_bytes(manifest))


def load_resume_manifest(
    contract: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
    path: os.PathLike[str] | str = RESUME_PATH,
) -> dict[str, Any]:
    manifest = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    validate_resume_manifest(manifest, contract, execution_manifest)
    return manifest


def resume_manifest_sha256(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
) -> str:
    validate_resume_manifest(manifest, contract, execution_manifest)
    return sha256_bytes(canonical_json_bytes(manifest))


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    _require(isinstance(value, Mapping), f"{field} must be an object")
    actual = set(value)
    _require(actual == expected, f"{field} keys mismatch: expected {sorted(expected)}, got {sorted(actual)}")


def _validate_relative_path(value: Any, field: str) -> None:
    _require(isinstance(value, str) and value != "", f"{field} must be a non-empty path")
    path = pathlib.PurePosixPath(value)
    _require(not path.is_absolute(), f"{field} must be relative")
    _require(".." not in path.parts and "." not in path.parts, f"{field} must not escape its root")
    _require(str(path) == value, f"{field} must be a normalized POSIX path")


def _validate_file_records(records: Any, field: str, *, allow_absolute: bool = False) -> None:
    _require(isinstance(records, list) and records, f"{field} must be a non-empty list")
    paths: list[str] = []
    for index, record in enumerate(records):
        _require_exact_keys(record, {"path", "sha256", "size"}, f"{field}[{index}]")
        path = record["path"]
        if not (allow_absolute and isinstance(path, str) and path.startswith("/")):
            _validate_relative_path(path, f"{field}[{index}].path")
        _require_sha256(record["sha256"], f"{field}[{index}].sha256")
        _require(isinstance(record["size"], int) and not isinstance(record["size"], bool) and record["size"] > 0, f"{field}[{index}].size must be a positive integer")
        paths.append(path)
    _require(len(paths) == len(set(paths)), f"{field} contains duplicate paths")


def validate_resume_manifest(
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    execution_manifest: Mapping[str, Any],
) -> None:
    validate_execution_manifest(execution_manifest, contract)
    _require_exact_keys(manifest, {
        "schema_version", "manifest_name", "scope", "scientific_contract_sha256",
        "execution_manifest_sha256", "source_checkpoint", "source_evidence",
        "resume_arithmetic", "approved_topology", "required_hydra_overrides",
    }, "resume manifest")
    _require(manifest["schema_version"] == 1, "resume schema version mismatch")
    _require(manifest["manifest_name"] == "task0-resume-job-7258005-global-step-1600-r1", "resume manifest name mismatch")
    _require(manifest["scientific_contract_sha256"] == contract_sha256(contract), "resume scientific contract digest mismatch")
    _require(manifest["execution_manifest_sha256"] == execution_manifest_sha256(execution_manifest, contract), "resume execution manifest digest mismatch")
    expected_scope = {
        "task": 0, "phase": "resume_full_training", "original_job_id": "7258005",
        "original_project_revision": "e52fa7d42660e86cd6c4b26acdabfd576029d215",
        "authorized_result_namespace": "official-task0-4h200-prod-r1",
        "continuation_subdirectory": "continuations/resume-global-step-1600-r1",
        "continuation_attempts": 1, "evaluation_authorized": False,
        "tasks_1_through_26_authorized": False,
    }
    _require(manifest["scope"] == expected_scope, "resume authorization scope mismatch")
    _validate_relative_path(manifest["scope"]["continuation_subdirectory"], "scope.continuation_subdirectory")
    _require(manifest["approved_topology"] == execution_manifest["approved_topology"], "resume topology mismatch")

    checkpoint = manifest["source_checkpoint"]
    _require_exact_keys(checkpoint, {"relative_path", "global_step", "world_size", "tracker", "dataloader_state", "inventory"}, "source_checkpoint")
    _validate_relative_path(checkpoint["relative_path"], "source_checkpoint.relative_path")
    _require(checkpoint["relative_path"].endswith("/global_step_1600"), "source checkpoint must be global_step_1600")
    _require(checkpoint["global_step"] == 1600 and checkpoint["world_size"] == 4, "source checkpoint identity mismatch")
    tracker = checkpoint["tracker"]
    _require_exact_keys(tracker, {"relative_path", "sha256", "size", "value"}, "source_checkpoint.tracker")
    _validate_relative_path(tracker["relative_path"], "source_checkpoint.tracker.relative_path")
    _require(tracker == {
        "relative_path": str(pathlib.PurePosixPath(checkpoint["relative_path"]).parent / "latest_checkpointed_iteration.txt"),
        "sha256": "b458944d9ec4322f9ff0de2885ce82d580341fcf0fd688422e9929b6e7ce3b2d",
        "size": 4, "value": 1600,
    }, "source checkpoint tracker mismatch")
    dataloader = checkpoint["dataloader_state"]
    _require(dataloader == {
        "snapshot_step": 1600, "samples_yielded": 12800,
        "sampler_iter_yielded": 1600, "num_workers": 8,
        "iterator_finished": False,
    }, "source dataloader state mismatch")
    _validate_file_records(checkpoint["inventory"], "source_checkpoint.inventory")
    inventory_paths = {record["path"] for record in checkpoint["inventory"]}
    expected_shards = {
        f"actor/{kind}_world_size_4_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state") for rank in range(4)
    }
    _require(expected_shards <= inventory_paths, "source checkpoint shard inventory is incomplete")
    _require("data.pt" in inventory_paths and "actor/fsdp_config.json" in inventory_paths, "source checkpoint metadata inventory is incomplete")
    _require(any(path.startswith("actor/huggingface/") for path in inventory_paths), "source checkpoint Hugging Face inventory is empty")
    _require(len(inventory_paths) == 23, "source checkpoint inventory must contain exactly 23 files")

    _validate_file_records(manifest["source_evidence"], "source_evidence", allow_absolute=True)
    _require([record["path"] for record in manifest["source_evidence"]] == [
        "preflight.json", "resolved-execution-topology.json", "resolved-scientific-config.json",
        "actual-hydra.yaml", "actual-python-argv.json", "official-runtime.log",
        "/cmlscratch/ssoni11/oprd-experiments-main/cluster/logs/oprd_official_task0_7258005.out",
    ], "source evidence inventory mismatch")

    arithmetic = manifest["resume_arithmetic"]
    _require_exact_keys(arithmetic, {"total_rows", "prompt_batch_size", "contract_consumed_rows", "checkpoint_consumed_rows", "remaining_rows", "remaining_updates", "first_resumed_step", "target_step"}, "resume_arithmetic")
    data = contract["training_data"]
    _require(arithmetic["total_rows"] == data["rows"], "resume total rows mismatch")
    _require(arithmetic["prompt_batch_size"] == data["prompt_batch_size"], "resume batch size mismatch")
    _require(arithmetic["contract_consumed_rows"] == data["expected_consumed_rows"], "resume contract consumed rows mismatch")
    _require(arithmetic["checkpoint_consumed_rows"] == dataloader["samples_yielded"], "resume checkpoint consumed rows mismatch")
    _require(arithmetic["remaining_rows"] == arithmetic["contract_consumed_rows"] - arithmetic["checkpoint_consumed_rows"], "resume remaining rows mismatch")
    _require(arithmetic["remaining_rows"] % arithmetic["prompt_batch_size"] == 0, "resume remaining rows must form complete batches")
    _require(arithmetic["remaining_updates"] == arithmetic["remaining_rows"] // arithmetic["prompt_batch_size"], "resume remaining update count mismatch")
    _require(arithmetic["first_resumed_step"] == checkpoint["global_step"] + 1, "first resumed step mismatch")
    _require(arithmetic["target_step"] == checkpoint["global_step"] + arithmetic["remaining_updates"] == data["expected_updates"], "resume target step mismatch")

    _require(manifest["required_hydra_overrides"] == {
        "trainer.resume_mode": "resume_path",
        "trainer.resume_from_path": "${OFFICIAL_NAMESPACE}/${SOURCE_CHECKPOINT_RELATIVE_PATH}",
        "trainer.default_local_dir": "${CONTINUATION_DIR}/checkpoints",
        "trainer.total_training_steps": 2239,
        "trainer.n_gpus_per_node": 4,
        "trainer.max_actor_ckpt_to_keep": 1,
        "hydra.run.dir": "${CONTINUATION_DIR}/hydra-run",
    }, "required resume Hydra overrides mismatch")
    _require(
        sha256_bytes(canonical_json_bytes(manifest)) == RESUME_MANIFEST_SHA256,
        "resume manifest immutable digest mismatch",
    )


def effective_resume_authorization(
    manifest: Mapping[str, Any], contract: Mapping[str, Any], execution_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    validate_resume_manifest(manifest, contract, execution_manifest)
    return {
        "scientific_contract_sha256": contract_sha256(contract),
        "execution_manifest_sha256": execution_manifest_sha256(execution_manifest, contract),
        "resume_manifest_sha256": resume_manifest_sha256(manifest, contract, execution_manifest),
        "scope": manifest["scope"], "source_checkpoint": manifest["source_checkpoint"],
        "source_evidence": manifest["source_evidence"], "resume_arithmetic": manifest["resume_arithmetic"],
        "approved_topology": manifest["approved_topology"],
        "required_hydra_overrides": manifest["required_hydra_overrides"],
    }


def validate_execution_manifest(
    manifest: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    validate_contract(contract)
    _require(manifest.get("schema_version") == 1, "execution schema version mismatch")
    _require(
        manifest.get("manifest_name") == "task0-full-production-four-h200",
        "execution manifest name mismatch",
    )
    scope = manifest["scope"]
    _require(scope == {
        "task": 0,
        "phase": "full_training",
        "production_runs": 1,
        "authorized_result_namespace": "official-task0-4h200-prod-r1",
        "evaluation_authorized": False,
        "tasks_1_through_26_authorized": False,
    }, "execution authorization scope mismatch")
    _require(
        manifest["scientific_contract_sha256"] == contract_sha256(contract),
        "execution scientific contract digest mismatch",
    )
    distributed = contract["distributed"]
    _require(manifest["historical_topology"] == {
        "nodes": distributed["nodes"],
        "gpus_per_node": distributed["gpus_per_node"],
        "world_size": distributed["world_size"],
    }, "historical topology mismatch")
    _require(manifest["approved_topology"] == {
        "nodes": 1,
        "gpus_per_node": 4,
        "world_size": 4,
        "accelerator": "h200-sxm",
    }, "approved topology mismatch")
    _require(manifest["allowed_hydra_deltas"] == [
        {
            "path": "trainer.n_gpus_per_node",
            "operation": "replace",
            "from": 8,
            "to": 4,
        },
        {
            "path": "trainer.max_actor_ckpt_to_keep",
            "operation": "set",
            "from": None,
            "to": 1,
        },
        {
            "path": "hydra.run.dir",
            "operation": "set",
            "from": "outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}",
            "to": "${PROJECT_PATH}/hydra-run",
        },
    ], "execution manifest Hydra deltas mismatch")
    _require(manifest["checkpoint_policy"] == {
        "save_frequency_updates": 200,
        "max_actor_checkpoints_to_keep": 1,
        "resume_checkpoints_preserved": True,
    }, "execution checkpoint policy mismatch")
    _require(manifest["prior_failed_attempts"] == [
        {
            "job_id": "7257861",
            "project_revision": "aa99ba1780a68f70cc7d0d1a15fd45e4ca696586",
            "phase": "ray_start_before_model_loading",
            "failure": "AF_UNIX socket path exceeded 107 bytes",
            "runtime_log_sha256": "85c921eaf831e66cd73c1a8bf0c3bedee3655c5ea20a7158a81fc53f3f2aa126",
            "slurm_log_sha256": "f2095f0f1b06e4f58f34f5fd5431e9a478c068a024b26cd704d039d5a81fdc51",
            "archived_locally": True,
        },
        {
            "job_id": "7257986",
            "project_revision": "d0d3dabcc1c3f85fbc2a846d4bcf93324c2b9f94",
            "phase": "vllm_model_initialization_before_training",
            "failure": "expandable CUDA allocator is incompatible with vLLM memory pool",
            "runtime_log_sha256": "2ed641a012072867e1a8d5763d9e22c863bd70627d86f9aef54054bbfb47177d",
            "slurm_log_sha256": "972c58a45448c11568ce16a0381d1194b9b5dba7a3ced7775ceb1c2492e7c4bb",
            "archived_locally": True,
        },
        {
            "job_id": "7257995",
            "project_revision": "f21646b29a63b79906264624be65dd32c225159f",
            "phase": "authenticated_preflight_before_namespace_reservation",
            "failure": "prior Hydra output polluted the authenticated source tree",
            "slurm_log_sha256": "43d08ff866410aefc9fe14ffad945335a4c154ece762ff5732a683d53101bba6",
            "source_output_config_sha256": "8f5989a411d7961298416968a90a5f843bb35595293f5ebd5cc89b8c639a2acf",
            "source_output_hydra_sha256": "7e7f5e5580fb9dd2a5efd729febb3214abfb41627dabee09a2eb2aa58b2ffdef",
            "source_output_overrides_sha256": "acb3708508f4c0b09448e3539113c1fdc4b761aea9092381280bda5ff39c632f",
            "archived_locally": True,
        },
    ], "prior failed attempt records mismatch")
    expected_invariants = {
        "prompt_batch_size": contract["training_data"]["prompt_batch_size"],
        "responses_per_prompt": contract["rollout"]["responses_per_prompt"],
        "responses_per_update": contract["rollout"]["responses_per_update"],
        "expected_updates": contract["training_data"]["expected_updates"],
        "model_dtype": "fp32",
        "optimizer": contract["optimizer"]["name"],
        "learning_rate": contract["optimizer"]["learning_rate"],
        "weight_decay": contract["optimizer"]["weight_decay"],
        "scheduler": contract["optimizer"]["scheduler"],
        "loss_aggregation": contract["objective"]["loss_aggregation"],
        "representation_coefficient": contract["objective"]["coefficient"],
    }
    _require(
        manifest["preserved_invariants"] == expected_invariants,
        "execution preserved invariants mismatch",
    )
    canary = manifest["canary_evidence"]
    _require(canary["rows"] == 80 and canary["expected_updates"] == 10, "canary size mismatch")
    _require(canary["world_size"] == 4, "canary world size mismatch")
    _require_git_commit(canary["execution_revision"], "canary execution revision")
    _require_git_commit(canary["acceptance_revision"], "canary acceptance revision")
    for field in (
        "smoke_result_sha256", "actual_hydra_sha256", "actual_argv_sha256",
        "runtime_log_sha256", "terminal_log_sha256",
    ):
        _require_sha256(canary[field], f"canary {field}")


def effective_execution_topology(
    manifest: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    validate_execution_manifest(manifest, contract)
    return {
        "scientific_contract_sha256": contract_sha256(contract),
        "execution_manifest_sha256": execution_manifest_sha256(manifest, contract),
        "historical_topology": manifest["historical_topology"],
        "effective_topology": manifest["approved_topology"],
        "allowed_hydra_deltas": manifest["allowed_hydra_deltas"],
        "checkpoint_policy": manifest["checkpoint_policy"],
        "prior_failed_attempts": manifest["prior_failed_attempts"],
        "preserved_invariants": manifest["preserved_invariants"],
        "canary_evidence": manifest["canary_evidence"],
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _require_sha256(value: Any, field: str) -> None:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{field} must be a lowercase SHA-256 digest",
    )


def _require_git_commit(value: Any, field: str) -> None:
    _require(
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value),
        f"{field} must be a lowercase 40-character Git object ID",
    )


def validate_contract(contract: Mapping[str, Any]) -> None:
    _require(contract.get("schema_version") == 1, "unsupported contract schema")
    source = contract["official_source"]
    _require_git_commit(source["commit"], "official_source.commit")
    _require(source["require_clean_tree"] is True, "official source must be clean")

    models = contract["models"]
    for name in ("student", "teacher", "tokenizer"):
        _require_git_commit(models[name]["revision"], f"models.{name}.revision")
    _require(models["student"]["num_layers"] == 28, "student layer count mismatch")
    _require(models["teacher"]["num_layers"] == 36, "teacher layer count mismatch")

    bridge = contract["stage1_bridge"]
    _require_sha256(
        bridge["artifact_manifest_sha256"],
        "stage1_bridge.artifact_manifest_sha256",
    )
    _require_sha256(
        bridge["stage1_contract_sha256"],
        "stage1_bridge.stage1_contract_sha256",
    )
    _require(bridge["rank"] == 8, "bridge rank must be 8")

    data = contract["training_data"]
    _require_git_commit(data["dataset_revision"], "training_data.dataset_revision")
    _require_git_commit(data["official_git_blob"], "training_data.official_git_blob")
    _require(data["post_filter_rows"] == data["rows"], "unexpected filtered DAPO row count")
    _require(data["overlong_prompt_rows"] == 0, "official DAPO should have no overlong prompts")
    _require(data["shuffle"] is False, "training data must not shuffle")
    _require(data["drop_last"] is True, "training data must drop the final partial batch")
    expected_updates = data["post_filter_rows"] // data["prompt_batch_size"]
    expected_consumed = expected_updates * data["prompt_batch_size"]
    expected_tail = list(range(expected_consumed, data["post_filter_rows"]))
    _require(data["expected_updates"] == expected_updates, "update count mismatch")
    _require(data["expected_consumed_rows"] == expected_consumed, "consumed row count mismatch")
    _require(data["expected_dropped_tail_rows"] == expected_tail, "drop-last rows mismatch")
    _require(data["enable_thinking"] is False, "training thinking must be disabled")

    rollout = contract["rollout"]
    _require(rollout["responses_per_prompt"] == 2, "must sample two responses per prompt")
    _require(
        rollout["responses_per_update"]
        == data["prompt_batch_size"] * rollout["responses_per_prompt"],
        "responses per update mismatch",
    )
    _require(rollout["temperature"] == 1.0, "training temperature must be 1.0")
    _require(rollout["top_p"] == 1.0, "training top-p must be 1.0")
    _require(rollout["top_k"] == -1, "training top-k must be disabled")
    _require(rollout["max_response_tokens"] == 16384, "response cap mismatch")
    _require(rollout["fresh_responses_each_update"] is True, "rollouts must be fresh")
    _require(rollout["retain_cap_hit_responses"] is True, "cap hitters must be retained")

    objective = contract["objective"]
    _require(objective["representation_distillation"] is True, "representation loss is required")
    _require(objective["representation_only"] is True, "task 0 must be representation-only")
    _require(objective["rank"] == bridge["rank"], "objective rank mismatch")
    _require(objective["last_k"] == 2000, "last-k mask mismatch")
    _require(objective["coefficient"] == 10.0, "representation coefficient mismatch")
    for field in (
        "freeze_student_projectors",
        "freeze_teacher_projectors",
        "freeze_teacher_backbone",
        "train_student_backbone_only",
    ):
        _require(objective[field] is True, f"objective.{field} must be true")
    _require(objective["reward_model_execution"] is True, "official reward model execution must remain enabled")
    for field in (
        "policy_gradient_enabled",
        "reward_gradient_enabled",
        "kl_enabled",
        "attention_distillation_enabled",
    ):
        _require(objective[field] is False, f"objective.{field} must be false")
    _require(objective["log_prob_top_k"] == 0, "output-space distillation must be disabled")

    optimizer = contract["optimizer"]
    _require(optimizer["name"] == "AdamW", "optimizer must be AdamW")
    _require(optimizer["learning_rate"] == 1e-5, "learning rate mismatch")
    _require(optimizer["weight_decay"] == 0.01, "weight decay mismatch")
    _require(optimizer["gradient_clip_norm"] == 1000.0, "gradient clip mismatch")
    _require(optimizer["scheduler"] == "constant", "scheduler must resolve to constant")
    _require(optimizer["warmup_steps"] == 0, "warmup steps must be zero")
    _require(optimizer["warmup_ratio"] == 0.0, "warmup ratio must be zero")

    distributed = contract["distributed"]
    _require(distributed["nodes"] == 1, "task 0 requires one node")
    _require(distributed["gpus_per_node"] == 8, "task 0 requires eight GPUs")
    _require(distributed["world_size"] == 8, "world size must be eight")
    _require(distributed["allow_single_gpu_fallback"] is False, "single-GPU fallback forbidden")
    _require(distributed["allow_hf_generation_fallback"] is False, "HF fallback forbidden")


def reject_scientific_environment(environment: Mapping[str, str] | None = None) -> None:
    environment = os.environ if environment is None else environment
    inherited = sorted(key for key in FORBIDDEN_SCIENTIFIC_ENVIRONMENT if key in environment)
    if inherited:
        raise ContractError(
            "official task 0 refuses inherited scientific environment variables: "
            + ", ".join(inherited)
        )


def _lookup(config: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for component in dotted_path.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise ContractError(f"resolved configuration is missing {dotted_path}")
        value = value[component]
    return value


def expected_resolved_config(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the version-stable scientific configuration expected from Hydra."""
    return {
        "data": {
            "rows": contract["training_data"]["rows"],
            "shuffle": False,
            "drop_last": True,
            "prompt_batch_size": 8,
            "expected_updates": contract["training_data"]["expected_updates"],
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


def validate_resolved_config(
    resolved: Mapping[str, Any], contract: Mapping[str, Any]
) -> None:
    """Validate the normalized sidecar emitted by the official launcher.

    The launcher must translate Hydra's version-specific field layout into these stable
    names before training. This keeps scientific validation independent of verl's config
    nesting while still checking every consequential value.
    """

    expected = {
        "data.rows": contract["training_data"]["rows"],
        "data.shuffle": False,
        "data.drop_last": True,
        "data.prompt_batch_size": 8,
        "data.expected_updates": contract["training_data"]["expected_updates"],
        "data.enable_thinking": False,
        "rollout.responses_per_prompt": 2,
        "rollout.temperature": 1.0,
        "rollout.top_p": 1.0,
        "rollout.top_k": -1,
        "rollout.max_response_tokens": 16384,
        "rollout.retain_cap_hit_responses": True,
        "objective.representation_only": True,
        "objective.rank": 8,
        "objective.layers": "all",
        "objective.last_k": 2000,
        "objective.coefficient": 10.0,
        "objective.log_prob_top_k": 0,
        "optimizer.learning_rate": 1e-5,
        "optimizer.weight_decay": 0.01,
        "optimizer.gradient_clip_norm": 1000.0,
        "optimizer.scheduler": "constant",
        "optimizer.warmup_steps": 0,
        "distributed.nodes": 1,
        "distributed.gpus_per_node": 8,
        "distributed.world_size": 8,
        "distributed.rollout_tensor_parallel_size": 1,
    }
    mismatches = []
    for dotted_path, expected_value in expected.items():
        actual_value = _lookup(resolved, dotted_path)
        if actual_value != expected_value:
            mismatches.append(
                f"{dotted_path}: expected {expected_value!r}, got {actual_value!r}"
            )
    if mismatches:
        raise ContractError("resolved configuration mismatch:\n" + "\n".join(mismatches))


def validate_hydra_config(
    hydra: Mapping[str, Any],
    *,
    official_root: pathlib.Path | None = None,
    expected_world_size: int = 8,
    execution_manifest: Mapping[str, Any] | None = None,
    resume_manifest: Mapping[str, Any] | None = None,
    official_namespace: pathlib.Path | None = None,
    continuation_dir: pathlib.Path | None = None,
    contract: Mapping[str, Any] | None = None,
) -> None:
    """Validate the actual resolved Hydra configuration used by official verl."""
    if execution_manifest is not None:
        _require(contract is not None, "contract is required with execution manifest")
        validate_execution_manifest(execution_manifest, contract)
        expected_world_size = execution_manifest["approved_topology"]["world_size"]
    if resume_manifest is not None:
        _require(contract is not None and execution_manifest is not None, "contract and execution manifest are required with resume manifest")
        validate_resume_manifest(resume_manifest, contract, execution_manifest)
        _require(official_namespace is not None and continuation_dir is not None, "official namespace and continuation directory are required for resume Hydra validation")
        expected_world_size = resume_manifest["approved_topology"]["world_size"]
    _require(expected_world_size in (4, 8), "expected world size must be four or eight")
    expected = {
        "data.shuffle": False,
        "data.train_batch_size": 8,
        "data.max_prompt_length": 2048,
        "data.max_response_length": 16384,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.return_raw_chat": True,
        "data.apply_chat_template_kwargs.enable_thinking": False,
        "actor_rollout_ref.actor.optim.lr": 1e-5,
        "actor_rollout_ref.actor.optim.weight_decay": 0.01,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
        "actor_rollout_ref.actor.ppo_mini_batch_size": 8,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.shuffle": False,
        "actor_rollout_ref.actor.grad_clip": 1000.0,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.use_rep_distillation": True,
        "actor_rollout_ref.actor.rep_distillation_only": True,
        "actor_rollout_ref.actor.rep_distillation_coef": 10.0,
        "actor_rollout_ref.actor.rep_distillation_positions": "last_k",
        "actor_rollout_ref.actor.rep_distillation_last_k": 2000,
        "actor_rollout_ref.actor.rep_distillation_layers": "all",
        "actor_rollout_ref.actor.rep_projector_mode": "low_rank",
        "actor_rollout_ref.actor.rep_low_rank": 8,
        "actor_rollout_ref.actor.rep_freeze_ps": True,
        "actor_rollout_ref.actor.use_att_distillation": False,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
        "actor_rollout_ref.ref.fsdp_config.model_dtype": "fp32",
        "reward_model.model.dtype": "fp32",
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.repetition_penalty": 1.0,
        "actor_rollout_ref.rollout.ignore_eos": False,
        "actor_rollout_ref.rollout.max_model_len": 18432,
        "actor_rollout_ref.rollout.n": 2,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.log_prob_top_k": 0,
        "reward_model.enable": True,
        "trainer.n_gpus_per_node": expected_world_size,
        "trainer.nnodes": 1,
        "trainer.total_epochs": 1,
    }
    if execution_manifest is not None:
        expected["trainer.save_freq"] = execution_manifest["checkpoint_policy"][
            "save_frequency_updates"
        ]
        expected["trainer.max_actor_ckpt_to_keep"] = execution_manifest[
            "checkpoint_policy"
        ]["max_actor_checkpoints_to_keep"]
    if resume_manifest is not None:
        namespace = official_namespace.resolve(strict=True)
        continuation = continuation_dir.resolve(strict=True)
        source_checkpoint = namespace / resume_manifest["source_checkpoint"]["relative_path"]
        _require(source_checkpoint.resolve(strict=True) == source_checkpoint, "resume checkpoint path must be canonical")
        expected.update({
            "trainer.resume_mode": "resume_path",
            "trainer.resume_from_path": str(source_checkpoint),
            "trainer.default_local_dir": str(continuation / "checkpoints"),
            "trainer.output_log_path": str(continuation / "logs" / "terminal" / "resume.log"),
            "trainer.total_training_steps": resume_manifest["resume_arithmetic"]["target_step"],
            "hydra.run.dir": str(continuation / "hydra-run"),
        })
    mismatches = []
    for dotted_path, expected_value in expected.items():
        actual_value = _lookup(hydra, dotted_path)
        if actual_value != expected_value:
            mismatches.append(
                f"{dotted_path}: expected {expected_value!r}, got {actual_value!r}"
            )
    if official_root is not None:
        root = official_root.resolve(strict=True)
        path_expectations = {
            "data.train_files": "../datasets/dapo-math-17k.parquet",
            "data.val_files": ["../datasets/test_data/AMC23/test.parquet"],
            "actor_rollout_ref.model.path": str(root / "models" / "Qwen3-1.7B-Base"),
            "reward_model.model.path": str(root / "models" / "Qwen3-4B"),
            "actor_rollout_ref.actor.rep_low_rank_init_checkpoint": str(
                root / "bridge" / "rank8_all_layers" / "ps_bank.pt"
            ),
        }
        for dotted_path, expected_value in path_expectations.items():
            actual_value = _lookup(hydra, dotted_path)
            if actual_value != expected_value:
                mismatches.append(
                    f"{dotted_path}: expected {expected_value!r}, got {actual_value!r}"
                )
    if mismatches:
        raise ContractError("actual Hydra configuration mismatch:\n" + "\n".join(mismatches))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=pathlib.Path, default=CONTRACT_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("digest")
    subparsers.add_parser("validate")
    validate_execution = subparsers.add_parser("validate-execution")
    validate_execution.add_argument("--execution", type=pathlib.Path, required=True)
    emit_execution = subparsers.add_parser("emit-execution")
    emit_execution.add_argument("--execution", type=pathlib.Path, required=True)
    for command in ("validate-resume", "digest-resume", "emit-resume"):
        resume = subparsers.add_parser(command)
        resume.add_argument("--execution", type=pathlib.Path, required=True)
        resume.add_argument("--resume", type=pathlib.Path, required=True)
    subparsers.add_parser("check-environment")
    subparsers.add_parser("emit-resolved")
    resolved = subparsers.add_parser("validate-resolved")
    resolved.add_argument("--resolved", type=pathlib.Path, required=True)
    hydra = subparsers.add_parser("validate-hydra")
    hydra.add_argument("--hydra", type=pathlib.Path, required=True)
    hydra.add_argument("--official-root", type=pathlib.Path, required=True)
    hydra.add_argument("--expected-world-size", type=int, choices=(4, 8), default=8)
    hydra.add_argument("--execution", type=pathlib.Path)
    hydra.add_argument("--resume", type=pathlib.Path)
    hydra.add_argument("--official-namespace", type=pathlib.Path)
    hydra.add_argument("--continuation-dir", type=pathlib.Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    contract = load_contract(arguments.contract)
    if arguments.command == "digest":
        print(contract_sha256(contract))
    elif arguments.command == "validate":
        print(json.dumps({"ok": True, "contract_sha256": contract_sha256(contract)}))
    elif arguments.command == "validate-execution":
        manifest = load_execution_manifest(contract, arguments.execution)
        print(json.dumps({
            "ok": True,
            "contract_sha256": contract_sha256(contract),
            "execution_manifest_sha256": execution_manifest_sha256(manifest, contract),
        }))
    elif arguments.command == "emit-execution":
        manifest = load_execution_manifest(contract, arguments.execution)
        print(json.dumps(
            effective_execution_topology(manifest, contract),
            sort_keys=True,
            separators=(",", ":"),
        ))
    elif arguments.command in ("validate-resume", "digest-resume", "emit-resume"):
        execution = load_execution_manifest(contract, arguments.execution)
        resume = load_resume_manifest(contract, execution, arguments.resume)
        digest = resume_manifest_sha256(resume, contract, execution)
        if arguments.command == "digest-resume":
            print(digest)
        elif arguments.command == "emit-resume":
            print(json.dumps(effective_resume_authorization(resume, contract, execution), sort_keys=True, separators=(",", ":")))
        else:
            print(json.dumps({"ok": True, "contract_sha256": contract_sha256(contract), "execution_manifest_sha256": execution_manifest_sha256(execution, contract), "resume_manifest_sha256": digest}))
    elif arguments.command == "check-environment":
        reject_scientific_environment()
        print(json.dumps({"ok": True}))
    elif arguments.command == "emit-resolved":
        print(
            json.dumps(
                expected_resolved_config(contract),
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    elif arguments.command == "validate-resolved":
        resolved = json.loads(arguments.resolved.read_text(encoding="utf-8"))
        validate_resolved_config(resolved, contract)
        print(json.dumps({"ok": True, "contract_sha256": contract_sha256(contract)}))
    elif arguments.command == "validate-hydra":
        try:
            import yaml
        except ImportError as exc:
            raise ContractError("PyYAML is required to validate resolved Hydra config") from exc
        hydra = yaml.safe_load(arguments.hydra.read_text(encoding="utf-8"))
        execution_manifest = (
            load_execution_manifest(contract, arguments.execution)
            if arguments.execution
            else None
        )
        resume_manifest = (
            load_resume_manifest(contract, execution_manifest, arguments.resume)
            if arguments.resume and execution_manifest is not None
            else None
        )
        _require(not arguments.resume or execution_manifest is not None, "--resume requires --execution")
        validate_hydra_config(
            hydra,
            official_root=arguments.official_root,
            expected_world_size=arguments.expected_world_size,
            execution_manifest=execution_manifest,
            resume_manifest=resume_manifest,
            official_namespace=arguments.official_namespace,
            continuation_dir=arguments.continuation_dir,
            contract=contract,
        )
        print(json.dumps({"ok": True, "contract_sha256": contract_sha256(contract)}))


if __name__ == "__main__":
    main()
