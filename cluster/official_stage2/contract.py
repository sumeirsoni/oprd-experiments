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
    hydra: Mapping[str, Any], *, official_root: pathlib.Path | None = None
) -> None:
    """Validate the actual resolved Hydra configuration used by official verl."""
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
        "trainer.n_gpus_per_node": 8,
        "trainer.nnodes": 1,
        "trainer.total_epochs": 1,
    }
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
    subparsers.add_parser("check-environment")
    subparsers.add_parser("emit-resolved")
    resolved = subparsers.add_parser("validate-resolved")
    resolved.add_argument("--resolved", type=pathlib.Path, required=True)
    hydra = subparsers.add_parser("validate-hydra")
    hydra.add_argument("--hydra", type=pathlib.Path, required=True)
    hydra.add_argument("--official-root", type=pathlib.Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    contract = load_contract(arguments.contract)
    if arguments.command == "digest":
        print(contract_sha256(contract))
    elif arguments.command == "validate":
        print(json.dumps({"ok": True, "contract_sha256": contract_sha256(contract)}))
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
        validate_hydra_config(hydra, official_root=arguments.official_root)
        print(json.dumps({"ok": True, "contract_sha256": contract_sha256(contract)}))


if __name__ == "__main__":
    main()
