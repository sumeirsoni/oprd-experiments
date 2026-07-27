#!/usr/bin/env python3
"""Build the fixed Qwen3-4B to Qwen3-1.7B OPRD Stage-1 bridge.

The CLI is deliberately phase-oriented. Every phase writes immutable or atomically
replaceable outputs, records input and output checksums in ``state.json``, and can be
re-run safely after interruption. Expensive model dependencies are imported lazily.

This is the selected paper/public-code-faithful contract, not a general experiment
launcher. Runtime controls such as device, dtype, batching, and vLLM memory use are
configurable; scientific settings are fixed in :data:`FIXED_CONTRACT`.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import datetime as dt
import fcntl
import hashlib
import heapq
import importlib.metadata as importlib_metadata
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile
import traceback
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from oprd_bridge import (
    ArtifactAlreadyExistsError,
    BridgeBank,
    canonical_stage1_contract,
    load_bridge_artifact,
    publish_bridge_artifact,
    read_bridge_manifest,
    resolve_layer_map,
    stage1_contract_sha256,
)


SCHEMA_VERSION = 1
STATE_FILENAME = "state.json"
CONTRACT_FILENAME = "contract.json"
LOCK_FILENAME = ".build.lock"

PRODUCTION_CONTRACT = canonical_stage1_contract()
FIXED_CONTRACT: dict[str, Any] = copy.deepcopy(PRODUCTION_CONTRACT)


def _smoke_contract() -> dict[str, Any]:
    return canonical_stage1_contract(smoke=True)


DEFAULT_STUDENT = "Qwen/Qwen3-1.7B-Base"
DEFAULT_TEACHER = "Qwen/Qwen3-4B"
DEFAULT_TOKENIZER = DEFAULT_STUDENT
DEFAULT_DATASET = "BytedTsinghua-SIA/DAPO-Math-17k"
DEFAULT_DATASET_SPLIT = "train"
DEFAULT_PROMPT_KEY = "prompt"


class BuildError(RuntimeError):
    """A contract, phase-state, or artifact validation error."""


@dataclasses.dataclass(frozen=True)
class Paths:
    root: pathlib.Path

    @property
    def contract(self) -> pathlib.Path:
        return self.root / CONTRACT_FILENAME

    @property
    def state(self) -> pathlib.Path:
        return self.root / STATE_FILENAME

    @property
    def lock(self) -> pathlib.Path:
        return self.root / LOCK_FILENAME

    @property
    def corpus(self) -> pathlib.Path:
        return self.root / "corpus"

    @property
    def prompts(self) -> pathlib.Path:
        return self.corpus / "prompts.jsonl"

    @property
    def attempts(self) -> pathlib.Path:
        return self.corpus / "attempts"

    @property
    def prompts_commit(self) -> pathlib.Path:
        return self.corpus / "prompts.commit.json"

    @property
    def attempt_journal(self) -> pathlib.Path:
        return self.corpus / "attempt_journal.json"

    @property
    def rollouts(self) -> pathlib.Path:
        return self.corpus / "rollouts.jsonl"

    @property
    def split(self) -> pathlib.Path:
        return self.corpus / "split.json"

    @property
    def corpus_metadata(self) -> pathlib.Path:
        return self.corpus / "metadata.json"

    @property
    def teacher(self) -> pathlib.Path:
        return self.root / "teacher"

    @property
    def pca_tensors(self) -> pathlib.Path:
        return self.teacher / "pca.safetensors"

    @property
    def pca_metadata(self) -> pathlib.Path:
        return self.teacher / "pca.json"

    @property
    def targets(self) -> pathlib.Path:
        return self.root / "teacher_targets"

    @property
    def targets_index(self) -> pathlib.Path:
        return self.targets / "index.json"

    @property
    def target_journal(self) -> pathlib.Path:
        return self.teacher / "target_journal.json"

    @property
    def groups(self) -> pathlib.Path:
        return self.root / "student_groups"

    def group(self, group_index: int) -> pathlib.Path:
        return self.groups / f"group_{group_index}"

    def cache(self, group_index: int) -> pathlib.Path:
        return self.root / "student_cache" / f"group_{group_index}"

    def cache_journal(self, group_index: int) -> pathlib.Path:
        return self.root / "student_cache_journals" / f"group_{group_index}.json"

    @property
    def selection(self) -> pathlib.Path:
        return self.root / "publish_selection.json"

    @property
    def publish_provenance(self) -> pathlib.Path:
        return self.root / "publish_provenance.json"

    @property
    def verify_report(self) -> pathlib.Path:
        return self.root / "verify.json"


# ---------------------------------------------------------------------------
# Atomic files, checksums, and phase state


def _utc_now() -> str:
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


def _fsync_dir(path: pathlib.Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: pathlib.Path, value: Any) -> None:
    _atomic_write_bytes(path, _canonical_json_bytes(value))


def _atomic_write_jsonl(path: pathlib.Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = bytearray()
    for row in rows:
        payload.extend(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
                "utf-8"
            )
        )
        payload.extend(b"\n")
    _atomic_write_bytes(path, bytes(payload))


def _atomic_torch_save(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _safetensors_api() -> tuple[Any, Any, Any]:
    try:
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise BuildError(
            "safetensors is required by the bridge build contract; install cluster/requirements.txt"
        ) from exc
    return load_file, save_file, safe_open


def _require_finite_tensors(
    tensors: Mapping[str, torch.Tensor], *, context: str
) -> None:
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise BuildError(f"{context} contains a non-tensor value: {name}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise BuildError(f"{context} contains a non-finite tensor: {name}")


def _atomic_save_safetensors(
    path: pathlib.Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, str] | None = None,
) -> None:
    normalized = {
        key: value.detach().cpu().contiguous() for key, value in tensors.items()
    }
    _require_finite_tensors(normalized, context=f"safetensors output {path}")
    _, save_file, _ = _safetensors_api()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file(normalized, os.fspath(temporary), metadata=dict(metadata or {}))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_tree_and_journal(
    tree: pathlib.Path, journal: pathlib.Path
) -> None:
    if os.path.lexists(os.fspath(journal)):
        if journal.is_dir() and not journal.is_symlink():
            raise BuildError(f"cache journal is unexpectedly a directory: {journal}")
        journal.unlink()
    if os.path.lexists(os.fspath(tree)):
        if tree.is_symlink() or not tree.is_dir():
            tree.unlink()
        else:
            shutil.rmtree(tree)
    remaining = [
        path for path in (tree, journal) if os.path.lexists(os.fspath(path))
    ]
    if remaining:
        raise BuildError(
            "cache cleanup did not remove: "
            + ", ".join(os.fspath(path) for path in remaining)
        )


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(path: pathlib.Path) -> str:
    if path.is_file():
        return _sha256_file(path)
    if not path.is_dir():
        raise BuildError(f"cannot checksum missing output: {path}")
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _path_label(root: pathlib.Path, path: pathlib.Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return os.fspath(path.resolve())


def _load_json(path: pathlib.Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read valid JSON from {path}: {exc}") from exc


def _load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise BuildError(f"{path}:{line_number} is not a JSON object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BuildError(f"cannot read valid JSONL from {path}: {exc}") from exc
    return rows


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _utc_now(),
        "phases": {},
    }


def _read_state(paths: Paths) -> dict[str, Any]:
    if not paths.state.exists():
        return _empty_state()
    state = _load_json(paths.state)
    if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
        raise BuildError(f"unsupported or invalid state file: {paths.state}")
    if not isinstance(state.get("phases"), dict):
        raise BuildError(f"state file has no valid phases mapping: {paths.state}")
    return state


def _write_state(paths: Paths, state: dict[str, Any]) -> None:
    state["updated_at"] = _utc_now()
    _atomic_write_json(paths.state, state)


@contextlib.contextmanager
def _workspace_lock(paths: Paths) -> Iterator[None]:
    paths.root.mkdir(parents=True, exist_ok=True)
    with paths.lock.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BuildError(
                f"another bridge phase holds the workspace lock: {paths.lock}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _outputs_match(paths: Paths, phase: Mapping[str, Any]) -> bool:
    outputs = phase.get("outputs")
    if not isinstance(outputs, list):
        return False
    for output in outputs:
        if not isinstance(output, dict):
            return False
        raw_path = output.get("path")
        expected = output.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            return False
        candidate = pathlib.Path(raw_path)
        if not candidate.is_absolute():
            candidate = paths.root / candidate
        if not candidate.exists() or _sha256_tree(candidate) != expected:
            return False
    return True


def _require_completed_phase_output(
    paths: Paths,
    phase_name: str,
    *,
    state: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    current = state if state is not None else _read_state(paths)
    phases = current.get("phases") if isinstance(current, Mapping) else None
    phase = phases.get(phase_name) if isinstance(phases, Mapping) else None
    if not isinstance(phase, Mapping) or phase.get("status") != "complete":
        raise BuildError(f"required phase is not complete: {phase_name}")
    if not _outputs_match(paths, phase):
        raise BuildError(f"phase output checksum failed: {phase_name}")
    return phase


def _begin_phase(
    paths: Paths,
    phase_name: str,
    input_fingerprint: str,
) -> tuple[dict[str, Any], bool]:
    state = _read_state(paths)
    previous = state["phases"].get(phase_name)
    if (
        isinstance(previous, dict)
        and previous.get("input_fingerprint") != input_fingerprint
    ):
        raise BuildError(
            f"phase {phase_name!r} already started with different inputs; "
            "use the original settings or a new work directory"
        )
    if isinstance(previous, dict) and previous.get("status") == "complete":
        if not _outputs_match(paths, previous):
            raise BuildError(
                f"phase {phase_name!r} is marked complete but an output checksum failed"
            )
        print(f"{phase_name}: already complete and checksums match", flush=True)
        return state, False

    attempts = 1
    if isinstance(previous, dict):
        attempts = int(previous.get("attempts", 0)) + 1
    commitments = (
        dict(previous.get("commitments", {}))
        if isinstance(previous, dict) and isinstance(previous.get("commitments"), dict)
        else {}
    )
    state["phases"][phase_name] = {
        "status": "running",
        "input_fingerprint": input_fingerprint,
        "started_at": _utc_now(),
        "attempts": attempts,
        "pid": os.getpid(),
        "commitments": commitments,
    }
    _write_state(paths, state)
    return state, True


def _finish_phase(
    paths: Paths,
    state: dict[str, Any],
    phase_name: str,
    output_paths: Sequence[pathlib.Path],
    details: Mapping[str, Any],
) -> None:
    outputs = [
        {
            "path": _path_label(paths.root, path),
            "sha256": _sha256_tree(path),
        }
        for path in output_paths
    ]
    phase = state["phases"][phase_name]
    phase.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "outputs": outputs,
            "details": dict(details),
        }
    )
    phase.pop("pid", None)
    _write_state(paths, state)


def _fail_phase(paths: Paths, phase_name: str, exc: BaseException) -> None:
    state = _read_state(paths)
    phase = state["phases"].get(phase_name)
    if isinstance(phase, dict) and phase.get("status") == "running":
        phase.update(
            {
                "status": "failed",
                "failed_at": _utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        phase.pop("pid", None)
        _write_state(paths, state)


@contextlib.contextmanager
def _phase(
    paths: Paths,
    phase_name: str,
    input_fingerprint: str,
) -> Iterator[tuple[dict[str, Any], bool]]:
    state, should_run = _begin_phase(paths, phase_name, input_fingerprint)
    try:
        yield state, should_run
    except BaseException as exc:
        _fail_phase(paths, phase_name, exc)
        raise


# ---------------------------------------------------------------------------
# Contract, models, layers, and corpus records


def _resolve_source_revision(
    identifier: str, repo_type: str, requested_revision: str = ""
) -> str:
    local = pathlib.Path(identifier).expanduser()
    if local.is_file():
        resolved = f"local-file-sha256:{_sha256_file(local)}"
        if requested_revision and requested_revision != resolved:
            raise BuildError(
                f"supplied {repo_type} revision does not match local file: "
                f"actual={resolved!r}, supplied={requested_revision!r}"
            )
        return resolved
    if local.is_dir():
        resolved = f"local-tree-sha256:{_sha256_tree(local)}"
        if requested_revision and requested_revision != resolved:
            raise BuildError(
                f"supplied {repo_type} revision does not match local tree: "
                f"actual={resolved!r}, supplied={requested_revision!r}"
            )
        return resolved
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise BuildError("huggingface_hub is required to pin remote revisions") from exc
    try:
        api = HfApi()
        kwargs = {"revision": requested_revision} if requested_revision else {}
        info = (
            api.dataset_info(identifier, **kwargs)
            if repo_type == "dataset"
            else api.model_info(identifier, **kwargs)
        )
    except Exception as exc:
        requested = f" at revision {requested_revision!r}" if requested_revision else ""
        raise BuildError(
            f"cannot resolve immutable {repo_type} revision for {identifier}{requested}: {exc}"
        ) from exc
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not revision:
        raise BuildError(f"Hub returned no immutable revision for {identifier}")
    return revision


def _validate_local_revision(identifier: str, revision: str) -> None:
    if revision.startswith("local-file-sha256:"):
        path = pathlib.Path(identifier).expanduser()
        actual = f"local-file-sha256:{_sha256_file(path)}" if path.is_file() else None
    elif revision.startswith("local-tree-sha256:"):
        path = pathlib.Path(identifier).expanduser()
        actual = f"local-tree-sha256:{_sha256_tree(path)}" if path.is_dir() else None
    else:
        return
    if actual != revision:
        raise BuildError(
            f"local source changed since init: {identifier} "
            f"(actual={actual!r}, expected={revision!r})"
        )


def _build_contract(args: argparse.Namespace) -> dict[str, Any]:
    global FIXED_CONTRACT
    FIXED_CONTRACT = _smoke_contract() if args.smoke else copy.deepcopy(
        PRODUCTION_CONTRACT
    )
    workspace = pathlib.Path(args.work_dir).expanduser().resolve()
    artifact_dir = pathlib.Path(
        args.artifact_dir or workspace / "bridge_artifact"
    ).expanduser().resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_by": pathlib.Path(__file__).name,
        "fixed": FIXED_CONTRACT,
        "stage1_contract_sha256": stage1_contract_sha256(smoke=args.smoke),
        "source": _source_provenance(),
        "identity": {
            "student_model_id": args.student_model,
            "student_model_revision": _resolve_source_revision(
                args.student_model, "model"
            ),
            "teacher_model_id": args.teacher_model,
            "teacher_model_revision": _resolve_source_revision(
                args.teacher_model, "model"
            ),
            "tokenizer_id": args.tokenizer,
            "tokenizer_revision": _resolve_source_revision(args.tokenizer, "model"),
            "dataset_id": args.dataset,
            "dataset_revision": _resolve_source_revision(
                args.dataset, "dataset", args.dataset_revision
            ),
            "dataset_split": args.dataset_split,
            "prompt_key": args.prompt_key,
        },
        "artifact_dir": os.fspath(artifact_dir),
    }


def _read_contract(paths: Paths) -> dict[str, Any]:
    global FIXED_CONTRACT
    if not paths.contract.is_file():
        raise BuildError(f"workspace is not initialized; run init first: {paths.root}")
    contract = _load_json(paths.contract)
    if not isinstance(contract, dict):
        raise BuildError("contract.json must contain an object")
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise BuildError("unsupported contract schema version")
    fixed = contract.get("fixed")
    allowed_contracts = (PRODUCTION_CONTRACT, _smoke_contract())
    if fixed not in allowed_contracts:
        raise BuildError("contract.json does not match a supported Stage-1 contract")
    FIXED_CONTRACT = copy.deepcopy(fixed)
    is_smoke = fixed == _smoke_contract()
    expected_contract_digest = stage1_contract_sha256(smoke=is_smoke)
    if contract.get("stage1_contract_sha256") != expected_contract_digest:
        raise BuildError("contract.json has the wrong canonical Stage-1 digest")
    source = contract.get("source")
    if not isinstance(source, dict):
        raise BuildError("contract.json has no source provenance")
    current_source = _source_provenance()
    for field in ("git_commit", "tracked_dirty_diff_sha256", "source_files_sha256"):
        if source.get(field) != current_source.get(field):
            raise BuildError(
                f"bridge builder source changed since init ({field}); use the original "
                "source tree or start a new build"
            )
    identity = contract.get("identity")
    required = {
        "student_model_id",
        "student_model_revision",
        "teacher_model_id",
        "teacher_model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "dataset_id",
        "dataset_revision",
        "dataset_split",
        "prompt_key",
    }
    if not isinstance(identity, dict) or not required.issubset(identity):
        raise BuildError("contract.json has an invalid identity section")
    for identifier_field, revision_field in (
        ("student_model_id", "student_model_revision"),
        ("teacher_model_id", "teacher_model_revision"),
        ("tokenizer_id", "tokenizer_revision"),
        ("dataset_id", "dataset_revision"),
    ):
        _validate_local_revision(
            identity[identifier_field], identity[revision_field]
        )
    if not isinstance(contract.get("artifact_dir"), str):
        raise BuildError("contract.json has no artifact directory")
    return contract


def _contract_hash(paths: Paths) -> str:
    return _sha256_file(paths.contract)


def _layer_map() -> dict[int, int]:
    return resolve_layer_map(
        "all",
        "",
        FIXED_CONTRACT["student"]["num_layers"],
        FIXED_CONTRACT["teacher"]["num_layers"],
    )


def _group_layers(group_index: int) -> list[int]:
    if not 0 <= group_index < FIXED_CONTRACT["student_groups"]["count"]:
        raise BuildError("--group must be within 0..6")
    start = group_index * FIXED_CONTRACT["student_groups"]["layers_per_group"] + 1
    return list(range(start, start + 4))


def _dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return values[name]


def _load_config(model_id: str, revision: str) -> Any:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise BuildError("transformers is required for model phases") from exc
    kwargs = {"trust_remote_code": True}
    if not revision.startswith("local-"):
        kwargs["revision"] = revision
    return AutoConfig.from_pretrained(model_id, **kwargs)


def _config_shape(config: Any) -> tuple[int, int]:
    layers = getattr(config, "num_hidden_layers", None)
    width = getattr(config, "hidden_size", None)
    if not isinstance(layers, int) or not isinstance(width, int):
        raise BuildError("model config lacks integer num_hidden_layers/hidden_size")
    return layers, width


def _validate_model_shapes(contract: Mapping[str, Any]) -> None:
    identity = contract["identity"]
    student_config = _load_config(
        identity["student_model_id"], identity["student_model_revision"]
    )
    teacher_config = _load_config(
        identity["teacher_model_id"], identity["teacher_model_revision"]
    )
    got_student = _config_shape(student_config)
    got_teacher = _config_shape(teacher_config)
    expected_student = (
        FIXED_CONTRACT["student"]["num_layers"],
        FIXED_CONTRACT["student"]["hidden_size"],
    )
    expected_teacher = (
        FIXED_CONTRACT["teacher"]["num_layers"],
        FIXED_CONTRACT["teacher"]["hidden_size"],
    )
    if got_student != expected_student:
        raise BuildError(
            f"student architecture mismatch: got layers,width={got_student}, expected {expected_student}"
        )
    if got_teacher != expected_teacher:
        raise BuildError(
            f"teacher architecture mismatch: got layers,width={got_teacher}, expected {expected_teacher}"
        )


def _validate_tokenizer_compatibility(contract: Mapping[str, Any]) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise BuildError("transformers is required to validate tokenizers") from exc
    identity = contract["identity"]

    def load(identifier: str, revision: str) -> Any:
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if not revision.startswith("local-"):
            kwargs["revision"] = revision
        return AutoTokenizer.from_pretrained(identifier, **kwargs)

    tokenizer = load(identity["tokenizer_id"], identity["tokenizer_revision"])
    teacher_tokenizer = load(
        identity["teacher_model_id"], identity["teacher_model_revision"]
    )
    if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise BuildError(
            "teacher and configured bridge tokenizers have different vocabularies"
        )
    _format_prompt_ids(tokenizer, "OPRD bridge contract validation")
    _format_prompt_text(tokenizer, "OPRD bridge contract validation")


def _vllm_identity_kwargs(contract: Mapping[str, Any]) -> dict[str, Any]:
    identity = contract["identity"]
    kwargs: dict[str, Any] = {
        "model": identity["student_model_id"],
        "tokenizer": identity["tokenizer_id"],
    }
    student_revision = identity["student_model_revision"]
    tokenizer_revision = identity["tokenizer_revision"]
    if not student_revision.startswith("local-"):
        kwargs["revision"] = student_revision
    if not tokenizer_revision.startswith("local-"):
        kwargs["tokenizer_revision"] = tokenizer_revision
    return kwargs


def _load_model(
    model_id: str,
    revision: str,
    device: str,
    dtype_name: str,
) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise BuildError("transformers is required for model phases") from exc
    kwargs: dict[str, Any] = {
        "torch_dtype": _dtype(dtype_name),
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    if not revision.startswith("local-"):
        kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    model.requires_grad_(False)
    return model.to(torch.device(device))


def _backbone_hidden(model: Any, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
    backbone = getattr(model, "model", model)
    output = backbone(
        input_ids=input_ids,
        use_cache=False,
        output_hidden_states=True,
        return_dict=True,
    )
    hidden_states = output.hidden_states
    if hidden_states is None:
        raise BuildError("model forward did not return hidden states")
    return tuple(hidden_states)


def _full_ids(record: Mapping[str, Any]) -> list[int]:
    prompt = record.get("prompt_token_ids")
    output = record.get("output_token_ids")
    if not isinstance(prompt, list) or not isinstance(output, list):
        raise BuildError("rollout record lacks token ID lists")
    if any(not isinstance(token, int) for token in prompt + output):
        raise BuildError("rollout token IDs must be integers")
    return prompt + output


def _response_window(record: Mapping[str, Any]) -> tuple[int, int]:
    prompt_len = len(record["prompt_token_ids"])
    response_len = len(record["output_token_ids"])
    keep = min(response_len, FIXED_CONTRACT["response_positions"]["k"])
    return prompt_len + response_len - keep, keep


def _require_rollouts(paths: Paths) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = _load_jsonl(paths.rollouts)
    split = _load_json(paths.split)
    if not paths.corpus_metadata.is_file():
        raise BuildError("corpus metadata is missing")
    corpus_metadata = _load_json(paths.corpus_metadata)
    if corpus_metadata.get("prompts_commit_sha256") != _sha256_file(
        paths.prompts_commit
    ):
        raise BuildError("corpus metadata prompt commitment checksum does not match")
    if corpus_metadata.get("attempt_journal_sha256") != _sha256_file(
        paths.attempt_journal
    ):
        raise BuildError("corpus metadata attempt journal checksum does not match")
    if corpus_metadata.get("rollouts_sha256") != _sha256_file(paths.rollouts):
        raise BuildError("corpus metadata rollout checksum does not match")
    if corpus_metadata.get("split_sha256") != _sha256_file(paths.split):
        raise BuildError("corpus metadata split checksum does not match")
    attempt_journal = _load_json(paths.attempt_journal)
    journal_entries = (
        attempt_journal.get("entries") if isinstance(attempt_journal, dict) else None
    )
    if not isinstance(journal_entries, dict):
        raise BuildError("attempt journal is invalid")
    expected = FIXED_CONTRACT["attempted_prompts"]
    if len(records) != expected:
        raise BuildError(f"expected {expected} rollout records, found {len(records)}")
    if not isinstance(split, dict) or not isinstance(split.get("validation_indices"), list):
        raise BuildError("split.json is invalid")
    for index, record in enumerate(records):
        if record.get("attempt_index") != index:
            raise BuildError(f"rollout order mismatch at index {index}")
        attempt_path = _attempt_path(paths, index)
        journal_entry = journal_entries.get(str(index))
        if (
            not attempt_path.is_file()
            or not isinstance(journal_entry, dict)
            or journal_entry.get("record_sha256") != _sha256_file(attempt_path)
            or _load_json(attempt_path) != record
        ):
            raise BuildError(f"rollout {index} does not match its committed attempt")
        _full_ids(record)
        if not isinstance(record.get("source_row"), dict):
            raise BuildError(f"rollout {index} lacks its complete source dataset row")
        if record.get("completion_length") != len(record["output_token_ids"]):
            raise BuildError(f"rollout {index} has inconsistent completion length")
        if record.get("request_finished") is not True:
            raise BuildError(f"rollout {index} did not finish in vLLM")
        if len(record["output_token_ids"]) > FIXED_CONTRACT["generation"]["max_new_tokens"]:
            raise BuildError(f"rollout {index} exceeds the 4096-token generation cap")
        if not isinstance(record.get("formatted_prompt"), str):
            raise BuildError(f"rollout {index} lacks its exact formatted prompt")
        if record.get("generation") != FIXED_CONTRACT["generation"]:
            raise BuildError(f"rollout {index} has different generation parameters")
        expected_cap_hit = (
            len(record["output_token_ids"])
            >= FIXED_CONTRACT["generation"]["max_new_tokens"]
        )
        if record.get("hit_token_cap") is not expected_cap_hit:
            raise BuildError(f"rollout {index} has inconsistent cap metadata")
        if "finish_reason" not in record or "stop_reason" not in record:
            raise BuildError(f"rollout {index} lacks finish metadata")
    return records, split


def _split_label(index: int, split: Mapping[str, Any]) -> str:
    validation = set(split["validation_indices"])
    return "validation" if index in validation else "train"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def _runtime_versions() -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "transformers": _package_version("transformers"),
        "datasets": _package_version("datasets"),
        "vllm": _package_version("vllm"),
        "safetensors": _package_version("safetensors"),
        "numpy": np.__version__,
    }


def _git_output(arguments: Sequence[str]) -> bytes | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=pathlib.Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _source_provenance() -> dict[str, Any]:
    revision = _git_output(["rev-parse", "HEAD"])
    diff = _git_output(["diff", "--binary", "HEAD", "--"])
    status = _git_output(["status", "--porcelain=v1", "--untracked-files=normal"])
    source_files = [
        pathlib.Path(__file__).resolve(),
        pathlib.Path(__file__).resolve().with_name("oprd_bridge.py"),
        pathlib.Path(__file__).resolve().with_name("train_harness.py"),
    ]
    return {
        "git_commit": revision.decode().strip() if revision else None,
        "tracked_dirty_diff_sha256": hashlib.sha256(diff).hexdigest() if diff is not None else None,
        "git_status": status.decode(errors="replace").splitlines() if status else [],
        "source_files_sha256": {
            path.name: _sha256_file(path) for path in source_files if path.is_file()
        },
    }


# ---------------------------------------------------------------------------
# init and generation


def command_init(args: argparse.Namespace) -> None:
    paths = Paths(pathlib.Path(args.work_dir).expanduser().resolve())
    with _workspace_lock(paths):
        if paths.contract.exists():
            contract = _read_contract(paths)
            requested_artifact = pathlib.Path(
                args.artifact_dir or paths.root / "bridge_artifact"
            ).expanduser().resolve()
            requested = {
                "student_model_id": args.student_model,
                "teacher_model_id": args.teacher_model,
                "tokenizer_id": args.tokenizer,
                "dataset_id": args.dataset,
                "dataset_split": args.dataset_split,
                "prompt_key": args.prompt_key,
            }
            requested["dataset_revision"] = _resolve_source_revision(
                args.dataset, "dataset", args.dataset_revision
            )
            existing_identity = contract["identity"]
            mismatches = {
                key: (existing_identity.get(key), value)
                for key, value in requested.items()
                if existing_identity.get(key) != value
            }
            if pathlib.Path(contract["artifact_dir"]) != requested_artifact:
                mismatches["artifact_dir"] = (
                    contract["artifact_dir"],
                    os.fspath(requested_artifact),
                )
            existing_smoke = contract["fixed"]["contract_name"].endswith("-smoke")
            if existing_smoke != args.smoke:
                mismatches["smoke"] = (existing_smoke, args.smoke)
            if mismatches:
                raise BuildError(
                    f"work directory already has a different contract: {mismatches}"
                )
        else:
            contract = _build_contract(args)
            _validate_model_shapes(contract)
            _validate_tokenizer_compatibility(contract)
            _atomic_write_json(paths.contract, contract)
        input_fingerprint = _fingerprint(contract)
        with _phase(paths, "init", input_fingerprint) as (state, should_run):
            if not should_run:
                return
            for directory in (paths.corpus, paths.teacher, paths.groups):
                directory.mkdir(parents=True, exist_ok=True)
            _finish_phase(
                paths,
                state,
                "init",
                [paths.contract],
                {
                    "contract_sha256": _sha256_file(paths.contract),
                    "artifact_dir": contract["artifact_dir"],
                },
            )
    print(f"initialized fixed OPRD bridge workspace: {paths.root}")


def _load_dataset_frame(
    identity: Mapping[str, str],
) -> tuple[Any, dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise BuildError("datasets is required for generate-corpus") from exc
    dataset_id = identity["dataset_id"]
    split = identity["dataset_split"]
    path = pathlib.Path(dataset_id).expanduser()
    if path.suffix == ".parquet" and path.is_file():
        dataset = load_dataset("parquet", data_files=os.fspath(path), split=split)
    elif path.suffix in {".json", ".jsonl"} and path.is_file():
        dataset = load_dataset("json", data_files=os.fspath(path), split=split)
    else:
        dataset = load_dataset(
            dataset_id,
            split=split,
            revision=identity["dataset_revision"],
        )
    metadata = {
        "dataset_id": dataset_id,
        "dataset_revision": identity["dataset_revision"],
        "dataset_split": split,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "rows": len(dataset),
        "features": _jsonable(getattr(dataset, "features", None)),
    }
    return dataset.to_pandas(), metadata


def _sample_prompts(
    contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    identity = contract["identity"]
    frame, dataset_metadata = _load_dataset_frame(identity)
    count = FIXED_CONTRACT["attempted_prompts"]
    if len(frame) < count:
        raise BuildError(f"DAPO dataset has {len(frame)} rows, need exactly {count}")
    prompt_key = identity["prompt_key"]
    if prompt_key not in frame.columns:
        raise BuildError(
            f"prompt key {prompt_key!r} not found; columns={list(frame.columns)}"
        )
    # Match the public bridge analysis script: pandas sample(random_state=42).
    sampled = frame.sample(n=count, random_state=FIXED_CONTRACT["prompt_seed"])
    rows: list[dict[str, Any]] = []
    for attempt_index, (dataset_index, row) in enumerate(sampled.iterrows()):
        rows.append(
            {
                "attempt_index": attempt_index,
                "dataset_index": _jsonable(dataset_index),
                "source_row": _jsonable(row.to_dict()),
                "raw_prompt": _jsonable(row[prompt_key]),
            }
        )
    dataset_metadata["selected_dataset_indices_sha256"] = _fingerprint(
        [row["dataset_index"] for row in rows]
    )
    return rows, dataset_metadata


def _load_or_commit_prompt_selection(
    paths: Paths, contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_prompts, expected_metadata = _sample_prompts(contract)
    if paths.prompts.exists():
        prompts = _load_jsonl(paths.prompts)
        if prompts != expected_prompts:
            raise BuildError(
                "existing prompts.jsonl does not match deterministic sampling from the pinned dataset"
            )
    else:
        prompts = expected_prompts
        _atomic_write_jsonl(paths.prompts, prompts)

    if paths.corpus_metadata.exists():
        corpus_metadata = _load_json(paths.corpus_metadata)
        if not isinstance(corpus_metadata, dict):
            raise BuildError("existing corpus metadata is not a JSON object")
        mismatches = [
            key
            for key, value in expected_metadata.items()
            if corpus_metadata.get(key) != value
        ]
        if mismatches:
            raise BuildError(
                "existing corpus metadata does not match deterministic sampling: "
                + ", ".join(sorted(mismatches))
            )
    else:
        corpus_metadata = dict(expected_metadata)
        _atomic_write_json(paths.corpus_metadata, corpus_metadata)
    return prompts, corpus_metadata


def _prompt_messages(raw_prompt: Any) -> list[dict[str, str]]:
    if isinstance(raw_prompt, str):
        return [{"role": "user", "content": raw_prompt}]
    if isinstance(raw_prompt, list):
        messages: list[dict[str, str]] = []
        for value in raw_prompt:
            if not isinstance(value, Mapping):
                raise BuildError("chat-format DAPO prompts must contain message objects")
            role = value.get("role")
            content = value.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise BuildError("chat-format DAPO messages require string role/content")
            messages.append({"role": role, "content": content})
        if not messages:
            raise BuildError("DAPO prompt has an empty message list")
        return messages
    raise BuildError(f"unsupported DAPO prompt type: {type(raw_prompt).__name__}")


def _format_prompt_ids(tokenizer: Any, raw_prompt: Any) -> list[int]:
    messages = _prompt_messages(raw_prompt)
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": True,
        "enable_thinking": False,
    }
    try:
        encoded = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        raise BuildError(
            "tokenizer must support apply_chat_template(enable_thinking=False)"
        ) from exc
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    ids = [int(token) for token in encoded]
    if not ids:
        raise BuildError("public chat template produced an empty prompt")
    return ids


def _format_prompt_text(tokenizer: Any, raw_prompt: Any) -> str:
    messages = _prompt_messages(raw_prompt)
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": False,
    }
    try:
        formatted = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError as exc:
        raise BuildError(
            "tokenizer must support apply_chat_template(enable_thinking=False)"
        ) from exc
    if not isinstance(formatted, str) or not formatted:
        raise BuildError("public chat template produced an empty formatted prompt")
    return formatted


def _attempt_provenance(
    contract: Mapping[str, Any],
    prompt_row: Mapping[str, Any],
    formatted_prompt: str,
    prompt_token_ids: Sequence[int],
) -> dict[str, Any]:
    index = int(prompt_row["attempt_index"])
    payload = {
        "attempt_index": index,
        "dataset_index": prompt_row["dataset_index"],
        "source_row": prompt_row["source_row"],
        "raw_prompt": prompt_row["raw_prompt"],
        "formatted_prompt": formatted_prompt,
        "prompt_token_ids": [int(token) for token in prompt_token_ids],
        "generation_seed": FIXED_CONTRACT["prompt_seed"] * 1_000_000 + index,
        "student_model_id": contract["identity"]["student_model_id"],
        "student_model_revision": contract["identity"]["student_model_revision"],
        "tokenizer_id": contract["identity"]["tokenizer_id"],
        "tokenizer_revision": contract["identity"]["tokenizer_revision"],
        "generation": FIXED_CONTRACT["generation"],
        "formatting": FIXED_CONTRACT["formatting"],
    }
    payload["sha256"] = _fingerprint(payload)
    return payload


def _attempt_path(paths: Paths, index: int) -> pathlib.Path:
    return paths.attempts / f"{index:06d}.json"


def _valid_attempt_record(
    path: pathlib.Path,
    expected_provenance: Mapping[str, Any],
    journal_entry: Mapping[str, Any] | None,
) -> bool:
    if not path.is_file() or not isinstance(journal_entry, Mapping):
        return False
    try:
        record = _load_json(path)
        expected_index = int(expected_provenance["attempt_index"])
        return (
            isinstance(record, dict)
            and journal_entry.get("attempt_index") == expected_index
            and journal_entry.get("record_sha256") == _sha256_file(path)
            and journal_entry.get("attempt_provenance_sha256")
            == expected_provenance["sha256"]
            and record.get("attempt_index") == expected_index
            and record.get("dataset_index") == expected_provenance["dataset_index"]
            and record.get("source_row") == expected_provenance["source_row"]
            and record.get("raw_prompt") == expected_provenance["raw_prompt"]
            and record.get("formatted_prompt")
            == expected_provenance["formatted_prompt"]
            and record.get("prompt_token_ids")
            == expected_provenance["prompt_token_ids"]
            and record.get("generation_seed")
            == expected_provenance["generation_seed"]
            and record.get("attempt_provenance_sha256")
            == expected_provenance["sha256"]
            and isinstance(record.get("output_token_ids"), list)
            and record.get("completion_length") == len(record["output_token_ids"])
            and record.get("generation") == FIXED_CONTRACT["generation"]
            and isinstance(record.get("hit_token_cap"), bool)
            and record.get("request_finished") is True
            and len(record["output_token_ids"])
            <= FIXED_CONTRACT["generation"]["max_new_tokens"]
            and "finish_reason" in record
            and "stop_reason" in record
        )
    except (BuildError, KeyError, TypeError, ValueError):
        return False


def _build_split() -> dict[str, Any]:
    count = FIXED_CONTRACT["attempted_prompts"]
    validation_count = FIXED_CONTRACT["validation_rollouts"]
    indices = np.arange(count)
    rng = np.random.default_rng(FIXED_CONTRACT["prompt_seed"])
    rng.shuffle(indices)
    validation = sorted(int(value) for value in indices[:validation_count])
    validation_set = set(validation)
    train = [index for index in range(count) if index not in validation_set]
    return {
        "seed": FIXED_CONTRACT["prompt_seed"],
        "method": (
            "numpy default_rng shuffle; first "
            f"{validation_count} validation"
        ),
        "train_indices": train,
        "validation_indices": validation,
    }


def command_generate_corpus(args: argparse.Namespace) -> None:
    paths = Paths(pathlib.Path(args.work_dir).expanduser().resolve())
    with _workspace_lock(paths):
        contract = _read_contract(paths)
        input_fingerprint = _fingerprint(
            {
                "contract": _contract_hash(paths),
                "phase": "generate-corpus",
                "runtime_backend": "vllm",
                "batch_size": args.batch_size,
                "tensor_parallel_size": args.tensor_parallel_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len,
                "runtime": _runtime_versions(),
            }
        )
        with _phase(paths, "generate-corpus", input_fingerprint) as (state, should_run):
            if not should_run:
                return
            prompts, corpus_metadata = _load_or_commit_prompt_selection(
                paths, contract
            )

            prompt_commit = {
                "contract_sha256": _contract_hash(paths),
                "prompts_sha256": _sha256_file(paths.prompts),
                "selected_dataset_indices_sha256": corpus_metadata.get(
                    "selected_dataset_indices_sha256"
                ),
                "dataset_revision": contract["identity"]["dataset_revision"],
                "dataset_fingerprint": corpus_metadata.get("dataset_fingerprint"),
                "attempted_prompts": FIXED_CONTRACT["attempted_prompts"],
            }
            if paths.prompts_commit.exists():
                if _load_json(paths.prompts_commit) != prompt_commit:
                    raise BuildError(
                        "selected prompt commitment changed during corpus resume"
                    )
            else:
                _atomic_write_json(paths.prompts_commit, prompt_commit)
            prompt_commit_sha256 = _sha256_file(paths.prompts_commit)
            commitments = state["phases"]["generate-corpus"]["commitments"]
            previous_prompt_commit = commitments.get("prompts_commit_sha256")
            if (
                previous_prompt_commit is not None
                and previous_prompt_commit != prompt_commit_sha256
            ):
                raise BuildError("state.json prompt commitment changed during resume")
            commitments["prompts_commit_sha256"] = prompt_commit_sha256
            _write_state(paths, state)

            try:
                from transformers import AutoTokenizer
                from vllm import LLM, SamplingParams
                try:
                    from vllm import TokensPrompt
                except ImportError:
                    from vllm.inputs import TokensPrompt
            except ImportError as exc:
                raise BuildError("generate-corpus requires transformers and vLLM") from exc

            tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": True}
            tokenizer_revision = contract["identity"]["tokenizer_revision"]
            if not tokenizer_revision.startswith("local-"):
                tokenizer_kwargs["revision"] = tokenizer_revision
            tokenizer = AutoTokenizer.from_pretrained(
                contract["identity"]["tokenizer_id"], **tokenizer_kwargs
            )
            journal_header = {
                "schema_version": SCHEMA_VERSION,
                "input_fingerprint": input_fingerprint,
                "prompts_commit_sha256": prompt_commit_sha256,
            }
            if paths.attempt_journal.exists():
                attempt_journal = _load_json(paths.attempt_journal)
                if not isinstance(attempt_journal, dict) or any(
                    attempt_journal.get(key) != value
                    for key, value in journal_header.items()
                ):
                    raise BuildError("attempt journal is incompatible with this corpus build")
                if not isinstance(attempt_journal.get("entries"), dict):
                    raise BuildError("attempt journal has no valid entries mapping")
            else:
                attempt_journal = {**journal_header, "entries": {}}
                _atomic_write_json(paths.attempt_journal, attempt_journal)
            prepared: list[dict[str, Any]] = []
            missing: list[dict[str, Any]] = []
            for row in prompts:
                index = int(row["attempt_index"])
                prompt_ids = _format_prompt_ids(tokenizer, row["raw_prompt"])
                formatted_prompt = _format_prompt_text(tokenizer, row["raw_prompt"])
                provenance = _attempt_provenance(
                    contract, row, formatted_prompt, prompt_ids
                )
                item = {
                    "row": row,
                    "prompt_ids": prompt_ids,
                    "formatted_prompt": formatted_prompt,
                    "provenance": provenance,
                }
                prepared.append(item)
                path = _attempt_path(paths, index)
                journal_entry = attempt_journal["entries"].get(str(index))
                if _valid_attempt_record(path, provenance, journal_entry):
                    continue
                if journal_entry is not None:
                    raise BuildError(
                        f"committed attempt record failed validation: {path}"
                    )
                if path.exists():
                    path.unlink()
                missing.append(item)
            if missing:
                llm_kwargs: dict[str, Any] = {
                    **_vllm_identity_kwargs(contract),
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "trust_remote_code": True,
                    "gpu_memory_utilization": args.gpu_memory_utilization,
                    "seed": FIXED_CONTRACT["prompt_seed"],
                }
                if args.max_model_len > 0:
                    llm_kwargs["max_model_len"] = args.max_model_len
                llm = LLM(**llm_kwargs)
                paths.attempts.mkdir(parents=True, exist_ok=True)
                for start in range(0, len(missing), args.batch_size):
                    chunk = missing[start : start + args.batch_size]
                    prompt_ids = [item["prompt_ids"] for item in chunk]
                    token_prompts = [
                        TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids
                    ]
                    # Deliberately omit top_p, top_k, min_p, and logits processors.
                    params = [
                        SamplingParams(
                            temperature=FIXED_CONTRACT["generation"]["temperature"],
                            max_tokens=FIXED_CONTRACT["generation"]["max_new_tokens"],
                            n=FIXED_CONTRACT["generation"]["n"],
                            seed=item["provenance"]["generation_seed"],
                        )
                        for item in chunk
                    ]
                    outputs = llm.generate(token_prompts, params)
                    if len(outputs) != len(chunk):
                        raise BuildError("vLLM returned a different number of request outputs")
                    for item, supplied_prompt_ids, request_output in zip(
                        chunk, prompt_ids, outputs
                    ):
                        row = item["row"]
                        provenance = item["provenance"]
                        formatted_prompt = item["formatted_prompt"]
                        if len(request_output.outputs) != 1:
                            raise BuildError("vLLM did not return exactly one output")
                        if not bool(getattr(request_output, "finished", False)):
                            raise BuildError(
                                f"vLLM request did not finish for attempt "
                                f"{row['attempt_index']}"
                            )
                        candidate = request_output.outputs[0]
                        actual_prompt_ids = [
                            int(token)
                            for token in (
                                getattr(
                                    request_output,
                                    "prompt_token_ids",
                                    supplied_prompt_ids,
                                )
                                or supplied_prompt_ids
                            )
                        ]
                        if actual_prompt_ids != supplied_prompt_ids:
                            raise BuildError(
                                f"vLLM changed prompt token IDs for attempt "
                                f"{row['attempt_index']}"
                            )
                        output_ids = [int(token) for token in candidate.token_ids]
                        record = {
                            "attempt_index": int(row["attempt_index"]),
                            "dataset_index": row["dataset_index"],
                            "source_row": row["source_row"],
                            "raw_prompt": row["raw_prompt"],
                            "formatted_prompt": formatted_prompt,
                            "prompt_token_ids": [int(token) for token in actual_prompt_ids],
                            "output_token_ids": output_ids,
                            "completion_length": len(output_ids),
                            "response_text": candidate.text,
                            "finish_reason": _jsonable(
                                getattr(candidate, "finish_reason", None)
                            ),
                            "stop_reason": _jsonable(
                                getattr(candidate, "stop_reason", None)
                            ),
                            "request_finished": True,
                            "request_id": str(
                                getattr(request_output, "request_id", row["attempt_index"])
                            ),
                            "generation_seed": provenance["generation_seed"],
                            "attempt_provenance_sha256": provenance["sha256"],
                            "hit_token_cap": (
                                len(output_ids)
                                >= FIXED_CONTRACT["generation"]["max_new_tokens"]
                            ),
                            "generation": FIXED_CONTRACT["generation"],
                        }
                        attempt_path = _attempt_path(paths, record["attempt_index"])
                        _atomic_write_json(attempt_path, record)
                        attempt_journal["entries"][str(record["attempt_index"])] = {
                            "attempt_index": record["attempt_index"],
                            "attempt_provenance_sha256": provenance["sha256"],
                            "record_sha256": _sha256_file(attempt_path),
                        }
                        _atomic_write_json(paths.attempt_journal, attempt_journal)
                    done = min(start + len(chunk), len(missing))
                    print(
                        f"generate-corpus: materialized {done}/{len(missing)} missing rollouts",
                        flush=True,
                    )
                del llm

            records: list[dict[str, Any]] = []
            for item in prepared:
                index = int(item["row"]["attempt_index"])
                attempt_path = _attempt_path(paths, index)
                journal_entry = attempt_journal["entries"].get(str(index))
                if not _valid_attempt_record(
                    attempt_path, item["provenance"], journal_entry
                ):
                    raise BuildError(
                        f"attempt record failed final provenance validation: {attempt_path}"
                    )
                records.append(_load_json(attempt_path))
            _atomic_write_jsonl(paths.rollouts, records)
            split = _build_split()
            _atomic_write_json(paths.split, split)
            if len(split["train_indices"]) != FIXED_CONTRACT["train_rollouts"]:
                raise AssertionError("fixed train split size changed")
            if len(split["validation_indices"]) != FIXED_CONTRACT["validation_rollouts"]:
                raise AssertionError("fixed validation split size changed")
            finish_histogram: dict[str, int] = {}
            for record in records:
                key = str(record.get("finish_reason"))
                finish_histogram[key] = finish_histogram.get(key, 0) + 1
            corpus_metadata.update(
                {
                    "student_model_id": contract["identity"]["student_model_id"],
                    "student_model_revision": contract["identity"][
                        "student_model_revision"
                    ],
                    "tokenizer_id": contract["identity"]["tokenizer_id"],
                    "tokenizer_revision": tokenizer_revision,
                    "generation": FIXED_CONTRACT["generation"],
                    "formatting": FIXED_CONTRACT["formatting"],
                    "runtime": _runtime_versions(),
                    "source": _source_provenance(),
                    "finish_histogram": finish_histogram,
                    "empty_responses": sum(
                        not record["output_token_ids"] for record in records
                    ),
                    "cap_hits": sum(record["hit_token_cap"] for record in records),
                    "prompts_sha256": _sha256_file(paths.prompts),
                    "prompts_commit_sha256": _sha256_file(paths.prompts_commit),
                    "attempt_journal_sha256": _sha256_file(paths.attempt_journal),
                    "rollouts_sha256": _sha256_file(paths.rollouts),
                    "split_sha256": _sha256_file(paths.split),
                }
            )
            _atomic_write_json(paths.corpus_metadata, corpus_metadata)
            _finish_phase(
                paths,
                state,
                "generate-corpus",
                [
                    paths.prompts,
                    paths.prompts_commit,
                    paths.attempt_journal,
                    paths.rollouts,
                    paths.split,
                    paths.corpus_metadata,
                ],
                {
                    "attempted_prompts": len(records),
                    "train_rollouts": len(split["train_indices"]),
                    "validation_rollouts": len(split["validation_indices"]),
                    "sampling": FIXED_CONTRACT["generation"],
                    "formatting": FIXED_CONTRACT["formatting"],
                },
            )
    print(
        f"generated and split exactly {FIXED_CONTRACT['attempted_prompts']:,} "
        f"rollout attempts: {paths.rollouts}"
    )


# ---------------------------------------------------------------------------
# Streamed teacher PCA


def _reservoir_updates(
    heap: list[tuple[float, int]],
    rng: random.Random,
    seen: int,
    row_count: int,
    capacity: int,
) -> tuple[dict[int, int], int]:
    """Return final slot-to-local-row writes for priority reservoir sampling."""
    updates: dict[int, int] = {}
    for local_row in range(row_count):
        priority = rng.random()
        if seen < capacity:
            slot = seen
            heapq.heappush(heap, (-priority, slot))
            updates[slot] = local_row
        elif priority < -heap[0][0]:
            _, slot = heapq.heapreplace(heap, (-priority, heap[0][1]))
            updates[slot] = local_row
        seen += 1
    return updates, seen


def _fit_covariance_pca(
    matrix: np.memmap,
    rows: int,
    rank: int,
    pca_device: str,
    chunk_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    if rows < 2:
        raise BuildError("teacher PCA needs at least two sampled token rows")
    width = matrix.shape[1]
    total = np.zeros(width, dtype=np.float64)
    for start in range(0, rows, chunk_rows):
        total += np.asarray(matrix[start : start + chunk_rows], dtype=np.float32).sum(
            axis=0, dtype=np.float64
        )
    mean = (total / rows).astype(np.float32)
    covariance = np.zeros((width, width), dtype=np.float32)
    for start in range(0, rows, chunk_rows):
        block = np.asarray(matrix[start : start + chunk_rows], dtype=np.float32)
        block -= mean
        covariance += block.T @ block
    covariance /= max(rows - 1, 1)
    cov_tensor = torch.from_numpy(covariance).to(torch.device(pca_device))
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_tensor)
    top_values = eigenvalues[-rank:].flip(0).float().cpu()
    components = eigenvectors[:, -rank:].flip(1).T.contiguous().float().cpu()
    return torch.from_numpy(mean.copy()), components, top_values.tolist()


def command_fit_teacher_pca(args: argparse.Namespace) -> None:
    paths = Paths(pathlib.Path(args.work_dir).expanduser().resolve())
    with _workspace_lock(paths):
        contract = _read_contract(paths)
        records, split = _require_rollouts(paths)
        input_fingerprint = _fingerprint(
            {
                "contract": _contract_hash(paths),
                "rollouts": _sha256_file(paths.rollouts),
                "split": _sha256_file(paths.split),
                "phase": "fit-teacher-pca",
                "device": args.device,
                "dtype": args.dtype,
                "pca_device": args.pca_device,
                "covariance_chunk_rows": args.covariance_chunk_rows,
                "runtime": _runtime_versions(),
            }
        )
        with _phase(paths, "fit-teacher-pca", input_fingerprint) as (
            state,
            should_run,
        ):
            if not should_run:
                return
            _validate_model_shapes(contract)
            layer_map = _layer_map()
            rank = FIXED_CONTRACT["rank"]
            capacity = FIXED_CONTRACT["teacher_pca"]["max_rows"]
            teacher_width = FIXED_CONTRACT["teacher"]["hidden_size"]
            work = paths.teacher / ".pca-work"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            memmaps: dict[int, np.memmap] = {
                student_layer: np.memmap(
                    work / f"layer_{student_layer:02d}.f16",
                    mode="w+",
                    dtype=np.float16,
                    shape=(capacity, teacher_width),
                )
                for student_layer in layer_map
            }
            model = _load_model(
                contract["identity"]["teacher_model_id"],
                contract["identity"]["teacher_model_revision"],
                args.device,
                args.dtype,
            )
            device = torch.device(args.device)
            heap: list[tuple[float, int]] = []
            sampled_row_ids: list[list[int] | None] = [None] * capacity
            rng = random.Random(FIXED_CONTRACT["prompt_seed"])
            seen = 0
            train_indices = split["train_indices"]
            try:
                for position, index in enumerate(train_indices, 1):
                    record = records[index]
                    response_start, row_count = _response_window(record)
                    updates, seen = _reservoir_updates(
                        heap, rng, seen, row_count, capacity
                    )
                    response_offset_start = len(record["output_token_ids"]) - row_count
                    for slot, local_row in updates.items():
                        sampled_row_ids[slot] = [
                            int(record["attempt_index"]),
                            response_offset_start + local_row,
                        ]
                    if updates:
                        ids = torch.tensor(
                            [_full_ids(record)], dtype=torch.long, device=device
                        )
                        with torch.inference_mode():
                            hidden_states = _backbone_hidden(model, ids)
                        slots = np.fromiter(updates.keys(), dtype=np.int64)
                        local_rows = torch.tensor(
                            list(updates.values()), dtype=torch.long, device=device
                        )
                        absolute_rows = local_rows + response_start
                        for student_layer, teacher_layer in layer_map.items():
                            selected = hidden_states[teacher_layer][
                                0
                            ].index_select(0, absolute_rows)
                            memmaps[student_layer][slots] = (
                                selected.float().cpu().numpy().astype(np.float16)
                            )
                        del selected, hidden_states, ids
                    if position % 25 == 0 or position == len(train_indices):
                        print(
                            f"fit-teacher-pca: streamed {position}/{len(train_indices)} train rollouts; "
                            f"seen_rows={seen:,}",
                            flush=True,
                        )
                sampled_rows = min(seen, capacity)
                final_row_ids = sampled_row_ids[:sampled_rows]
                if any(value is None for value in final_row_ids):
                    raise BuildError("teacher PCA reservoir has unassigned sampled-row slots")
                for matrix in memmaps.values():
                    matrix.flush()
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                tensors: dict[str, torch.Tensor] = {}
                spectra: dict[str, list[float]] = {}
                for ordinal, student_layer in enumerate(layer_map, 1):
                    mean, components, eigenvalues = _fit_covariance_pca(
                        memmaps[student_layer],
                        sampled_rows,
                        rank,
                        args.pca_device,
                        args.covariance_chunk_rows,
                    )
                    tensors[f"mean.{student_layer}"] = mean
                    tensors[f"components.{student_layer}"] = components
                    spectra[str(student_layer)] = eigenvalues
                    print(
                        f"fit-teacher-pca: solved layer {ordinal}/{len(layer_map)}",
                        flush=True,
                    )
                _atomic_save_safetensors(
                    paths.pca_tensors,
                    tensors,
                    metadata={
                        "rank": str(rank),
                        "sampled_rows": str(sampled_rows),
                        "seen_rows": str(seen),
                    },
                )
                metadata = {
                    "rank": rank,
                    "layer_map": {str(key): value for key, value in layer_map.items()},
                    "train_rollouts": len(train_indices),
                    "seen_token_rows": seen,
                    "sampled_rows": sampled_rows,
                    "max_rows": capacity,
                    "seed": FIXED_CONTRACT["prompt_seed"],
                    "sampling": FIXED_CONTRACT["teacher_pca"]["sampling"],
                    "sampled_row_ids": final_row_ids,
                    "sampled_row_ids_sha256": _fingerprint(final_row_ids),
                    "teacher_model_id": contract["identity"]["teacher_model_id"],
                    "teacher_model_revision": contract["identity"][
                        "teacher_model_revision"
                    ],
                    "model_dtype": args.dtype,
                    "runtime": _runtime_versions(),
                    "top_eigenvalues": spectra,
                }
                _atomic_write_json(paths.pca_metadata, metadata)
            finally:
                for matrix in memmaps.values():
                    try:
                        matrix.flush()
                    except Exception:
                        pass
                shutil.rmtree(work, ignore_errors=True)
            _finish_phase(
                paths,
                state,
                "fit-teacher-pca",
                [paths.pca_tensors, paths.pca_metadata],
                metadata,
            )
    print(f"fit streamed rank-8 teacher PCA for all layers: {paths.pca_tensors}")


# ---------------------------------------------------------------------------
# Compact teacher targets


def _load_pca(paths: Paths) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    load_file, _, _ = _safetensors_api()
    tensors = load_file(os.fspath(paths.pca_tensors), device="cpu")
    _require_finite_tensors(tensors, context=f"teacher PCA {paths.pca_tensors}")
    metadata = _load_json(paths.pca_metadata)
    expected_layers = set(_layer_map())
    for layer in expected_layers:
        mean = tensors.get(f"mean.{layer}")
        components = tensors.get(f"components.{layer}")
        if mean is None or tuple(mean.shape) != (
            FIXED_CONTRACT["teacher"]["hidden_size"],
        ):
            raise BuildError(f"teacher PCA mean is missing or invalid for layer {layer}")
        if components is None or tuple(components.shape) != (
            FIXED_CONTRACT["rank"],
            FIXED_CONTRACT["teacher"]["hidden_size"],
        ):
            raise BuildError(
                f"teacher PCA components are missing or invalid for layer {layer}"
            )
    return tensors, metadata


def _target_path(paths: Paths, index: int) -> pathlib.Path:
    return paths.targets / f"{index:06d}.safetensors"


def _cache_path(paths: Paths, group_index: int, index: int) -> pathlib.Path:
    return paths.cache(group_index) / f"{index:06d}.safetensors"


def _valid_tensor_shard(
    path: pathlib.Path,
    tensor_name: str,
    expected_shape_tail: tuple[int, ...],
    expected_rows: int,
) -> bool:
    if not path.is_file():
        return False
    load_file, _, _ = _safetensors_api()
    try:
        tensors = load_file(os.fspath(path), device="cpu")
        value = tensors[tensor_name]
        return tuple(value.shape) == (expected_rows,) + expected_shape_tail
    except Exception:
        return False


def _safetensors_metadata(path: pathlib.Path) -> dict[str, str]:
    _, _, safe_open = _safetensors_api()
    try:
        with safe_open(os.fspath(path), framework="pt", device="cpu") as handle:
            return dict(handle.metadata() or {})
    except Exception as exc:
        raise BuildError(f"cannot read safetensors metadata from {path}: {exc}") from exc


def _valid_committed_tensor_shard(
    path: pathlib.Path,
    journal_entry: Mapping[str, Any] | None,
    *,
    tensor_name: str,
    expected_shape: Sequence[int],
    expected_dtype: torch.dtype,
    expected_metadata: Mapping[str, str],
) -> bool:
    if not path.is_file() or not isinstance(journal_entry, Mapping):
        return False
    if journal_entry.get("sha256") != _sha256_file(path):
        return False
    load_file, _, _ = _safetensors_api()
    try:
        tensors = load_file(os.fspath(path), device="cpu")
        if set(tensors) != {tensor_name}:
            return False
        tensor = tensors[tensor_name]
        if tuple(tensor.shape) != tuple(expected_shape):
            return False
        if tensor.dtype != expected_dtype or not torch.isfinite(tensor).all():
            return False
        if _safetensors_metadata(path) != dict(expected_metadata):
            return False
        return (
            journal_entry.get("tensor_name") == tensor_name
            and journal_entry.get("shape") == list(expected_shape)
            and journal_entry.get("dtype") == str(expected_dtype).removeprefix("torch.")
            and journal_entry.get("metadata") == dict(expected_metadata)
        )
    except Exception:
        return False


def _valid_student_cache_shard(
    path: pathlib.Path,
    journal_entry: Mapping[str, Any] | None,
    *,
    attempt_index: int,
    group_index: int,
    group_layers: Sequence[int],
    response_rows: int,
    training_fingerprint: str,
) -> bool:
    metadata = {
        "attempt_index": str(attempt_index),
        "group": str(group_index),
        "layers": ",".join(str(layer) for layer in group_layers),
        "response_rows": str(response_rows),
        "training_fingerprint": training_fingerprint,
    }
    return _valid_committed_tensor_shard(
        path,
        journal_entry,
        tensor_name="student_hidden",
        expected_shape=(
            response_rows,
            len(group_layers),
            FIXED_CONTRACT["student"]["hidden_size"],
        ),
        expected_dtype=torch.bfloat16,
        expected_metadata=metadata,
    )


def command_materialize_teacher_targets(args: argparse.Namespace) -> None:
    paths = Paths(pathlib.Path(args.work_dir).expanduser().resolve())
    with _workspace_lock(paths):
        contract = _read_contract(paths)
        records, split = _require_rollouts(paths)
        pca, pca_metadata = _load_pca(paths)
        input_fingerprint = _fingerprint(
            {
                "contract": _contract_hash(paths),
                "rollouts": _sha256_file(paths.rollouts),
                "split": _sha256_file(paths.split),
                "pca": _sha256_file(paths.pca_tensors),
                "pca_metadata": _sha256_file(paths.pca_metadata),
                "phase": "materialize-teacher-targets",
                "device": args.device,
                "dtype": args.dtype,
                "runtime": _runtime_versions(),
            }
        )
        with _phase(paths, "materialize-teacher-targets", input_fingerprint) as (
            state,
            should_run,
        ):
            if not should_run:
                return
            _validate_model_shapes(contract)
            paths.targets.mkdir(parents=True, exist_ok=True)
            journal_header = {
                "schema_version": SCHEMA_VERSION,
                "input_fingerprint": input_fingerprint,
                "pca_sha256": _sha256_file(paths.pca_tensors),
            }
            if paths.target_journal.exists():
                target_journal = _load_json(paths.target_journal)
                if not isinstance(target_journal, dict) or any(
                    target_journal.get(key) != value
                    for key, value in journal_header.items()
                ):
                    raise BuildError("teacher target journal is incompatible")
                if not isinstance(target_journal.get("entries"), dict):
                    raise BuildError("teacher target journal has no entries mapping")
            else:
                target_journal = {**journal_header, "entries": {}}
                _atomic_write_json(paths.target_journal, target_journal)
            model = _load_model(
                contract["identity"]["teacher_model_id"],
                contract["identity"]["teacher_model_revision"],
                args.device,
                args.dtype,
            )
            device = torch.device(args.device)
            layer_map = _layer_map()
            pca_device = {
                key: value.to(device=device, dtype=torch.float32) for key, value in pca.items()
            }
            for position, record in enumerate(records, 1):
                index = int(record["attempt_index"])
                _, row_count = _response_window(record)
                destination = _target_path(paths, index)
                shard_metadata = {
                    "attempt_index": str(index),
                    "split": _split_label(index, split),
                    "response_rows": str(row_count),
                    "rank": str(FIXED_CONTRACT["rank"]),
                    "layers": str(len(layer_map)),
                    "input_fingerprint": input_fingerprint,
                }
                journal_entry = target_journal["entries"].get(str(index))
                if _valid_committed_tensor_shard(
                    destination,
                    journal_entry,
                    tensor_name="targets",
                    expected_shape=(
                        row_count,
                        len(layer_map),
                        FIXED_CONTRACT["rank"],
                    ),
                    expected_dtype=torch.bfloat16,
                    expected_metadata=shard_metadata,
                ):
                    continue
                if journal_entry is not None:
                    raise BuildError(
                        f"committed teacher target shard failed validation: {destination}"
                    )
                if destination.exists():
                    destination.unlink()
                ids = torch.tensor([_full_ids(record)], dtype=torch.long, device=device)
                response_start, _ = _response_window(record)
                with torch.inference_mode():
                    hidden_states = _backbone_hidden(model, ids)
                    projected: list[torch.Tensor] = []
                    for student_layer, teacher_layer in layer_map.items():
                        hidden = hidden_states[teacher_layer][
                            0, response_start : response_start + row_count
                        ].float()
                        mean = pca_device[f"mean.{student_layer}"]
                        components = pca_device[f"components.{student_layer}"]
                        projected.append((hidden - mean) @ components.T)
                    targets = torch.stack(projected, dim=1).to(torch.bfloat16).cpu()
                _atomic_save_safetensors(
                    destination,
                    {"targets": targets},
                    metadata=shard_metadata,
                )
                target_journal["entries"][str(index)] = {
                    "attempt_index": index,
                    "sha256": _sha256_file(destination),
                    "tensor_name": "targets",
                    "shape": [row_count, len(layer_map), FIXED_CONTRACT["rank"]],
                    "dtype": "bfloat16",
                    "metadata": shard_metadata,
                }
                _atomic_write_json(paths.target_journal, target_journal)
                if position % 25 == 0 or position == len(records):
                    print(
                        f"materialize-teacher-targets: {position}/{len(records)}",
                        flush=True,
                    )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            entries: list[dict[str, Any]] = []
            for record in records:
                index = int(record["attempt_index"])
                target = _target_path(paths, index)
                _, row_count = _response_window(record)
                shard_metadata = {
                    "attempt_index": str(index),
                    "split": _split_label(index, split),
                    "response_rows": str(row_count),
                    "rank": str(FIXED_CONTRACT["rank"]),
                    "layers": str(len(layer_map)),
                    "input_fingerprint": input_fingerprint,
                }
                if not _valid_committed_tensor_shard(
                    target,
                    target_journal["entries"].get(str(index)),
                    tensor_name="targets",
                    expected_shape=(
                        row_count,
                        len(layer_map),
                        FIXED_CONTRACT["rank"],
                    ),
                    expected_dtype=torch.bfloat16,
                    expected_metadata=shard_metadata,
                ):
                    raise BuildError(f"teacher target shard failed validation: {target}")
                entries.append(
                    {
                        "attempt_index": index,
                        "split": _split_label(index, split),
                        "rows": row_count,
                        "file": target.name,
                        "sha256": _sha256_file(target),
                    }
                )
            index_payload = {
                "schema_version": SCHEMA_VERSION,
                "rank": FIXED_CONTRACT["rank"],
                "layers": list(layer_map),
                "dtype": "bfloat16",
                "projected_only": True,
                "teacher_model_id": contract["identity"]["teacher_model_id"],
                "teacher_model_revision": contract["identity"][
                    "teacher_model_revision"
                ],
                "corpus_sha256": _sha256_file(paths.rollouts),
                "split_sha256": _sha256_file(paths.split),
                "response_positions": FIXED_CONTRACT["response_positions"],
                "pca_sha256": _sha256_file(paths.pca_tensors),
                "pca_sampled_rows": pca_metadata["sampled_rows"],
                "target_journal_sha256": _sha256_file(paths.target_journal),
                "entries": entries,
            }
            _atomic_write_json(paths.targets_index, index_payload)
            _finish_phase(
                paths,
                state,
                "materialize-teacher-targets",
                [paths.targets, paths.target_journal],
                {
                    "shards": len(entries),
                    "total_rows": sum(entry["rows"] for entry in entries),
                    "shape": ["response_rows", 28, 8],
                    "dtype": "bfloat16",
                },
            )
    print(f"materialized compact rank-8 teacher targets: {paths.targets}")


# ---------------------------------------------------------------------------
# Seven serial four-layer student caches and projector training


def _initial_group_bank(group_index: int) -> BridgeBank:
    all_layers = list(_layer_map())
    group_layers = _group_layers(group_index)
    student_width = FIXED_CONTRACT["student"]["hidden_size"]
    teacher_width = FIXED_CONTRACT["teacher"]["hidden_size"]
    rank = FIXED_CONTRACT["rank"]
    devices: list[int] = []
    if torch.cuda.is_available():
        devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(FIXED_CONTRACT["prompt_seed"])
        full = BridgeBank(all_layers, student_width, teacher_width, rank)
        group = BridgeBank(group_layers, student_width, teacher_width, rank)
        for layer in group_layers:
            group.student_projectors[str(layer)].weight.data.copy_(
                full.student_projectors[str(layer)].weight.data
            )
    return group


def _projector_tensors(bank: BridgeBank) -> dict[str, torch.Tensor]:
    return {
        f"student_projectors.{layer}.weight": bank.student_projectors[
            str(layer)
        ].weight.detach().cpu()
        for layer in bank.layers
    }


def _load_projector_tensors(bank: BridgeBank, path: pathlib.Path) -> None:
    load_file, _, _ = _safetensors_api()
    tensors = load_file(os.fspath(path), device="cpu")
    with torch.no_grad():
        for layer in bank.layers:
            key = f"student_projectors.{layer}.weight"
            expected = bank.student_projectors[str(layer)].weight.shape
            if key not in tensors or tensors[key].shape != expected:
                raise BuildError(f"invalid projector checkpoint tensor {key} in {path}")
            bank.student_projectors[str(layer)].weight.copy_(tensors[key])


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _require_finite_training_state(value: Any, *, context: str) -> None:
    if torch.is_tensor(value):
        if not bool(torch.isfinite(value).all().item()):
            raise BuildError(f"{context} contains a non-finite tensor")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BuildError(f"{context} contains a non-finite scalar")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require_finite_training_state(item, context=f"{context}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite_training_state(item, context=f"{context}[{index}]")


def _target_index_by_attempt(paths: Paths) -> dict[int, dict[str, Any]]:
    payload = _load_json(paths.targets_index)
    contract = _load_json(paths.contract)
    identity = contract.get("identity", {}) if isinstance(contract, dict) else {}
    expected_metadata = {
        "schema_version": SCHEMA_VERSION,
        "rank": FIXED_CONTRACT["rank"],
        "layers": list(_layer_map()),
        "dtype": "bfloat16",
        "projected_only": True,
        "teacher_model_id": identity.get("teacher_model_id"),
        "teacher_model_revision": identity.get("teacher_model_revision"),
        "corpus_sha256": _sha256_file(paths.rollouts),
        "split_sha256": _sha256_file(paths.split),
        "response_positions": FIXED_CONTRACT["response_positions"],
        "pca_sha256": _sha256_file(paths.pca_tensors),
        "target_journal_sha256": _sha256_file(paths.target_journal),
    }
    if not isinstance(payload, dict) or any(
        payload.get(key) != value for key, value in expected_metadata.items()
    ):
        raise BuildError("teacher target index metadata is incompatible with this build")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != FIXED_CONTRACT["attempted_prompts"]:
        raise BuildError("teacher target index is invalid")
    result: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("attempt_index"), int):
            raise BuildError("teacher target index has an invalid entry")
        index = entry["attempt_index"]
        if index in result:
            raise BuildError(f"teacher target index duplicates attempt {index}")
        result[index] = entry
    if set(result) != set(range(FIXED_CONTRACT["attempted_prompts"])):
        raise BuildError(
            "teacher target index does not cover the exact attempted-prompt set"
        )
    return result


def _validate_target_shards(
    paths: Paths,
    records: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    entries = _target_index_by_attempt(paths)
    target_journal = _load_json(paths.target_journal)
    journal_entries = (
        target_journal.get("entries") if isinstance(target_journal, dict) else None
    )
    if not isinstance(journal_entries, dict):
        raise BuildError("teacher target journal is invalid")
    for record in records:
        index = int(record["attempt_index"])
        entry = entries[index]
        target = _target_path(paths, index)
        _, rows = _response_window(record)
        expected = {
            "attempt_index": index,
            "split": _split_label(index, split),
            "rows": rows,
            "file": target.name,
        }
        for field, value in expected.items():
            if entry.get(field) != value:
                raise BuildError(
                    f"teacher target index mismatch for attempt {index}: "
                    f"{field}={entry.get(field)!r}, expected {value!r}"
                )
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise BuildError(f"teacher target index has invalid SHA-256 at attempt {index}")
        if not target.is_file() or _sha256_file(target) != digest:
            raise BuildError(f"teacher target checksum failed at attempt {index}")
        journal_entry = journal_entries.get(str(index))
        expected_metadata = (
            journal_entry.get("metadata")
            if isinstance(journal_entry, dict)
            else None
        )
        if (
            not isinstance(expected_metadata, dict)
            or expected_metadata.get("attempt_index") != str(index)
            or expected_metadata.get("split") != _split_label(index, split)
            or expected_metadata.get("response_rows") != str(rows)
            or expected_metadata.get("rank") != str(FIXED_CONTRACT["rank"])
            or expected_metadata.get("layers") != str(len(_layer_map()))
            or not _valid_committed_tensor_shard(
                target,
                journal_entry,
                tensor_name="targets",
                expected_shape=(
                    rows,
                    len(_layer_map()),
                    FIXED_CONTRACT["rank"],
                ),
                expected_dtype=torch.bfloat16,
                expected_metadata=expected_metadata,
            )
        ):
            raise BuildError(f"teacher target shard failed journal validation at {index}")
    return entries


def _load_shard_tensor(path: pathlib.Path, name: str) -> torch.Tensor:
    load_file, _, _ = _safetensors_api()
    try:
        return load_file(os.fspath(path), device="cpu")[name]
    except Exception as exc:
        raise BuildError(f"cannot load {name!r} from {path}: {exc}") from exc


def _extract_student_cache(
    paths: Paths,
    contract: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    group_index: int,
    args: argparse.Namespace,
    training_fingerprint: str,
) -> None:
    group_layers = _group_layers(group_index)
    cache_dir = paths.cache(group_index)
    cache_dir.mkdir(parents=True, exist_ok=True)
    journal_path = paths.cache_journal(group_index)
    journal_header = {
        "schema_version": SCHEMA_VERSION,
        "group": group_index,
        "layers": group_layers,
        "training_fingerprint": training_fingerprint,
        "student_model_id": contract["identity"]["student_model_id"],
        "student_model_revision": contract["identity"]["student_model_revision"],
    }
    if journal_path.exists():
        cache_journal = _load_json(journal_path)
        if not isinstance(cache_journal, dict) or any(
            cache_journal.get(key) != value for key, value in journal_header.items()
        ):
            raise BuildError(f"student cache journal is incompatible: {journal_path}")
        if not isinstance(cache_journal.get("entries"), dict):
            raise BuildError("student cache journal has no entries mapping")
    else:
        cache_journal = {**journal_header, "entries": {}}
        _atomic_write_json(journal_path, cache_journal)
    model = _load_model(
        contract["identity"]["student_model_id"],
        contract["identity"]["student_model_revision"],
        args.device,
        args.dtype,
    )
    device = torch.device(args.device)
    for position, record in enumerate(records, 1):
        index = int(record["attempt_index"])
        response_start, row_count = _response_window(record)
        destination = _cache_path(paths, group_index, index)
        journal_entry = cache_journal["entries"].get(str(index))
        if _valid_student_cache_shard(
            destination,
            journal_entry,
            attempt_index=index,
            group_index=group_index,
            group_layers=group_layers,
            response_rows=row_count,
            training_fingerprint=training_fingerprint,
        ):
            continue
        if journal_entry is not None:
            raise BuildError(
                f"committed student cache shard failed validation: {destination}"
            )
        if destination.exists():
            destination.unlink()
        ids = torch.tensor([_full_ids(record)], dtype=torch.long, device=device)
        with torch.inference_mode():
            hidden_states = _backbone_hidden(model, ids)
            selected = torch.stack(
                [
                    hidden_states[layer][
                        0, response_start : response_start + row_count
                    ]
                    for layer in group_layers
                ],
                dim=1,
            ).to(torch.bfloat16).cpu()
        shard_metadata = {
            "attempt_index": str(index),
            "group": str(group_index),
            "layers": ",".join(str(layer) for layer in group_layers),
            "response_rows": str(row_count),
            "training_fingerprint": training_fingerprint,
        }
        _atomic_save_safetensors(
            destination,
            {"student_hidden": selected},
            metadata=shard_metadata,
        )
        cache_journal["entries"][str(index)] = {
            "attempt_index": index,
            "sha256": _sha256_file(destination),
            "tensor_name": "student_hidden",
            "shape": [
                row_count,
                len(group_layers),
                FIXED_CONTRACT["student"]["hidden_size"],
            ],
            "dtype": "bfloat16",
            "metadata": shard_metadata,
        }
        _atomic_write_json(journal_path, cache_journal)
        if position % 25 == 0 or position == len(records):
            print(
                f"train-student-group {group_index}: cached {position}/{len(records)} rollouts",
                flush=True,
            )
    cache_entries: list[dict[str, Any]] = []
    for record in records:
        index = int(record["attempt_index"])
        _, row_count = _response_window(record)
        shard = _cache_path(paths, group_index, index)
        if not _valid_student_cache_shard(
            shard,
            cache_journal["entries"].get(str(index)),
            attempt_index=index,
            group_index=group_index,
            group_layers=group_layers,
            response_rows=row_count,
            training_fingerprint=training_fingerprint,
        ):
            raise BuildError(f"student cache shard failed validation: {shard}")
        cache_entries.append(
            {
                "attempt_index": index,
                "rows": row_count,
                "file": shard.name,
                "sha256": _sha256_file(shard),
            }
        )
    _atomic_write_json(
        cache_dir / "index.json",
        {
            "schema_version": SCHEMA_VERSION,
            "group": group_index,
            "layers": group_layers,
            "student_model_id": contract["identity"]["student_model_id"],
            "student_model_revision": contract["identity"]["student_model_revision"],
            "dtype": "bfloat16",
            "training_fingerprint": training_fingerprint,
            "cache_journal_sha256": _sha256_file(journal_path),
            "entries": cache_entries,
        },
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validation_metrics(
    bank: BridgeBank,
    paths: Paths,
    validation_indices: Sequence[int],
    group_index: int,
    layer_positions: Sequence[int],
    device: torch.device,
    batch_rows: int,
) -> tuple[float, int, float]:
    cosine_sum = 0.0
    cosine_count = 0
    mse_sum = 0.0
    mse_count = 0
    bank.eval()
    with torch.inference_mode():
        for index in validation_indices:
            hidden = _load_shard_tensor(
                _cache_path(paths, group_index, index), "student_hidden"
            )
            targets = _load_shard_tensor(_target_path(paths, index), "targets")[
                :, layer_positions, :
            ]
            for start in range(0, hidden.shape[0], batch_rows):
                h = hidden[start : start + batch_rows].to(device).float()
                target = targets[start : start + batch_rows].to(device).float()
                for local_layer, layer in enumerate(bank.layers):
                    prediction = bank.project_student(layer, h[:, local_layer, :])
                    reference = target[:, local_layer, :]
                    cos = F.cosine_similarity(prediction, reference, dim=-1)
                    cosine_sum += float(cos.sum().item())
                    cosine_count += int(cos.numel())
                    squared = (prediction - reference).square()
                    mse_sum += float(squared.sum().item())
                    mse_count += int(squared.numel())
    if cosine_count == 0 or mse_count == 0:
        raise BuildError("validation split produced no token rows")
    validation_mse = mse_sum / mse_count
    if not math.isfinite(cosine_sum) or not math.isfinite(validation_mse):
        raise BuildError("validation produced non-finite Stage-1 metrics")
    return cosine_sum, cosine_count, validation_mse


def _epoch_rollout_batches(
    records_by_index: Mapping[int, Mapping[str, Any]],
    train_indices: Sequence[int],
) -> list[list[int]]:
    batching = FIXED_CONTRACT["optimizer"]["batching"]
    max_rollouts = batching["max_rollouts"]
    max_tokens = batching["max_padded_tokens"]
    order = sorted(
        (int(value) for value in train_indices),
        key=lambda index: (-len(_full_ids(records_by_index[index])), index),
    )
    batches: list[list[int]] = []
    current: list[int] = []
    current_max_length = 0
    for index in order:
        token_count = len(_full_ids(records_by_index[index]))
        if token_count > max_tokens:
            raise BuildError(
                f"rollout {index} has {token_count} full-sequence tokens, exceeding "
                f"the fixed {max_tokens}-token Stage-1 batch cap"
            )
        candidate_size = len(current) + 1
        candidate_max_length = max(current_max_length, token_count)
        candidate_tokens = candidate_size * candidate_max_length
        if current and (
            len(current) >= max_rollouts or candidate_tokens > max_tokens
        ):
            batches.append(current)
            current = []
            current_max_length = 0
        current.append(index)
        current_max_length = max(current_max_length, token_count)
    if current:
        batches.append(current)
    return batches


def _group_objective(
    bank: BridgeBank,
    group_layers: Sequence[int],
    hidden: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    layer_losses = torch.stack(
        [
            F.mse_loss(
                bank.project_student(layer, hidden[:, local_layer, :]),
                targets[:, local_layer, :],
            )
            for local_layer, layer in enumerate(group_layers)
        ]
    )
    all_layer_contribution = layer_losses.sum() / len(_layer_map())
    return all_layer_contribution, layer_losses.mean()


def _train_group(
    paths: Paths,
    records: Sequence[Mapping[str, Any]],
    split: Mapping[str, Any],
    group_index: int,
    args: argparse.Namespace,
    training_fingerprint: str,
) -> dict[str, Any]:
    group_layers = _group_layers(group_index)
    all_layers = list(_layer_map())
    layer_positions = [all_layers.index(layer) for layer in group_layers]
    records_by_index = {
        int(record["attempt_index"]): record for record in records
    }
    group_dir = paths.group(group_index)
    epochs_dir = group_dir / "epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    bank = _initial_group_bank(group_index).to(device).float()
    optimizer = torch.optim.AdamW(
        bank.student_projectors.parameters(),
        lr=FIXED_CONTRACT["optimizer"]["learning_rate"],
        weight_decay=FIXED_CONTRACT["optimizer"]["weight_decay"],
    )
    state_path = group_dir / "training_state.pt"
    metrics_path = group_dir / "metrics.json"
    start_epoch = 1
    start_batch = 0
    resume_plan_sha256: str | None = None
    epoch_loss_sum = 0.0
    epoch_objective_sum = 0.0
    epoch_loss_rows = 0
    metrics: list[dict[str, Any]] = []
    if state_path.is_file():
        checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
        if checkpoint.get("training_state_version") != 2:
            raise BuildError(
                f"unsupported group training checkpoint in {state_path}; "
                "remove this session-created checkpoint and restart the group"
            )
        if checkpoint.get("training_fingerprint") != training_fingerprint:
            raise BuildError("group training checkpoint inputs or settings changed")
        _require_finite_training_state(
            checkpoint, context=f"group {group_index} training checkpoint"
        )
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint["next_batch"])
        resume_plan_sha256 = checkpoint.get("batch_plan_sha256")
        epoch_loss_sum = float(checkpoint.get("epoch_loss_sum", 0.0))
        epoch_objective_sum = float(checkpoint.get("epoch_objective_sum", 0.0))
        epoch_loss_rows = int(checkpoint.get("epoch_loss_rows", 0))
        projector_state = checkpoint.get("projectors")
        if not isinstance(projector_state, Mapping):
            raise BuildError("group training checkpoint has no projector state")
        with torch.no_grad():
            for layer in group_layers:
                key = f"student_projectors.{layer}.weight"
                value = projector_state.get(key)
                expected = bank.student_projectors[str(layer)].weight.shape
                if not torch.is_tensor(value) or value.shape != expected:
                    raise BuildError(f"invalid projector state {key} in {state_path}")
                bank.student_projectors[str(layer)].weight.copy_(value.to(device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        _move_optimizer_state(optimizer, device)
        if metrics_path.is_file():
            metrics_payload = _load_json(metrics_path)
            metrics = list(metrics_payload.get("epochs", []))
            metrics = [
                value for value in metrics if int(value["epoch"]) < start_epoch
            ]

    train_indices = list(split["train_indices"])
    validation_indices = list(split["validation_indices"])
    final_epoch = FIXED_CONTRACT["optimizer"]["epochs"]
    for epoch in range(start_epoch, final_epoch + 1):
        bank.train()
        batches = _epoch_rollout_batches(records_by_index, train_indices)
        batch_plan_sha256 = _fingerprint(batches)
        batch_start = start_batch if epoch == start_epoch else 0
        if batch_start < 0 or batch_start > len(batches):
            raise BuildError(
                f"group checkpoint next_batch={batch_start} outside 0..{len(batches)}"
            )
        if epoch == start_epoch and start_batch and resume_plan_sha256 != batch_plan_sha256:
            raise BuildError("group checkpoint batch plan does not match this build")
        if epoch != start_epoch:
            epoch_loss_sum = 0.0
            epoch_objective_sum = 0.0
            epoch_loss_rows = 0

        for batch_number in range(batch_start, len(batches)):
            batch_indices = batches[batch_number]
            hidden_parts: list[torch.Tensor] = []
            target_parts: list[torch.Tensor] = []
            for index in batch_indices:
                hidden = _load_shard_tensor(
                    _cache_path(paths, group_index, index), "student_hidden"
                )
                targets = _load_shard_tensor(_target_path(paths, index), "targets")[
                    :, layer_positions, :
                ]
                if hidden.shape[0] != targets.shape[0]:
                    raise BuildError(
                        f"student/teacher row mismatch for attempt {index}: "
                        f"{hidden.shape[0]} != {targets.shape[0]}"
                    )
                if hidden.shape[0]:
                    hidden_parts.append(hidden)
                    target_parts.append(targets)
            if hidden_parts:
                h = torch.cat(hidden_parts, dim=0).to(device).float()
                target = torch.cat(target_parts, dim=0).to(device).float()
                optimizer.zero_grad(set_to_none=True)
                objective, group_mse = _group_objective(
                    bank, group_layers, h, target
                )
                if not torch.isfinite(objective) or not torch.isfinite(group_mse):
                    raise BuildError(
                        f"non-finite Stage-1 loss in group {group_index}, "
                        f"epoch {epoch}, batch {batch_number}"
                    )
                objective.backward()
                for name, parameter in bank.student_projectors.named_parameters():
                    gradient = parameter.grad
                    if gradient is None or not bool(torch.isfinite(gradient).all().item()):
                        raise BuildError(
                            f"non-finite or missing Stage-1 gradient {name} in group "
                            f"{group_index}, epoch {epoch}, batch {batch_number}"
                        )
                optimizer.step()
                rows = int(h.shape[0])
                epoch_loss_sum += float(group_mse.item()) * rows
                epoch_objective_sum += float(objective.item()) * rows
                epoch_loss_rows += rows

            projector_state = _projector_tensors(bank)
            optimizer_state = optimizer.state_dict()
            _require_finite_training_state(
                projector_state, context=f"group {group_index} projector state"
            )
            _require_finite_training_state(
                optimizer_state, context=f"group {group_index} optimizer state"
            )
            _atomic_torch_save(
                state_path,
                {
                    "training_state_version": 2,
                    "training_fingerprint": training_fingerprint,
                    "group": group_index,
                    "epoch": epoch,
                    "next_batch": batch_number + 1,
                    "batch_plan_sha256": batch_plan_sha256,
                    "epoch_loss_sum": epoch_loss_sum,
                    "epoch_objective_sum": epoch_objective_sum,
                    "epoch_loss_rows": epoch_loss_rows,
                    "projectors": projector_state,
                    "optimizer": optimizer_state,
                },
            )

        cosine_sum, cosine_count, validation_mse = _validation_metrics(
            bank,
            paths,
            validation_indices,
            group_index,
            layer_positions,
            device,
            args.batch_rows,
        )
        epoch_path = epochs_dir / f"epoch_{epoch:02d}.safetensors"
        _atomic_save_safetensors(
            epoch_path,
            _projector_tensors(bank),
            metadata={
                "group": str(group_index),
                "epoch": str(epoch),
                "layers": ",".join(str(layer) for layer in group_layers),
                "batch_plan_sha256": batch_plan_sha256,
            },
        )
        epoch_metrics = {
            "epoch": epoch,
            "optimizer_steps": sum(
                any(_response_window(records_by_index[index])[1] for index in batch)
                for batch in batches
            ),
            "batch_plan_sha256": batch_plan_sha256,
            "train_mse": epoch_loss_sum / max(epoch_loss_rows, 1),
            "train_all_layer_objective_contribution": (
                epoch_objective_sum / max(epoch_loss_rows, 1)
            ),
            "train_token_rows": epoch_loss_rows,
            "validation_mse": validation_mse,
            "validation_cosine_sum": cosine_sum,
            "validation_cosine_count": cosine_count,
            "validation_cosine": cosine_sum / cosine_count,
        }
        metrics = [value for value in metrics if int(value["epoch"]) != epoch]
        metrics.append(epoch_metrics)
        metrics.sort(key=lambda value: int(value["epoch"]))
        _atomic_write_json(
            metrics_path,
            {
                "group": group_index,
                "layers": group_layers,
                "training_fingerprint": training_fingerprint,
                "optimizer": FIXED_CONTRACT["optimizer"],
                "epochs": metrics,
                "selection_deferred_to_publish": True,
            },
        )
        projector_state = _projector_tensors(bank)
        optimizer_state = optimizer.state_dict()
        _require_finite_training_state(
            projector_state, context=f"group {group_index} epoch projector state"
        )
        _require_finite_training_state(
            optimizer_state, context=f"group {group_index} epoch optimizer state"
        )
        _atomic_torch_save(
            state_path,
            {
                "training_state_version": 2,
                "training_fingerprint": training_fingerprint,
                "group": group_index,
                "epoch": epoch + 1,
                "next_batch": 0,
                "batch_plan_sha256": None,
                "epoch_loss_sum": 0.0,
                "epoch_objective_sum": 0.0,
                "epoch_loss_rows": 0,
                "projectors": projector_state,
                "optimizer": optimizer_state,
            },
        )
        print(
            f"train-student-group {group_index}: epoch {epoch}/{final_epoch} "
            f"train_mse={epoch_metrics['train_mse']:.6f} "
            f"val_cos={epoch_metrics['validation_cosine']:.6f}",
            flush=True,
        )

    if len(metrics) != final_epoch:
        raise BuildError(
            f"group {group_index} does not have all {final_epoch} epoch metrics"
        )
    for epoch in range(1, final_epoch + 1):
        checkpoint = epochs_dir / f"epoch_{epoch:02d}.safetensors"
        if not checkpoint.is_file():
            raise BuildError(f"missing group {group_index} epoch checkpoint: {checkpoint}")
    summary = {
        "group": group_index,
        "layers": group_layers,
        "epochs": final_epoch,
        "batching": FIXED_CONTRACT["optimizer"]["batching"],
        "training_fingerprint": training_fingerprint,
        "epoch_selection": "deferred; publish chooses one global epoch",
        "metrics_sha256": _sha256_file(metrics_path),
    }
    _atomic_write_json(group_dir / "complete.json", summary)
    return summary


def command_train_student_group(args: argparse.Namespace) -> None:
    paths = Paths(pathlib.Path(args.work_dir).expanduser().resolve())
    group_index = args.group
    phase_name = f"train-student-group-{group_index}"
    with _workspace_lock(paths):
        contract = _read_contract(paths)
        records, split = _require_rollouts(paths)
        build_state = _read_state(paths)
        _require_completed_phase_output(
            paths, "materialize-teacher-targets", state=build_state
        )
        for previous in range(group_index):
            previous_name = f"train-student-group-{previous}"
            try:
                _require_completed_phase_output(
                    paths, previous_name, state=build_state
                )
            except BuildError as exc:
                raise BuildError(
                    f"groups are serial: valid {previous_name} outputs are required first"
                ) from exc
        input_fingerprint = _fingerprint(
            {
                "contract": _contract_hash(paths),
                "rollouts": _sha256_file(paths.rollouts),
                "split": _sha256_file(paths.split),
                "targets_index": _sha256_file(paths.targets_index),
                "targets_tree": _sha256_tree(paths.targets),
                "group": group_index,
                "layers": _group_layers(group_index),
                "student_cache_device": args.device,
                "student_cache_dtype": args.dtype,
                "validation_batch_rows": args.batch_rows,
                "optimizer": FIXED_CONTRACT["optimizer"],
                "runtime": _runtime_versions(),
            }
        )
        with _phase(paths, phase_name, input_fingerprint) as (state, should_run):
            if not should_run:
                return
            _validate_model_shapes(contract)
            _validate_target_shards(paths, records, split)
            other_caches = [
                path
                for path in (paths.root / "student_cache").glob("group_*")
                if path.is_dir() and path != paths.cache(group_index)
            ]
            if other_caches and not args.keep_cache:
                raise BuildError(
                    "serial cache invariant violated; remove completed caches or use --keep-cache consistently"
                )
            _extract_student_cache(
                paths,
                contract,
                records,
                group_index,
                args,
                input_fingerprint,
            )
            summary = _train_group(
                paths,
                records,
                split,
                group_index,
                args,
                input_fingerprint,
            )
            if not args.keep_cache:
                _remove_tree_and_journal(
                    paths.cache(group_index), paths.cache_journal(group_index)
                )
            output_paths = [paths.group(group_index)]
            if args.keep_cache:
                output_paths.extend(
                    [paths.cache(group_index), paths.cache_journal(group_index)]
                )
            _finish_phase(
                paths,
                state,
                phase_name,
                output_paths,
                {
                    **summary,
                    "cache_policy": "retained" if args.keep_cache else "deleted after training",
                    "cache_shape": ["response_rows", 4, 2048],
                },
            )
    print(
        f"trained serial four-layer student group {group_index}: {paths.group(group_index)}"
    )


# ---------------------------------------------------------------------------
# Global epoch selection, immutable publish, and verification


def _global_epoch_table(paths: Paths) -> list[dict[str, Any]]:
    epoch_count = FIXED_CONTRACT["optimizer"]["epochs"]
    table = [
        {
            "epoch": epoch,
            "valid": True,
            "validation_cosine_sum": 0.0,
            "validation_cosine_count": 0,
            "invalid_groups": [],
        }
        for epoch in range(1, epoch_count + 1)
    ]
    for group_index in range(FIXED_CONTRACT["student_groups"]["count"]):
        metrics = _load_json(paths.group(group_index) / "metrics.json")
        epochs = metrics.get("epochs") if isinstance(metrics, dict) else None
        if not isinstance(epochs, list) or len(epochs) != epoch_count:
            raise BuildError(
                f"group {group_index} lacks {epoch_count} complete epoch metrics"
            )
        try:
            by_epoch = {int(value["epoch"]): value for value in epochs}
        except (KeyError, TypeError, ValueError) as exc:
            raise BuildError(f"group {group_index} has invalid epoch metadata") from exc
        for aggregate in table:
            value = by_epoch.get(aggregate["epoch"])
            if not isinstance(value, dict):
                raise BuildError(
                    f"group {group_index} lacks metrics for epoch {aggregate['epoch']}"
                )
            invalid_reason: str | None = None
            try:
                cosine_sum = float(value["validation_cosine_sum"])
                cosine_count = int(value["validation_cosine_count"])
                for metric_name in (
                    "train_mse",
                    "validation_mse",
                    "validation_cosine",
                ):
                    if not math.isfinite(float(value[metric_name])):
                        invalid_reason = f"non-finite {metric_name}"
                        break
                if invalid_reason is None and (
                    not math.isfinite(cosine_sum) or cosine_count <= 0
                ):
                    invalid_reason = "invalid validation accumulators"
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                invalid_reason = f"invalid metric structure: {type(exc).__name__}"
            if invalid_reason is not None:
                aggregate["valid"] = False
                aggregate["invalid_groups"].append(
                    {"group": group_index, "reason": invalid_reason}
                )
                continue
            aggregate["validation_cosine_sum"] += cosine_sum
            aggregate["validation_cosine_count"] += cosine_count
    for aggregate in table:
        count = aggregate["validation_cosine_count"]
        if not aggregate["valid"] or count <= 0:
            aggregate["valid"] = False
            aggregate["validation_cosine"] = None
            continue
        score = aggregate["validation_cosine_sum"] / count
        if not math.isfinite(score):
            aggregate["valid"] = False
            aggregate["validation_cosine_sum"] = None
            aggregate["validation_cosine"] = None
            aggregate["invalid_groups"].append(
                {"group": None, "reason": "non-finite global validation cosine"}
            )
            continue
        aggregate["validation_cosine"] = score
    return table


def _selected_epoch(table: Sequence[Mapping[str, Any]]) -> int:
    best_epoch: int | None = None
    best_score = -float("inf")
    for value in table:
        if value.get("valid", True) is False:
            continue
        raw_score = value.get("validation_cosine")
        if raw_score is None:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = int(value["epoch"])
    if best_epoch is None:
        raise BuildError("no finite global validation epoch is available")
    return best_epoch


def _build_published_bridge(paths: Paths, selected_epoch: int) -> BridgeBank:
    layer_map = _layer_map()
    bridge = BridgeBank(
        list(layer_map),
        FIXED_CONTRACT["student"]["hidden_size"],
        FIXED_CONTRACT["teacher"]["hidden_size"],
        FIXED_CONTRACT["rank"],
    )
    pca, _ = _load_pca(paths)
    for layer in layer_map:
        bridge.set_teacher_projection(
            layer, pca[f"mean.{layer}"], pca[f"components.{layer}"]
        )
    for group_index in range(FIXED_CONTRACT["student_groups"]["count"]):
        group_bank = _initial_group_bank(group_index)
        epoch_path = (
            paths.group(group_index)
            / "epochs"
            / f"epoch_{selected_epoch:02d}.safetensors"
        )
        _load_projector_tensors(group_bank, epoch_path)
        with torch.no_grad():
            for layer in group_bank.layers:
                bridge.student_projectors[str(layer)].weight.copy_(
                    group_bank.student_projectors[str(layer)].weight
                )
    for name, tensor in bridge.state_dict().items():
        if not torch.isfinite(tensor).all():
            raise BuildError(f"selected bridge tensor is non-finite: {name}")
    return bridge.freeze()


def _corpus_id(paths: Paths) -> str:
    return f"sha256:{_sha256_file(paths.rollouts)}"


def _artifact_expected_kwargs(
    contract: Mapping[str, Any], paths: Paths
) -> dict[str, Any]:
    return {
        "expected_student_model_id": contract["identity"]["student_model_id"],
        "expected_student_model_revision": contract["identity"][
            "student_model_revision"
        ],
        "expected_teacher_model_id": contract["identity"]["teacher_model_id"],
        "expected_teacher_model_revision": contract["identity"][
            "teacher_model_revision"
        ],
        "expected_tokenizer_id": contract["identity"]["tokenizer_id"],
        "expected_tokenizer_revision": contract["identity"]["tokenizer_revision"],
        "expected_stage1_contract_sha256": contract[
            "stage1_contract_sha256"
        ],
        "expected_corpus_id": _corpus_id(paths),
        "expected_layer_map": _layer_map(),
        "expected_rank": FIXED_CONTRACT["rank"],
        "expected_student_dim": FIXED_CONTRACT["student"]["hidden_size"],
        "expected_teacher_dim": FIXED_CONTRACT["teacher"]["hidden_size"],
    }


def _assert_exact_artifact(
    artifact_dir: pathlib.Path,
    expected_bridge: BridgeBank,
    expected_stage1: Mapping[str, Any],
    expected_sidecars: Mapping[str, Mapping[str, Any]],
    expected_load_kwargs: Mapping[str, Any],
) -> None:
    actual_bridge, actual_stage1 = load_bridge_artifact(
        artifact_dir, device="cpu", **expected_load_kwargs
    )
    if actual_stage1 != dict(expected_stage1):
        raise BuildError(
            "existing artifact Stage-1 provenance does not exactly match this build"
        )
    expected_state = expected_bridge.state_dict()
    actual_state = actual_bridge.state_dict()
    if set(actual_state) != set(expected_state):
        raise BuildError("existing artifact bridge tensor names do not match this build")
    for name, expected_tensor in expected_state.items():
        if not torch.equal(actual_state[name].cpu(), expected_tensor.cpu()):
            raise BuildError(
                f"existing artifact bridge tensor differs from this build: {name}"
            )
    for filename, expected_payload in expected_sidecars.items():
        sidecar = artifact_dir / filename
        if _load_json(sidecar) != dict(expected_payload):
            raise BuildError(
                f"existing artifact sidecar differs from this build: {filename}"
            )


def command_publish(args: argparse.Namespace) -> None:
    paths = Paths(pathlib.Path(args.work_dir).expanduser().resolve())
    with _workspace_lock(paths):
        contract = _read_contract(paths)
        _require_completed_phase_outputs(paths, include_publish=False)
        _require_rollouts(paths)
        for group_index in range(FIXED_CONTRACT["student_groups"]["count"]):
            complete = paths.group(group_index) / "complete.json"
            if not complete.is_file():
                raise BuildError(f"student group {group_index} is not complete")
        group_inputs = {
            str(group_index): {
                "tree": _sha256_tree(paths.group(group_index)),
                "metrics": _sha256_file(paths.group(group_index) / "metrics.json"),
                "complete": _sha256_file(paths.group(group_index) / "complete.json"),
            }
            for group_index in range(FIXED_CONTRACT["student_groups"]["count"])
        }
        targets_tree_sha256 = _sha256_tree(paths.targets)
        input_fingerprint = _fingerprint(
            {
                "contract": _contract_hash(paths),
                "corpus": _sha256_file(paths.rollouts),
                "pca": _sha256_file(paths.pca_tensors),
                "teacher_targets_tree": targets_tree_sha256,
                "groups": group_inputs,
            }
        )
        with _phase(paths, "publish", input_fingerprint) as (state, should_run):
            if not should_run:
                return
            table = _global_epoch_table(paths)
            selected_epoch = _selected_epoch(table)
            for group_index in range(FIXED_CONTRACT["student_groups"]["count"]):
                epoch_path = (
                    paths.group(group_index)
                    / "epochs"
                    / f"epoch_{selected_epoch:02d}.safetensors"
                )
                if not epoch_path.is_file():
                    raise BuildError(f"missing selected epoch checkpoint: {epoch_path}")
            selection_payload = {
                "selection_rule": FIXED_CONTRACT["selection"],
                "selected_epoch": selected_epoch,
                "selected_validation_cosine": table[selected_epoch - 1][
                    "validation_cosine"
                ],
                "epochs": table,
            }
            _atomic_write_json(paths.selection, selection_payload)
            artifact_dir = pathlib.Path(contract["artifact_dir"])
            expected = _artifact_expected_kwargs(contract, paths)
            bridge = _build_published_bridge(paths, selected_epoch)
            stage1_metadata = {
                "builder_schema_version": SCHEMA_VERSION,
                "contract": FIXED_CONTRACT,
                "contract_sha256": _contract_hash(paths),
                "corpus_sha256": _sha256_file(paths.rollouts),
                "split_sha256": _sha256_file(paths.split),
                "pca_sha256": _sha256_file(paths.pca_tensors),
                "teacher_targets_tree_sha256": targets_tree_sha256,
                "teacher_targets_index_sha256": _sha256_file(paths.targets_index),
                "selected_epoch": selected_epoch,
                "selected_validation_cosine": table[selected_epoch - 1][
                    "validation_cosine"
                ],
                "global_epoch_metrics": table,
                "group_inputs": group_inputs,
            }
            corpus_metadata = _load_json(paths.corpus_metadata)
            if paths.publish_provenance.exists():
                committed_provenance = _load_json(paths.publish_provenance)
                if (
                    not isinstance(committed_provenance, dict)
                    or committed_provenance.get("input_fingerprint")
                    != input_fingerprint
                    or not isinstance(committed_provenance.get("payload"), dict)
                ):
                    raise BuildError(
                        "committed publish provenance is incompatible with this build"
                    )
                provenance_payload = committed_provenance["payload"]
                if provenance_payload.get("publish_runtime") != _runtime_versions():
                    raise BuildError(
                        "publish runtime changed after provenance commitment"
                    )
            else:
                provenance_payload = {
                    "source": _source_provenance(),
                    "publish_runtime": _runtime_versions(),
                    "identity": contract["identity"],
                    "pca_metadata": _load_json(paths.pca_metadata),
                    "teacher_targets_index_sha256": _sha256_file(
                        paths.targets_index
                    ),
                    "teacher_targets_tree_sha256": targets_tree_sha256,
                    "group_inputs": group_inputs,
                }
                _atomic_write_json(
                    paths.publish_provenance,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "input_fingerprint": input_fingerprint,
                        "payload": provenance_payload,
                    },
                )
            sidecars = {
                "stage1_metrics.json": {
                    **selection_payload,
                    "group_metrics": {
                        str(group_index): _load_json(
                            paths.group(group_index) / "metrics.json"
                        )
                        for group_index in range(
                            FIXED_CONTRACT["student_groups"]["count"]
                        )
                    },
                },
                "corpus_ref.json": {
                    "corpus_id": _corpus_id(paths),
                    "corpus_metadata": corpus_metadata,
                    "rollouts_sha256": _sha256_file(paths.rollouts),
                    "split_sha256": _sha256_file(paths.split),
                    "split": _load_json(paths.split),
                },
                "build_config.json": contract,
                "provenance.json": provenance_payload,
            }
            if artifact_dir.exists():
                # A crash can occur after atomic rename but before state.json updates.
                _assert_exact_artifact(
                    artifact_dir,
                    bridge,
                    stage1_metadata,
                    sidecars,
                    expected,
                )
            else:
                try:
                    publish_bridge_artifact(
                        artifact_dir,
                        bridge,
                        student_model_id=contract["identity"]["student_model_id"],
                        student_model_revision=contract["identity"][
                            "student_model_revision"
                        ],
                        teacher_model_id=contract["identity"]["teacher_model_id"],
                        teacher_model_revision=contract["identity"][
                            "teacher_model_revision"
                        ],
                        tokenizer_id=contract["identity"]["tokenizer_id"],
                        tokenizer_revision=contract["identity"]["tokenizer_revision"],
                        corpus_id=_corpus_id(paths),
                        stage1_contract_sha256=contract[
                            "stage1_contract_sha256"
                        ],
                        layer_map=_layer_map(),
                        stage1_metadata=stage1_metadata,
                        extra_json_files=sidecars,
                    )
                except ArtifactAlreadyExistsError:
                    _assert_exact_artifact(
                        artifact_dir,
                        bridge,
                        stage1_metadata,
                        sidecars,
                        expected,
                    )
            _finish_phase(
                paths,
                state,
                "publish",
                [paths.selection, paths.publish_provenance, artifact_dir],
                {
                    "artifact_dir": os.fspath(artifact_dir),
                    "selected_epoch": selected_epoch,
                    "selected_validation_cosine": table[selected_epoch - 1][
                        "validation_cosine"
                    ],
                },
            )
    print(
        f"published immutable bridge at global best epoch {selected_epoch}: {artifact_dir}"
    )


def _required_phase_names(*, include_publish: bool) -> list[str]:
    phases = [
        "init",
        "generate-corpus",
        "fit-teacher-pca",
        "materialize-teacher-targets",
        *[
            f"train-student-group-{index}"
            for index in range(FIXED_CONTRACT["student_groups"]["count"])
        ],
    ]
    if include_publish:
        phases.append("publish")
    return phases


def _require_completed_phase_outputs(
    paths: Paths, *, include_publish: bool
) -> list[str]:
    state = _read_state(paths)
    checked: list[str] = []
    for phase_name in _required_phase_names(include_publish=include_publish):
        _require_completed_phase_output(paths, phase_name, state=state)
        checked.append(phase_name)
    return checked


def _verify_input_fingerprint(
    paths: Paths,
    contract: Mapping[str, Any],
    max_pca_orthogonality_error: float,
) -> str:
    return _fingerprint(
        {
            "contract": _contract_hash(paths),
            "rollouts": _sha256_file(paths.rollouts),
            "split": _sha256_file(paths.split),
            "pca": _sha256_file(paths.pca_tensors),
            "targets_index": _sha256_file(paths.targets_index),
            "selection": _sha256_file(paths.selection),
            "artifact": _sha256_tree(pathlib.Path(contract["artifact_dir"])),
            "max_pca_orthogonality_error": max_pca_orthogonality_error,
            "phase": "verify",
        }
    )


def command_verify(args: argparse.Namespace) -> None:
    paths = Paths(pathlib.Path(args.work_dir).expanduser().resolve())
    with _workspace_lock(paths):
        contract = _read_contract(paths)
        checked_phases = _require_completed_phase_outputs(
            paths, include_publish=True
        )
        input_fingerprint = _verify_input_fingerprint(
            paths, contract, args.max_pca_orthogonality_error
        )
        with _phase(paths, "verify", input_fingerprint) as (state, should_run):
            if not should_run:
                return
            records, split = _require_rollouts(paths)
            if len(split["train_indices"]) != FIXED_CONTRACT["train_rollouts"]:
                raise BuildError("train split has the wrong fixed rollout count")
            if len(split["validation_indices"]) != FIXED_CONTRACT[
                "validation_rollouts"
            ]:
                raise BuildError("validation split has the wrong fixed rollout count")
            if set(split["train_indices"]) & set(split["validation_indices"]):
                raise BuildError("train and validation rollout splits overlap")
            if set(split["train_indices"]) | set(split["validation_indices"]) != set(
                range(FIXED_CONTRACT["attempted_prompts"])
            ):
                raise BuildError("rollout split does not cover every attempted prompt")

            pca, pca_metadata = _load_pca(paths)
            expected_pca_rows = min(
                int(pca_metadata.get("seen_token_rows", -1)),
                FIXED_CONTRACT["teacher_pca"]["max_rows"],
            )
            if pca_metadata.get("sampled_rows") != expected_pca_rows:
                raise BuildError(
                    "teacher PCA sampled-row count does not equal min(seen rows, cap)"
                )
            sampled_row_ids = pca_metadata.get("sampled_row_ids")
            if (
                not isinstance(sampled_row_ids, list)
                or len(sampled_row_ids) != expected_pca_rows
                or pca_metadata.get("sampled_row_ids_sha256")
                != _fingerprint(sampled_row_ids)
            ):
                raise BuildError("teacher PCA sampled-row identity metadata is invalid")
            normalized_row_ids = [tuple(value) for value in sampled_row_ids]
            if len(set(normalized_row_ids)) != len(normalized_row_ids):
                raise BuildError("teacher PCA sampled-row IDs contain duplicates")
            train_set = set(split["train_indices"])
            for rollout_index, response_offset in normalized_row_ids:
                if rollout_index not in train_set:
                    raise BuildError("teacher PCA sampled a validation rollout")
                response_length = len(records[rollout_index]["output_token_ids"])
                first_kept = max(
                    0,
                    response_length - FIXED_CONTRACT["response_positions"]["k"],
                )
                if not first_kept <= response_offset < response_length:
                    raise BuildError("teacher PCA sampled-row offset is outside last-k")
            max_orthogonality_error = 0.0
            for layer in _layer_map():
                components = pca[f"components.{layer}"].float()
                gram = components @ components.T
                error = float(
                    (gram - torch.eye(FIXED_CONTRACT["rank"])).abs().max().item()
                )
                max_orthogonality_error = max(max_orthogonality_error, error)
            if max_orthogonality_error > args.max_pca_orthogonality_error:
                raise BuildError(
                    f"teacher PCA orthogonality error {max_orthogonality_error:.3e} exceeds "
                    f"{args.max_pca_orthogonality_error:.3e}"
                )

            target_entries = _target_index_by_attempt(paths)
            for index, record in enumerate(records):
                entry = target_entries[index]
                target_path = paths.targets / entry["file"]
                if _sha256_file(target_path) != entry["sha256"]:
                    raise BuildError(f"teacher target checksum failed at index {index}")
                _, rows = _response_window(record)
                if not _valid_tensor_shard(
                    target_path,
                    "targets",
                    (28, 8),
                    rows,
                ):
                    raise BuildError(f"teacher target shape failed at index {index}")

            table = _global_epoch_table(paths)
            selected_epoch = _selected_epoch(table)
            selection = _load_json(paths.selection)
            if selection.get("selected_epoch") != selected_epoch:
                raise BuildError("published epoch is not the recomputed global best epoch")
            artifact_dir = pathlib.Path(contract["artifact_dir"])
            bridge, stage1 = load_bridge_artifact(
                artifact_dir,
                device="cpu",
                **_artifact_expected_kwargs(contract, paths),
            )
            if bridge.training or any(parameter.requires_grad for parameter in bridge.parameters()):
                raise BuildError("published bridge is not frozen in evaluation mode")
            if stage1.get("selected_epoch") != selected_epoch:
                raise BuildError("artifact metadata has the wrong global selected epoch")
            if stage1.get("contract") != FIXED_CONTRACT:
                raise BuildError("artifact metadata does not embed the fixed contract")
            manifest, manifest_sha256 = read_bridge_manifest(artifact_dir)
            required_sidecars = {
                "stage1_metrics.json",
                "corpus_ref.json",
                "build_config.json",
                "provenance.json",
            }
            if set(manifest["files"]) != required_sidecars:
                raise BuildError("artifact does not contain the exact required sidecar set")
            if _load_json(artifact_dir / "build_config.json") != contract:
                raise BuildError("artifact build_config.json does not match contract.json")
            corpus_ref = _load_json(artifact_dir / "corpus_ref.json")
            if corpus_ref.get("rollouts_sha256") != _sha256_file(paths.rollouts):
                raise BuildError("artifact corpus_ref has the wrong rollout hash")
            if corpus_ref.get("split_sha256") != _sha256_file(paths.split):
                raise BuildError("artifact corpus_ref has the wrong split hash")
            stage1_metrics = _load_json(artifact_dir / "stage1_metrics.json")
            if stage1_metrics.get("selected_epoch") != selected_epoch:
                raise BuildError("artifact stage1_metrics has the wrong selected epoch")
            provenance = _load_json(artifact_dir / "provenance.json")
            if provenance.get("identity") != contract["identity"]:
                raise BuildError("artifact provenance has the wrong model/data identity")

            report = {
                "verified_at": _utc_now(),
                "status": "ok",
                "checked_phases": checked_phases,
                "attempted_prompts": len(records),
                "train_rollouts": len(split["train_indices"]),
                "validation_rollouts": len(split["validation_indices"]),
                "teacher_pca_rows": pca_metadata["sampled_rows"],
                "teacher_pca_max_orthogonality_error": max_orthogonality_error,
                "teacher_target_shards": len(target_entries),
                "student_groups": FIXED_CONTRACT["student_groups"]["count"],
                "layers_per_group": FIXED_CONTRACT["student_groups"][
                    "layers_per_group"
                ],
                "selected_global_epoch": selected_epoch,
                "selected_validation_cosine": table[selected_epoch - 1][
                    "validation_cosine"
                ],
                "artifact_dir": os.fspath(artifact_dir),
                "artifact_manifest_sha256": manifest_sha256,
                "artifact_sha256": _sha256_tree(artifact_dir),
            }
            _atomic_write_json(paths.verify_report, report)
            _finish_phase(
                paths,
                state,
                "verify",
                [paths.verify_report],
                report,
            )
    print(f"verification passed: {paths.verify_report}")


# ---------------------------------------------------------------------------
# CLI


def _add_work_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--work-dir",
        "--workspace",
        dest="work_dir",
        required=True,
        help="restartable Stage-1 workspace",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init", help="initialize and lock the fixed scientific contract"
    )
    _add_work_dir(init_parser)
    init_parser.add_argument(
        "--student-model", "--student", dest="student_model", default=DEFAULT_STUDENT
    )
    init_parser.add_argument(
        "--teacher-model", "--teacher", dest="teacher_model", default=DEFAULT_TEACHER
    )
    init_parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    init_parser.add_argument(
        "--dataset", "--prompts", dest="dataset", default=DEFAULT_DATASET
    )
    init_parser.add_argument(
        "--dataset-revision",
        default="",
        help="immutable dataset commit; empty resolves the current Hub commit",
    )
    init_parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    init_parser.add_argument("--prompt-key", default=DEFAULT_PROMPT_KEY)
    init_parser.add_argument(
        "--artifact-dir",
        "--artifact",
        dest="artifact_dir",
        default=None,
        help="new immutable artifact directory; default: WORK_DIR/bridge_artifact",
    )
    init_parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "use an eight-prompt, two-epoch non-production contract while preserving "
            "all layers, 4096-token generation, and the last-2000 position policy"
        ),
    )
    init_parser.set_defaults(function=command_init)

    generate_parser = subparsers.add_parser(
        "generate-corpus",
        help="sample exactly 2,000 DAPO prompts and generate one public-formatted rollout each",
    )
    _add_work_dir(generate_parser)
    generate_parser.add_argument("--batch-size", type=int, default=64)
    generate_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    generate_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    generate_parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="vLLM max_model_len; 0 uses the model default",
    )
    generate_parser.set_defaults(function=command_generate_corpus)

    pca_parser = subparsers.add_parser(
        "fit-teacher-pca",
        help="stream the 1,800 training rollouts into capped all-layer teacher PCA",
    )
    _add_work_dir(pca_parser)
    pca_parser.add_argument("--device", default="cuda:0")
    pca_parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    pca_parser.add_argument(
        "--pca-device",
        default="cpu",
        help="device used for the 2560x2560 symmetric eigendecompositions",
    )
    pca_parser.add_argument("--covariance-chunk-rows", type=int, default=512)
    pca_parser.set_defaults(function=command_fit_teacher_pca)

    targets_parser = subparsers.add_parser(
        "materialize-teacher-targets",
        help="write projected rank-8 teacher targets for all 2,000 rollouts",
    )
    _add_work_dir(targets_parser)
    targets_parser.add_argument("--device", default="cuda:0")
    targets_parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    targets_parser.set_defaults(function=command_materialize_teacher_targets)

    group_parser = subparsers.add_parser(
        "train-student-group",
        help="cache and train one of seven serial four-layer projector groups",
    )
    _add_work_dir(group_parser)
    group_parser.add_argument(
        "--group",
        "--group-index",
        dest="group",
        type=int,
        required=True,
        choices=range(7),
    )
    group_parser.add_argument("--device", default="cuda:0")
    group_parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    group_parser.add_argument("--batch-rows", type=int, default=2_048)
    group_parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="retain the large four-layer hidden cache after training",
    )
    group_parser.set_defaults(function=command_train_student_group)

    publish_parser = subparsers.add_parser(
        "publish",
        help="choose one global best epoch and atomically publish the frozen bridge",
    )
    _add_work_dir(publish_parser)
    publish_parser.set_defaults(function=command_publish)

    verify_parser = subparsers.add_parser(
        "verify", help="checksum and semantically verify the complete build"
    )
    _add_work_dir(verify_parser)
    verify_parser.add_argument(
        "--max-pca-orthogonality-error", type=float, default=5e-4
    )
    verify_parser.set_defaults(function=command_verify)
    return parser


def _validate_runtime_args(args: argparse.Namespace) -> None:
    for name in (
        "batch_size",
        "tensor_parallel_size",
        "covariance_chunk_rows",
        "batch_rows",
    ):
        if hasattr(args, name) and getattr(args, name) < 1:
            raise BuildError(f"--{name.replace('_', '-')} must be positive")
    if hasattr(args, "gpu_memory_utilization") and not (
        0 < args.gpu_memory_utilization <= 1
    ):
        raise BuildError("--gpu-memory-utilization must be in (0, 1]")
    if hasattr(args, "max_model_len") and args.max_model_len < 0:
        raise BuildError("--max-model-len must be nonnegative")
    if hasattr(args, "max_pca_orthogonality_error") and not (
        math.isfinite(args.max_pca_orthogonality_error)
        and args.max_pca_orthogonality_error > 0
    ):
        raise BuildError("--max-pca-orthogonality-error must be finite and positive")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_runtime_args(args)
        args.function(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.environ.get("OPRD_BRIDGE_TRACEBACK") == "1":
            traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
