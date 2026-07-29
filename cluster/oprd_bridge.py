"""Layer mapping, bridge modules, and immutable OPRD bridge artifacts.

This module deliberately depends only on PyTorch and the safetensors package used for
artifact I/O. It does not import transformers or instantiate any model or tokenizer.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pathlib
import shutil
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from oprd_atomic_directory import (
    AmbiguousPostconditionError,
    AtomicDirectoryError,
    DestinationCollisionError,
    KnownNoChangeMoveError,
    UnsafePathError,
    UnsupportedPlatformError,
    atomic_move_directory_noreplace,
    directory_matches_identity,
)


ARTIFACT_TYPE = "oprd_bridge"
SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
TENSOR_FILENAME = "bridge.safetensors"
COMPLETE_FILENAME = "COMPLETE"

PathLike = Union[str, os.PathLike[str]]

_STAGE1_PRODUCTION_CONTRACT: Dict[str, Any] = {
    "contract_name": "oprd-stage1-qwen3-4b-to-qwen3-1.7b",
    "paper": {
        "title": "OPRD: On-Policy Representation Distillation",
        "arxiv_id": "2606.06021v4",
        "version_date": "2026-06-20",
    },
    "official_code": {
        "repository": "https://github.com/ShenzhiYang2000/OPRD",
        "commit": "a63236c990b1bfa9551e9d020719490d3ecd93e8",
    },
    "attempted_prompts": 2_000,
    "prompt_seed": 42,
    "train_rollouts": 1_800,
    "validation_rollouts": 200,
    "formatting": {
        "method": "tokenizer.apply_chat_template",
        "add_generation_prompt": True,
        "enable_thinking": False,
        "local_instruction_suffix": None,
    },
    "generation": {
        "temperature": 1.0,
        "max_new_tokens": 4_096,
        "n": 1,
        "top_p": None,
        "top_k": None,
        "min_p": None,
        "filters": None,
    },
    "student": {"num_layers": 28, "hidden_size": 2_048},
    "teacher": {"num_layers": 36, "hidden_size": 2_560},
    "layer_selection": "all",
    "layer_mapping": "OPRD Eq. 11 proportional depth, 1-based blocks",
    "rank": 8,
    "response_positions": {
        "mode": "last_k",
        "k": 2_000,
        "short_response_behavior": "last min(response_length, k)",
    },
    "teacher_pca": {
        "max_rows": 16_384,
        "sampling": "shared streamed uniform reservoir over train token rows",
        "centered": True,
        "solver": "symmetric covariance eigendecomposition",
    },
    "teacher_targets": {
        "projected_only": True,
        "dtype": "bfloat16",
        "rank": 8,
    },
    "student_groups": {
        "count": 7,
        "layers_per_group": 4,
        "execution": "serial",
    },
    "optimizer": {
        "name": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "epochs": 20,
        "batching": {
            "max_rollouts": 4,
            "max_padded_tokens": 65_536,
            "cost": "batch_size * maximum full sequence length",
            "ordering": "stable descending full sequence length",
            "reuse_plan_each_epoch": True,
        },
    },
    "selection": {
        "metric": "global token-layer mean validation cosine",
        "scope": "one epoch shared by all 28 projectors",
        "tie_break": "earliest epoch",
    },
}


def canonical_stage1_contract(*, smoke: bool = False) -> Dict[str, Any]:
    contract = copy.deepcopy(_STAGE1_PRODUCTION_CONTRACT)
    if smoke:
        contract["contract_name"] += "-smoke"
        contract["attempted_prompts"] = 8
        contract["train_rollouts"] = 6
        contract["validation_rollouts"] = 2
        contract["teacher_pca"]["max_rows"] = 64
        contract["optimizer"]["epochs"] = 2
    return contract


def stage1_contract_sha256(*, smoke: bool = False) -> str:
    payload = json.dumps(
        canonical_stage1_contract(smoke=smoke),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class OPRDBridgeError(Exception):
    """Base class for bridge artifact errors."""


class ArtifactDependencyError(OPRDBridgeError, ImportError):
    """A required artifact serialization dependency is unavailable."""


class ArtifactAlreadyExistsError(OPRDBridgeError, FileExistsError):
    """The requested immutable artifact destination already exists."""


class ArtifactIncompleteError(OPRDBridgeError):
    """An artifact was not completely published."""


class ArtifactCorruptError(OPRDBridgeError):
    """An artifact failed structural, checksum, or tensor validation."""


class ArtifactMismatchError(OPRDBridgeError, ValueError):
    """An artifact does not match the requested training identity or shape."""


class ArtifactValidationError(OPRDBridgeError, ValueError):
    """Values supplied while creating an artifact are invalid."""


def proportional_teacher_layer(
    student_layer: int,
    num_student_layers: int,
    num_teacher_layers: int,
) -> int:
    """Map a 1-based student block to a 1-based teacher block using paper Eq. 11."""
    if num_student_layers < 1 or num_teacher_layers < 1:
        raise ValueError("model layer counts must be positive")
    if not 1 <= student_layer <= num_student_layers:
        raise ValueError(
            f"student layer {student_layer} outside 1..{num_student_layers}"
        )
    if num_student_layers == 1:
        return num_teacher_layers
    return round(
        (student_layer - 1)
        * (num_teacher_layers - 1)
        / (num_student_layers - 1)
    ) + 1


def resolve_layer_map(
    layer_spec: str,
    layer_fracs: str,
    num_student_layers: int,
    num_teacher_layers: int,
) -> Dict[int, int]:
    """Resolve selected student blocks, then map them with paper Eq. 11."""
    if layer_spec and layer_fracs:
        raise ValueError(
            "use only one of --supervised-layers and --supervised-layer-fracs"
        )
    if layer_spec:
        if layer_spec.strip().lower() == "all":
            layers = list(range(1, num_student_layers + 1))
        else:
            try:
                layers = [
                    int(value.strip())
                    for value in layer_spec.split(",")
                    if value.strip()
                ]
            except ValueError as exc:
                raise ValueError(
                    "--supervised-layers must be 'all' or comma-separated integers"
                ) from exc
    else:
        fraction_spec = layer_fracs or "0.25,0.5"
        try:
            fractions = [
                float(value.strip())
                for value in fraction_spec.split(",")
                if value.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "--supervised-layer-fracs must contain comma-separated numbers"
            ) from exc
        if not fractions or any(
            not math.isfinite(fraction) or not 0 < fraction <= 1
            for fraction in fractions
        ):
            raise ValueError(
                "supervised layer fractions must be finite values in (0, 1]"
            )
        layers = [max(1, round(fraction * num_student_layers)) for fraction in fractions]

    if not layers:
        raise ValueError("at least one supervised layer is required")
    if len(set(layers)) != len(layers):
        raise ValueError(f"supervised layer selection resolves to duplicates: {layers}")
    if any(not 1 <= layer <= num_student_layers for layer in layers):
        raise ValueError(
            f"supervised layers must be within 1..{num_student_layers}: {layers}"
        )

    return {
        layer: proportional_teacher_layer(
            layer, num_student_layers, num_teacher_layers
        )
        for layer in sorted(layers)
    }


class BridgeBank(torch.nn.Module):
    """Per-student-layer projectors plus teacher means and PCA directions."""

    def __init__(
        self,
        layers: Any,
        student_dim: int,
        teacher_dim: int,
        rank: int,
    ) -> None:
        super().__init__()
        normalized_layers = list(layers)
        if not normalized_layers:
            raise ValueError("at least one bridge layer is required")
        if any(not isinstance(layer, int) or isinstance(layer, bool) or layer < 1
               for layer in normalized_layers):
            raise ValueError("bridge layers must be positive integers")
        if len(set(normalized_layers)) != len(normalized_layers):
            raise ValueError("bridge layers must be unique")
        if student_dim < 1 or teacher_dim < 1:
            raise ValueError("student and teacher dimensions must be positive")
        if rank < 1 or rank > min(student_dim, teacher_dim):
            raise ValueError(
                f"rank must be within 1..{min(student_dim, teacher_dim)}"
            )

        self.layers = normalized_layers
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.rank = rank
        self.student_projectors = torch.nn.ModuleDict(
            {
                str(layer): torch.nn.Linear(student_dim, rank, bias=False)
                for layer in self.layers
            }
        )
        for layer in self.layers:
            self.register_buffer(f"teacher_mean_{layer}", torch.zeros(teacher_dim))
            self.register_buffer(
                f"teacher_pca_{layer}", torch.zeros(rank, teacher_dim)
            )

    def _validate_layer(self, layer: int) -> None:
        if layer not in self.layers:
            raise ValueError(f"layer {layer} is not present in this bridge")

    @torch.no_grad()
    def set_teacher_projection(
        self,
        layer: int,
        mean: torch.Tensor,
        pca: torch.Tensor,
    ) -> None:
        self._validate_layer(layer)
        expected_mean = (self.teacher_dim,)
        expected_pca = (self.rank, self.teacher_dim)
        if tuple(mean.shape) != expected_mean:
            raise ValueError(
                f"teacher mean for layer {layer} has shape {tuple(mean.shape)}, "
                f"expected {expected_mean}"
            )
        if tuple(pca.shape) != expected_pca:
            raise ValueError(
                f"teacher PCA for layer {layer} has shape {tuple(pca.shape)}, "
                f"expected {expected_pca}"
            )
        getattr(self, f"teacher_mean_{layer}").copy_(mean)
        getattr(self, f"teacher_pca_{layer}").copy_(pca)

    def project_student(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        self._validate_layer(layer)
        if hidden.shape[-1] != self.student_dim:
            raise ValueError(
                f"student hidden width is {hidden.shape[-1]}, expected {self.student_dim}"
            )
        return self.student_projectors[str(layer)](hidden)

    def project_teacher(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        self._validate_layer(layer)
        if hidden.shape[-1] != self.teacher_dim:
            raise ValueError(
                f"teacher hidden width is {hidden.shape[-1]}, expected {self.teacher_dim}"
            )
        mean = getattr(self, f"teacher_mean_{layer}")
        pca = getattr(self, f"teacher_pca_{layer}")
        return (hidden - mean) @ pca.T

    def freeze(self) -> "BridgeBank":
        """Permanently disable gradients for the current bridge parameters and buffers."""
        self.requires_grad_(False)
        self.eval()
        return self


def _require_nonempty_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{name} must be a non-empty string")
    return value


def _normalize_layer_map(
    layer_map: Mapping[int, int],
    *,
    error_type: type[Exception] = ArtifactValidationError,
) -> Dict[int, int]:
    if not isinstance(layer_map, Mapping) or not layer_map:
        raise error_type("layer map must be a non-empty mapping")
    normalized: Dict[int, int] = {}
    for student_layer, teacher_layer in layer_map.items():
        if (
            not isinstance(student_layer, int)
            or isinstance(student_layer, bool)
            or student_layer < 1
            or not isinstance(teacher_layer, int)
            or isinstance(teacher_layer, bool)
            or teacher_layer < 1
        ):
            raise error_type("layer map keys and values must be positive integers")
        normalized[student_layer] = teacher_layer
    return dict(sorted(normalized.items()))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    try:
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
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(
            f"artifact metadata must be finite and JSON serializable: {exc}"
        ) from exc


def _safetensors_functions() -> Tuple[Any, Any]:
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise ArtifactDependencyError(
            "safetensors is required for OPRD bridge artifact I/O"
        ) from exc
    return load_file, save_file


def _fsync_file(path: pathlib.Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


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


def _path_exists(path: pathlib.Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _manifest_layer_map(layer_map: Mapping[int, int]) -> Dict[str, int]:
    return {str(student): teacher for student, teacher in layer_map.items()}


def _require_finite_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    error_type: type[Exception],
    context: str,
) -> None:
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise error_type(f"{context} contains a non-tensor value: {name}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise error_type(f"{context} contains a non-finite tensor: {name}")


def _rename_directory_noreplace(source: pathlib.Path, destination: pathlib.Path) -> None:
    source_path = pathlib.Path(os.path.abspath(os.fspath(source)))
    destination_path = pathlib.Path(os.path.abspath(os.fspath(destination)))
    if source_path.parent != destination_path.parent:
        raise ArtifactValidationError(
            "bridge artifact staging and destination directories must be siblings"
        )
    try:
        atomic_move_directory_noreplace(
            source_path,
            destination_path,
            trusted_root=destination_path.parent,
        )
    except DestinationCollisionError as exc:
        raise ArtifactAlreadyExistsError(
            f"refusing to overwrite existing bridge artifact: {destination}"
        ) from exc
    except AmbiguousPostconditionError as exc:
        raise ArtifactValidationError(
            "atomic bridge artifact promotion has an ambiguous postcondition; "
            f"automatic staging cleanup was skipped: {exc}"
        ) from exc
    except (
        KnownNoChangeMoveError,
        UnsafePathError,
        UnsupportedPlatformError,
    ) as exc:
        raise ArtifactValidationError(
            f"cannot atomically publish bridge artifact {destination}: {exc}"
        ) from exc
    except AtomicDirectoryError as exc:
        raise ArtifactValidationError(
            f"atomic bridge artifact publication failed for {destination}: {exc}"
        ) from exc


def _require_exact_serialized_tensors(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    *,
    context: str,
) -> None:
    if set(actual) != set(expected):
        raise ArtifactCorruptError(
            f"{context} bridge tensor names changed during serialization"
        )
    for name, tensor in expected.items():
        if not torch.equal(actual[name], tensor):
            raise ArtifactCorruptError(
                f"{context} bridge tensor changed during serialization: {name}"
            )


def publish_bridge_artifact(
    artifact_dir: PathLike,
    bridge: BridgeBank,
    *,
    student_model_id: str,
    student_model_revision: str,
    teacher_model_id: str,
    teacher_model_revision: str,
    tokenizer_id: str,
    tokenizer_revision: str,
    corpus_id: str,
    stage1_contract_sha256: str,
    layer_map: Mapping[int, int],
    stage1_metadata: Optional[Mapping[str, Any]] = None,
    extra_json_files: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> pathlib.Path:
    """Atomically publish a new immutable bridge artifact directory.

    The destination must not exist. All files are written and synced in a sibling
    temporary directory, with ``COMPLETE`` written last. Promotion uses a native
    no-replace rename when available. On filesystems that reject the native operation,
    the fallback briefly reserves the destination with an empty private directory
    before atomically renaming the completed staging directory over that reservation.
    """
    if not isinstance(bridge, BridgeBank):
        raise ArtifactValidationError("bridge must be a BridgeBank")
    normalized_map = _normalize_layer_map(layer_map)
    if list(normalized_map) != bridge.layers:
        raise ArtifactValidationError(
            f"layer map keys {list(normalized_map)} do not match bridge layers "
            f"{bridge.layers}"
        )

    identity = {
        "student_model_id": _require_nonempty_string(
            student_model_id, "student model id"
        ),
        "student_model_revision": _require_nonempty_string(
            student_model_revision, "student model revision"
        ),
        "teacher_model_id": _require_nonempty_string(
            teacher_model_id, "teacher model id"
        ),
        "teacher_model_revision": _require_nonempty_string(
            teacher_model_revision, "teacher model revision"
        ),
        "tokenizer_id": _require_nonempty_string(tokenizer_id, "tokenizer id"),
        "tokenizer_revision": _require_nonempty_string(
            tokenizer_revision, "tokenizer revision"
        ),
        "corpus_id": _require_nonempty_string(corpus_id, "corpus id"),
        "stage1_contract_sha256": _require_nonempty_string(
            stage1_contract_sha256, "Stage-1 contract SHA-256"
        ),
    }
    if (
        len(identity["stage1_contract_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in identity["stage1_contract_sha256"]
        )
    ):
        raise ArtifactValidationError("Stage-1 contract SHA-256 is invalid")
    if stage1_metadata is not None and not isinstance(stage1_metadata, Mapping):
        raise ArtifactValidationError("stage1 metadata must be a mapping")
    metadata = dict(stage1_metadata or {})
    if extra_json_files is not None and not isinstance(extra_json_files, Mapping):
        raise ArtifactValidationError("extra JSON files must be a mapping")
    normalized_extra: Dict[str, Mapping[str, Any]] = {}
    for filename, value in dict(extra_json_files or {}).items():
        if (
            not isinstance(filename, str)
            or pathlib.PurePosixPath(filename).name != filename
            or not filename.endswith(".json")
            or filename in {MANIFEST_FILENAME}
        ):
            raise ArtifactValidationError(
                f"invalid artifact sidecar filename: {filename!r}"
            )
        if not isinstance(value, Mapping):
            raise ArtifactValidationError(
                f"artifact sidecar {filename!r} must contain a JSON object"
            )
        normalized_extra[filename] = value
    destination = pathlib.Path(artifact_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _path_exists(destination):
        raise ArtifactAlreadyExistsError(
            f"bridge artifact destination already exists: {destination}"
        )

    _, save_file = _safetensors_functions()
    temporary_name = tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-", dir=os.fspath(destination.parent)
    )
    temporary_stat = os.lstat(temporary_name)
    temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
    temporary = pathlib.Path(temporary_name)
    preserve_staging = False
    try:
        tensor_path = temporary / TENSOR_FILENAME
        tensors = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in bridge.state_dict().items()
        }
        _require_finite_tensors(
            tensors,
            error_type=ArtifactValidationError,
            context="bridge state",
        )
        save_file(tensors, os.fspath(tensor_path))
        _fsync_file(tensor_path)

        tensor_digest = _sha256_file(tensor_path)
        sidecars: Dict[str, Dict[str, Any]] = {}
        for filename, value in sorted(normalized_extra.items()):
            sidecar_path = temporary / filename
            sidecar_bytes = _canonical_json(value)
            sidecar_path.write_bytes(sidecar_bytes)
            _fsync_file(sidecar_path)
            sidecars[filename] = {
                "sha256": _sha256_bytes(sidecar_bytes),
                "size_bytes": len(sidecar_bytes),
            }
        manifest: Dict[str, Any] = {
            "artifact_type": ARTIFACT_TYPE,
            "schema_version": SCHEMA_VERSION,
            "storage": {
                "format": "safetensors",
                "filename": TENSOR_FILENAME,
                "sha256": tensor_digest,
                "size_bytes": tensor_path.stat().st_size,
            },
            "identity": identity,
            "bridge": {
                "layers": list(bridge.layers),
                "layer_map": _manifest_layer_map(normalized_map),
                "rank": bridge.rank,
                "student_dim": bridge.student_dim,
                "teacher_dim": bridge.teacher_dim,
            },
            "stage1": metadata,
            "files": sidecars,
        }
        manifest_bytes = _canonical_json(manifest)
        manifest_path = temporary / MANIFEST_FILENAME
        manifest_path.write_bytes(manifest_bytes)
        _fsync_file(manifest_path)

        manifest_digest = _sha256_bytes(manifest_bytes)
        _, _, _, _, staged_tensors = _load_bridge_artifact_common(
            temporary,
            expected_student_model_id=identity["student_model_id"],
            expected_student_model_revision=identity["student_model_revision"],
            expected_teacher_model_id=identity["teacher_model_id"],
            expected_teacher_model_revision=identity["teacher_model_revision"],
            expected_tokenizer_id=identity["tokenizer_id"],
            expected_tokenizer_revision=identity["tokenizer_revision"],
            expected_stage1_contract_sha256=identity["stage1_contract_sha256"],
            expected_corpus_id=identity["corpus_id"],
            expected_layer_map=normalized_map,
            expected_rank=bridge.rank,
            expected_student_dim=bridge.student_dim,
            expected_teacher_dim=bridge.teacher_dim,
            device="cpu",
            require_complete=False,
            expected_manifest_sha256=manifest_digest,
        )
        _require_exact_serialized_tensors(
            staged_tensors,
            tensors,
            context="staged",
        )
        del staged_tensors

        complete_path = temporary / COMPLETE_FILENAME
        complete_path.write_text(f"manifest_sha256 {manifest_digest}\n", encoding="ascii")
        _fsync_file(complete_path)

        preserve_staging = True
        _rename_directory_noreplace(temporary, destination)

        _, _, _, _, canonical_tensors = _load_bridge_artifact_common(
            destination,
            expected_student_model_id=identity["student_model_id"],
            expected_student_model_revision=identity["student_model_revision"],
            expected_teacher_model_id=identity["teacher_model_id"],
            expected_teacher_model_revision=identity["teacher_model_revision"],
            expected_tokenizer_id=identity["tokenizer_id"],
            expected_tokenizer_revision=identity["tokenizer_revision"],
            expected_stage1_contract_sha256=identity["stage1_contract_sha256"],
            expected_corpus_id=identity["corpus_id"],
            expected_layer_map=normalized_map,
            expected_rank=bridge.rank,
            expected_student_dim=bridge.student_dim,
            expected_teacher_dim=bridge.teacher_dim,
            device="cpu",
            require_complete=True,
            expected_manifest_sha256=manifest_digest,
        )
        _require_exact_serialized_tensors(
            canonical_tensors,
            tensors,
            context="published",
        )
        return destination
    finally:
        if (
            not preserve_staging
            and directory_matches_identity(temporary, temporary_identity)
        ):
            shutil.rmtree(temporary, ignore_errors=True)


def _read_complete_digest(path: pathlib.Path) -> str:
    try:
        content = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ArtifactIncompleteError(f"cannot read COMPLETE marker: {exc}") from exc
    parts = content.strip().split()
    if len(parts) != 2 or parts[0] != "manifest_sha256":
        raise ArtifactCorruptError("COMPLETE marker has an invalid format")
    digest = parts[1]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ArtifactCorruptError("COMPLETE marker has an invalid SHA-256 digest")
    return digest


def _read_manifest(
    artifact_dir: pathlib.Path,
    *,
    require_complete: bool = True,
    expected_manifest_sha256: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    if not artifact_dir.is_dir() or artifact_dir.is_symlink():
        raise ArtifactIncompleteError(
            f"bridge artifact directory is missing or invalid: {artifact_dir}"
        )
    complete_path = artifact_dir / COMPLETE_FILENAME
    manifest_path = artifact_dir / MANIFEST_FILENAME
    tensor_path = artifact_dir / TENSOR_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ArtifactIncompleteError("bridge artifact has no valid manifest.json")
    if not tensor_path.is_file() or tensor_path.is_symlink():
        raise ArtifactIncompleteError(
            f"bridge artifact has no valid {TENSOR_FILENAME}"
        )

    if require_complete:
        if not complete_path.is_file() or complete_path.is_symlink():
            raise ArtifactIncompleteError("bridge artifact has no valid COMPLETE marker")
        authenticated_digest = _read_complete_digest(complete_path)
        if (
            expected_manifest_sha256 is not None
            and authenticated_digest != expected_manifest_sha256
        ):
            raise ArtifactCorruptError(
                "COMPLETE marker does not match the expected manifest SHA-256"
            )
    else:
        if expected_manifest_sha256 is None:
            raise ArtifactValidationError(
                "staged artifact validation requires an expected manifest SHA-256"
            )
        if _path_exists(complete_path):
            raise ArtifactCorruptError(
                "staged artifact unexpectedly contains a COMPLETE marker"
            )
        authenticated_digest = expected_manifest_sha256

    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise ArtifactIncompleteError(f"cannot read manifest.json: {exc}") from exc
    actual_manifest_digest = _sha256_bytes(manifest_bytes)
    if actual_manifest_digest != authenticated_digest:
        source = "COMPLETE marker" if require_complete else "publisher digest"
        raise ArtifactCorruptError(
            f"manifest.json SHA-256 does not match the {source}"
        )
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactCorruptError(f"manifest.json is invalid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArtifactCorruptError("manifest.json must contain a JSON object")
    return manifest, actual_manifest_digest


def _manifest_value(manifest: Mapping[str, Any], section: str, field: str) -> Any:
    section_value = manifest.get(section)
    if not isinstance(section_value, dict) or field not in section_value:
        raise ArtifactCorruptError(
            f"manifest.json is missing required field {section}.{field}"
        )
    return section_value[field]


def _validate_manifest_structure(manifest: Mapping[str, Any]) -> Dict[int, int]:
    if manifest.get("artifact_type") != ARTIFACT_TYPE:
        raise ArtifactCorruptError(
            f"unsupported artifact type: {manifest.get('artifact_type')!r}"
        )
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ArtifactCorruptError(
            f"unsupported bridge schema version: {manifest.get('schema_version')!r}"
        )

    storage = manifest.get("storage")
    if not isinstance(storage, dict):
        raise ArtifactCorruptError("manifest.json storage section is invalid")
    if storage.get("format") != "safetensors" or storage.get("filename") != TENSOR_FILENAME:
        raise ArtifactCorruptError("manifest.json declares an unsupported tensor format")
    digest = storage.get("sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ArtifactCorruptError("manifest.json has an invalid tensor SHA-256")
    size = storage.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ArtifactCorruptError("manifest.json has an invalid tensor size")

    layers = _manifest_value(manifest, "bridge", "layers")
    raw_map = _manifest_value(manifest, "bridge", "layer_map")
    if (
        not isinstance(layers, list)
        or not layers
        or any(not isinstance(layer, int) or isinstance(layer, bool) or layer < 1
               for layer in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ArtifactCorruptError("manifest.json bridge.layers is invalid")
    if not isinstance(raw_map, dict):
        raise ArtifactCorruptError("manifest.json bridge.layer_map is invalid")
    try:
        converted_map = {
            int(student): teacher for student, teacher in raw_map.items()
        }
    except (TypeError, ValueError) as exc:
        raise ArtifactCorruptError("manifest.json bridge.layer_map is invalid") from exc
    normalized_map = _normalize_layer_map(
        converted_map, error_type=ArtifactCorruptError
    )
    if list(normalized_map) != layers:
        raise ArtifactCorruptError(
            "manifest bridge.layers and bridge.layer_map keys disagree"
        )

    for field in ("rank", "student_dim", "teacher_dim"):
        value = _manifest_value(manifest, "bridge", field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ArtifactCorruptError(
                f"manifest.json bridge.{field} must be a positive integer"
            )
    rank = _manifest_value(manifest, "bridge", "rank")
    student_dim = _manifest_value(manifest, "bridge", "student_dim")
    teacher_dim = _manifest_value(manifest, "bridge", "teacher_dim")
    if rank > min(student_dim, teacher_dim):
        raise ArtifactCorruptError("manifest bridge rank exceeds a bridge dimension")

    for field in (
        "student_model_id",
        "student_model_revision",
        "teacher_model_id",
        "teacher_model_revision",
        "tokenizer_id",
        "tokenizer_revision",
        "corpus_id",
        "stage1_contract_sha256",
    ):
        value = _manifest_value(manifest, "identity", field)
        if not isinstance(value, str) or not value.strip():
            raise ArtifactCorruptError(
                f"manifest.json identity.{field} must be a non-empty string"
            )
    contract_digest = manifest["identity"]["stage1_contract_sha256"]
    if (
        len(contract_digest) != 64
        or any(character not in "0123456789abcdef" for character in contract_digest)
    ):
        raise ArtifactCorruptError(
            "manifest.json identity.stage1_contract_sha256 is invalid"
        )
    if not isinstance(manifest.get("stage1"), dict):
        raise ArtifactCorruptError("manifest.json stage1 must be a JSON object")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ArtifactCorruptError("manifest.json files must be a JSON object")
    for filename, metadata in files.items():
        if (
            not isinstance(filename, str)
            or pathlib.PurePosixPath(filename).name != filename
            or not filename.endswith(".json")
            or filename == MANIFEST_FILENAME
            or not isinstance(metadata, dict)
        ):
            raise ArtifactCorruptError("manifest.json has an invalid sidecar entry")
        digest = metadata.get("sha256")
        size = metadata.get("size_bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
        ):
            raise ArtifactCorruptError(
                f"manifest.json has invalid sidecar metadata for {filename}"
            )
    return normalized_map


def _check_match(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ArtifactMismatchError(
            f"bridge {name} mismatch: artifact={actual!r}, expected={expected!r}"
        )


def read_bridge_manifest(
    artifact_dir: PathLike,
) -> Tuple[Dict[str, Any], str]:
    """Read and structurally validate a completed artifact manifest.

    Tensor bytes are checksum-validated by :func:`load_bridge_artifact`; this helper is
    intended for recording immutable artifact identity before or after a full load.
    """
    directory = pathlib.Path(artifact_dir)
    manifest, manifest_digest = _read_manifest(directory)
    _validate_manifest_structure(manifest)
    return manifest, manifest_digest


def _load_bridge_artifact_common(
    artifact_dir: PathLike,
    *,
    expected_student_model_id: str,
    expected_student_model_revision: str,
    expected_teacher_model_id: str,
    expected_teacher_model_revision: str,
    expected_tokenizer_id: str,
    expected_tokenizer_revision: str,
    expected_stage1_contract_sha256: str,
    expected_layer_map: Mapping[int, int],
    expected_rank: int,
    expected_student_dim: int,
    expected_teacher_dim: int,
    expected_corpus_id: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    require_complete: bool = True,
    expected_manifest_sha256: Optional[str] = None,
) -> Tuple[
    BridgeBank,
    Dict[str, Any],
    Dict[str, Any],
    str,
    Dict[str, torch.Tensor],
]:
    directory = pathlib.Path(artifact_dir)
    manifest, manifest_digest = _read_manifest(
        directory,
        require_complete=require_complete,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    manifest_map = _validate_manifest_structure(manifest)
    for filename, metadata in manifest["files"].items():
        sidecar = directory / filename
        if not sidecar.is_file() or sidecar.is_symlink():
            raise ArtifactIncompleteError(
                f"bridge artifact sidecar is missing or invalid: {filename}"
            )
        try:
            sidecar_size = sidecar.stat().st_size
        except OSError as exc:
            raise ArtifactIncompleteError(
                f"cannot stat bridge artifact sidecar {filename}: {exc}"
            ) from exc
        if sidecar_size != metadata["size_bytes"]:
            raise ArtifactCorruptError(
                f"bridge artifact sidecar size mismatch: {filename}"
            )
        if _sha256_file(sidecar) != metadata["sha256"]:
            raise ArtifactCorruptError(
                f"bridge artifact sidecar SHA-256 mismatch: {filename}"
            )
    expected_map = _normalize_layer_map(
        expected_layer_map, error_type=ArtifactMismatchError
    )

    identity = manifest["identity"]
    bridge_data = manifest["bridge"]
    _check_match("student model", identity["student_model_id"], expected_student_model_id)
    _check_match(
        "student model revision",
        identity["student_model_revision"],
        expected_student_model_revision,
    )
    _check_match("teacher model", identity["teacher_model_id"], expected_teacher_model_id)
    _check_match(
        "teacher model revision",
        identity["teacher_model_revision"],
        expected_teacher_model_revision,
    )
    _check_match("tokenizer", identity["tokenizer_id"], expected_tokenizer_id)
    _check_match(
        "tokenizer revision",
        identity["tokenizer_revision"],
        expected_tokenizer_revision,
    )
    if expected_corpus_id is not None:
        _check_match("corpus", identity["corpus_id"], expected_corpus_id)
    _check_match(
        "Stage-1 contract",
        identity["stage1_contract_sha256"],
        expected_stage1_contract_sha256,
    )
    _check_match("layer map", manifest_map, expected_map)
    _check_match("rank", bridge_data["rank"], expected_rank)
    _check_match("student dimension", bridge_data["student_dim"], expected_student_dim)
    _check_match("teacher dimension", bridge_data["teacher_dim"], expected_teacher_dim)

    tensor_path = directory / TENSOR_FILENAME
    storage = manifest["storage"]
    try:
        actual_size = tensor_path.stat().st_size
    except OSError as exc:
        raise ArtifactIncompleteError(f"cannot stat {TENSOR_FILENAME}: {exc}") from exc
    if actual_size != storage["size_bytes"]:
        raise ArtifactCorruptError(
            f"{TENSOR_FILENAME} size does not match manifest.json"
        )
    if _sha256_file(tensor_path) != storage["sha256"]:
        raise ArtifactCorruptError(
            f"{TENSOR_FILENAME} SHA-256 does not match manifest.json"
        )

    load_file, _ = _safetensors_functions()
    try:
        state_dict = load_file(os.fspath(tensor_path), device="cpu")
    except Exception as exc:
        raise ArtifactCorruptError(
            f"cannot load validated {TENSOR_FILENAME}: {exc}"
        ) from exc
    _require_finite_tensors(
        state_dict,
        error_type=ArtifactCorruptError,
        context="serialized bridge state",
    )

    bridge = BridgeBank(
        bridge_data["layers"],
        bridge_data["student_dim"],
        bridge_data["teacher_dim"],
        bridge_data["rank"],
    )
    try:
        bridge.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise ArtifactCorruptError(
            f"bridge tensor names or shapes do not match manifest.json: {exc}"
        ) from exc
    bridge.to(device).float()
    _require_finite_tensors(
        bridge.state_dict(),
        error_type=ArtifactCorruptError,
        context="loaded float32 bridge state",
    )
    bridge.freeze()
    return (
        bridge,
        dict(manifest["stage1"]),
        dict(manifest),
        manifest_digest,
        dict(state_dict),
    )


def load_bridge_artifact(
    artifact_dir: PathLike,
    *,
    expected_student_model_id: str,
    expected_student_model_revision: str,
    expected_teacher_model_id: str,
    expected_teacher_model_revision: str,
    expected_tokenizer_id: str,
    expected_tokenizer_revision: str,
    expected_stage1_contract_sha256: str,
    expected_layer_map: Mapping[int, int],
    expected_rank: int,
    expected_student_dim: int,
    expected_teacher_dim: int,
    expected_corpus_id: Optional[str] = None,
    expected_manifest_sha256: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[BridgeBank, Dict[str, Any]]:
    """Validate and load a bridge artifact as an evaluation-only frozen module."""
    bridge, stage1, _, _, _ = _load_bridge_artifact_common(
        artifact_dir,
        expected_student_model_id=expected_student_model_id,
        expected_student_model_revision=expected_student_model_revision,
        expected_teacher_model_id=expected_teacher_model_id,
        expected_teacher_model_revision=expected_teacher_model_revision,
        expected_tokenizer_id=expected_tokenizer_id,
        expected_tokenizer_revision=expected_tokenizer_revision,
        expected_stage1_contract_sha256=expected_stage1_contract_sha256,
        expected_layer_map=expected_layer_map,
        expected_rank=expected_rank,
        expected_student_dim=expected_student_dim,
        expected_teacher_dim=expected_teacher_dim,
        expected_corpus_id=expected_corpus_id,
        expected_manifest_sha256=expected_manifest_sha256,
        device=device,
    )
    return bridge, stage1


save_bridge_artifact = publish_bridge_artifact


__all__ = [
    "ARTIFACT_TYPE",
    "SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "TENSOR_FILENAME",
    "COMPLETE_FILENAME",
    "OPRDBridgeError",
    "ArtifactDependencyError",
    "ArtifactAlreadyExistsError",
    "ArtifactIncompleteError",
    "ArtifactCorruptError",
    "ArtifactMismatchError",
    "ArtifactValidationError",
    "canonical_stage1_contract",
    "stage1_contract_sha256",
    "proportional_teacher_layer",
    "resolve_layer_map",
    "BridgeBank",
    "publish_bridge_artifact",
    "save_bridge_artifact",
    "read_bridge_manifest",
    "load_bridge_artifact",
]
