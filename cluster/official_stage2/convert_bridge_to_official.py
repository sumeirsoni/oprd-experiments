#!/usr/bin/env python3
"""Convert an immutable OPRD bridge into the pinned official projector-bank schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
from typing import Any, Mapping

import torch
import torch.nn.functional as F

CLUSTER_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

from oprd_atomic_directory import atomic_move_directory_noreplace
from oprd_bridge import load_bridge_artifact, read_bridge_manifest

from contract import canonical_json_bytes, contract_sha256, load_contract


AUDIT_TENSOR_FILENAME = "ps_bank.audit.safetensors"
OFFICIAL_BANK_FILENAME = "ps_bank.pt"
MANIFEST_FILENAME = "manifest.json"
PARITY_FILENAME = "parity.json"
COMPLETE_FILENAME = "COMPLETE"


class AdapterError(RuntimeError):
    """The bridge could not be converted or authenticated."""


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    raw = contiguous.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: pathlib.Path, value: Any) -> bytes:
    payload = canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return payload


def _fsync_file(path: pathlib.Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: pathlib.Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _normalized_layer_map(manifest: Mapping[str, Any]) -> dict[int, int]:
    pairs = sorted(
        (
            (int(student), int(teacher))
            for student, teacher in manifest["bridge"]["layer_map"].items()
        ),
        key=lambda pair: pair[0],
    )
    return dict(pairs)


def _load_source(
    artifact: pathlib.Path, contract: Mapping[str, Any]
) -> tuple[Any, dict[str, Any], str, dict[int, int]]:
    manifest, manifest_sha256 = read_bridge_manifest(artifact)
    expected_bridge = contract["stage1_bridge"]
    if manifest_sha256 != expected_bridge["artifact_manifest_sha256"]:
        raise AdapterError(
            "source bridge manifest mismatch: "
            f"expected {expected_bridge['artifact_manifest_sha256']}, got {manifest_sha256}"
        )
    layer_map = _normalized_layer_map(manifest)
    models = contract["models"]
    bridge, _ = load_bridge_artifact(
        artifact,
        expected_student_model_id=models["student"]["id"],
        expected_student_model_revision=models["student"]["revision"],
        expected_teacher_model_id=models["teacher"]["id"],
        expected_teacher_model_revision=models["teacher"]["revision"],
        expected_tokenizer_id=models["tokenizer"]["id"],
        expected_tokenizer_revision=models["tokenizer"]["revision"],
        expected_stage1_contract_sha256=expected_bridge["stage1_contract_sha256"],
        expected_layer_map=layer_map,
        expected_rank=expected_bridge["rank"],
        expected_student_dim=models["student"]["hidden_size"],
        expected_teacher_dim=models["teacher"]["hidden_size"],
        expected_manifest_sha256=manifest_sha256,
        device="cpu",
    )
    return bridge, manifest, manifest_sha256, layer_map


def _official_pair(student_layer: int, teacher_layer: int) -> str:
    return f"s{student_layer - 1}_t{teacher_layer - 1}"


def build_official_payload(
    bridge: Any, layer_map: Mapping[int, int]
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    state_dict: dict[str, torch.Tensor] = {}
    frozen_pt_weights: dict[str, torch.Tensor] = {}
    frozen_pt_means: dict[str, torch.Tensor] = {}
    audit: dict[str, torch.Tensor] = {}
    pairs = []
    for student_layer, teacher_layer in layer_map.items():
        pair = _official_pair(student_layer, teacher_layer)
        student_weight = (
            bridge.student_projectors[str(student_layer)].weight.detach().cpu().contiguous()
        )
        teacher_pca = (
            getattr(bridge, f"teacher_pca_{student_layer}").detach().cpu().contiguous()
        )
        teacher_mean = (
            getattr(bridge, f"teacher_mean_{student_layer}").detach().cpu().contiguous()
        )
        state_dict[f"projectors.{pair}.weight"] = student_weight
        frozen_pt_weights[pair] = teacher_pca
        frozen_pt_means[pair] = teacher_mean
        audit[f"student_projectors.{pair}.weight"] = student_weight
        audit[f"teacher_pca.{pair}"] = teacher_pca
        audit[f"teacher_mean.{pair}"] = teacher_mean
        pairs.append((student_layer - 1, teacher_layer - 1))
    payload = {
        "rank": bridge.rank,
        "projector_type": "linear",
        "mlp_hidden_mult": None,
        "subspace_mode": "full",
        "layer_pairs": pairs,
        "state_dict": state_dict,
        "frozen_pt_weights": frozen_pt_weights,
        "frozen_pt_means": frozen_pt_means,
    }
    return payload, audit


def parity_report(
    bridge: Any,
    layer_map: Mapping[int, int],
    official_payload: Mapping[str, Any],
    *,
    seed: int = 20260731,
    rows: int = 5,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    layers = []
    maximum_student_difference = 0.0
    maximum_teacher_difference = 0.0
    maximum_loss_difference = 0.0
    for student_layer, teacher_layer in layer_map.items():
        pair = _official_pair(student_layer, teacher_layer)
        student_hidden = torch.randn(rows, bridge.student_dim, generator=generator)
        teacher_hidden = torch.randn(rows, bridge.teacher_dim, generator=generator)
        expected_student = bridge.project_student(student_layer, student_hidden)
        expected_teacher = bridge.project_teacher(student_layer, teacher_hidden)
        actual_student = F.linear(
            student_hidden,
            official_payload["state_dict"][f"projectors.{pair}.weight"],
        )
        actual_teacher = F.linear(
            teacher_hidden - official_payload["frozen_pt_means"][pair],
            official_payload["frozen_pt_weights"][pair],
        )
        student_difference = float((actual_student - expected_student).abs().max().item())
        teacher_difference = float((actual_teacher - expected_teacher).abs().max().item())
        expected_loss = F.mse_loss(
            F.normalize(expected_student, dim=-1),
            F.normalize(expected_teacher, dim=-1),
        )
        actual_loss = F.mse_loss(
            F.normalize(actual_student, dim=-1),
            F.normalize(actual_teacher, dim=-1),
        )
        loss_difference = abs(float(actual_loss.item()) - float(expected_loss.item()))
        maximum_student_difference = max(maximum_student_difference, student_difference)
        maximum_teacher_difference = max(maximum_teacher_difference, teacher_difference)
        maximum_loss_difference = max(maximum_loss_difference, loss_difference)
        layers.append(
            {
                "student_layer_1_based": student_layer,
                "teacher_layer_1_based": teacher_layer,
                "official_pair": pair,
                "student_projection_max_abs_difference": student_difference,
                "teacher_projection_max_abs_difference": teacher_difference,
                "normalized_mse_abs_difference": loss_difference,
            }
        )
    if maximum_student_difference != 0.0 or maximum_teacher_difference != 0.0:
        raise AdapterError("official projector payload does not exactly match source bridge")
    return {
        "schema_version": 1,
        "seed": seed,
        "probe_rows": rows,
        "layers": layers,
        "maximum_student_projection_abs_difference": maximum_student_difference,
        "maximum_teacher_projection_abs_difference": maximum_teacher_difference,
        "maximum_normalized_mse_abs_difference": maximum_loss_difference,
        "status": "ok",
    }


def _tensor_manifest(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "sha256": _sha256_tensor(tensor),
            "finite": bool(torch.isfinite(tensor).all().item()),
        }
        for name, tensor in sorted(tensors.items())
    }


def verify_converted_bridge(
    artifact: pathlib.Path,
    adapter_directory: pathlib.Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    bridge, _, source_manifest_sha256, layer_map = _load_source(
        artifact.resolve(strict=True), contract
    )
    expected_payload, expected_audit = build_official_payload(bridge, layer_map)
    adapter_directory = adapter_directory.resolve(strict=True)
    try:
        actual_payload = torch.load(
            adapter_directory / OFFICIAL_BANK_FILENAME,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise AdapterError(f"cannot safely load official projector bank: {exc}") from exc
    from safetensors.torch import load_file
    actual_audit = load_file(
        str(adapter_directory / AUDIT_TENSOR_FILENAME), device="cpu"
    )
    for field in ("rank", "projector_type", "mlp_hidden_mult", "subspace_mode", "layer_pairs"):
        if actual_payload.get(field) != expected_payload[field]:
            raise AdapterError(f"official projector metadata mismatch: {field}")
    for group in ("state_dict", "frozen_pt_weights", "frozen_pt_means"):
        expected_group = expected_payload[group]
        actual_group = actual_payload.get(group)
        if not isinstance(actual_group, Mapping) or set(actual_group) != set(expected_group):
            raise AdapterError(f"official projector tensor keys mismatch: {group}")
        for name, expected_tensor in expected_group.items():
            if not torch.equal(actual_group[name], expected_tensor):
                raise AdapterError(f"official projector tensor mismatch: {group}.{name}")
    if set(actual_audit) != set(expected_audit):
        raise AdapterError("official audit tensor keys mismatch")
    for name, expected_tensor in expected_audit.items():
        if not torch.equal(actual_audit[name], expected_tensor):
            raise AdapterError(f"official audit tensor mismatch: {name}")
    parity = parity_report(bridge, layer_map, actual_payload)
    return {
        "status": "adapter_ok",
        "source_manifest_sha256": source_manifest_sha256,
        "bank_sha256": _sha256_file(adapter_directory / OFFICIAL_BANK_FILENAME),
        "audit_sha256": _sha256_file(adapter_directory / AUDIT_TENSOR_FILENAME),
        "parity": parity,
    }


def convert_bridge(
    artifact: pathlib.Path,
    destination: pathlib.Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    destination = destination.resolve(strict=False)
    if destination.exists() or destination.is_symlink():
        raise AdapterError(f"refusing to overwrite adapter output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bridge, source_manifest, source_manifest_sha256, layer_map = _load_source(
        artifact.resolve(strict=True), contract
    )
    official_payload, audit_tensors = build_official_payload(bridge, layer_map)
    parity = parity_report(bridge, layer_map, official_payload)

    from safetensors.torch import load_file, save_file

    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    promoted = False
    try:
        bank_path = staging / OFFICIAL_BANK_FILENAME
        audit_path = staging / AUDIT_TENSOR_FILENAME
        torch.save(official_payload, bank_path)
        save_file(audit_tensors, str(audit_path))
        reloaded_audit = load_file(str(audit_path), device="cpu")
        if set(reloaded_audit) != set(audit_tensors):
            raise AdapterError("audit safetensors names changed during serialization")
        for name, tensor in audit_tensors.items():
            if not torch.equal(reloaded_audit[name], tensor):
                raise AdapterError(f"audit tensor changed during serialization: {name}")

        parity_bytes = _write_json(staging / PARITY_FILENAME, parity)
        manifest = {
            "schema_version": 1,
            "artifact_kind": "official_oprd_projector_bank_adapter",
            "contract_sha256": contract_sha256(contract),
            "official_source_commit": contract["official_source"]["commit"],
            "source_bridge": {
                "artifact_manifest_sha256": source_manifest_sha256,
                "tensor_sha256": source_manifest["storage"]["sha256"],
                "stage1_contract_sha256": source_manifest["identity"][
                    "stage1_contract_sha256"
                ],
            },
            "layer_map_1_based": {
                str(student): teacher for student, teacher in layer_map.items()
            },
            "official_layer_pairs_0_based": official_payload["layer_pairs"],
            "rank": bridge.rank,
            "student_dim": bridge.student_dim,
            "teacher_dim": bridge.teacher_dim,
            "files": {
                OFFICIAL_BANK_FILENAME: {
                    "sha256": _sha256_file(bank_path),
                    "size": bank_path.stat().st_size,
                    "trust": "pickle execution artifact; authenticate tensors through audit safetensors",
                },
                AUDIT_TENSOR_FILENAME: {
                    "sha256": _sha256_file(audit_path),
                    "size": audit_path.stat().st_size,
                },
                PARITY_FILENAME: {
                    "sha256": hashlib.sha256(parity_bytes).hexdigest(),
                    "size": len(parity_bytes),
                },
            },
            "tensors": _tensor_manifest(audit_tensors),
            "parity_status": parity["status"],
        }
        manifest_bytes = _write_json(staging / MANIFEST_FILENAME, manifest)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        (staging / COMPLETE_FILENAME).write_text(
            f"manifest_sha256 {manifest_sha256}\n", encoding="ascii"
        )
        for path in staging.iterdir():
            _fsync_file(path)
        _fsync_directory(staging)
        atomic_move_directory_noreplace(
            staging,
            destination,
            trusted_root=destination.parent,
        )
        promoted = True
        return {
            "destination": str(destination),
            "manifest_sha256": manifest_sha256,
            "bank_sha256": manifest["files"][OFFICIAL_BANK_FILENAME]["sha256"],
            "audit_sha256": manifest["files"][AUDIT_TENSOR_FILENAME]["sha256"],
        }
    finally:
        if not promoted:
            shutil.rmtree(staging, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=pathlib.Path, default=None)
    parser.add_argument("--artifact", type=pathlib.Path, required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    contract = load_contract(arguments.contract) if arguments.contract else load_contract()
    result = convert_bridge(arguments.artifact, arguments.destination, contract)
    print(json.dumps({"ok": True, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
