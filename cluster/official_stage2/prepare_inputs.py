#!/usr/bin/env python3
"""Authenticate the pinned official source tree and vendored DAPO parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import tempfile
from typing import Any, Iterable

from contract import canonical_json_bytes, contract_sha256, load_contract


class InputError(RuntimeError):
    """An official Stage-2 source or data input failed authentication."""


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(*arguments: str, cwd: pathlib.Path | None = None) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise InputError(
            f"command failed ({completed.returncode}): {' '.join(arguments)}:\n"
            f"{completed.stderr}"
        )
    return completed.stdout


def _atomic_write_json(path: pathlib.Path, value: Any) -> None:
    if path.exists():
        raise InputError(f"refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _tracked_paths(source: pathlib.Path) -> list[pathlib.Path]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=source,
        check=True,
        capture_output=True,
    ).stdout
    return [source / os.fsdecode(name) for name in output.split(b"\0") if name]


def _file_mode(path: pathlib.Path) -> str:
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode):
        return "symlink"
    if mode & stat.S_IXUSR:
        return "executable"
    return "file"


def authenticate_source(
    source: pathlib.Path,
    manifest_path: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    source = source.resolve(strict=True)
    expected_commit = contract["official_source"]["commit"]
    actual_commit = _run("git", "rev-parse", "HEAD", cwd=source).strip()
    if actual_commit != expected_commit:
        raise InputError(
            f"official source commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    status = _run("git", "status", "--porcelain", "--untracked-files=all", cwd=source)
    if status:
        raise InputError("official source tree is dirty:\n" + status)

    records = []
    for path in _tracked_paths(source):
        relative = path.relative_to(source).as_posix()
        mode = _file_mode(path)
        if mode == "symlink":
            target = os.readlink(path)
            payload = os.fsencode(target)
            size = len(payload)
            digest = hashlib.sha256(payload).hexdigest()
            records.append(
                {
                    "path": relative,
                    "mode": mode,
                    "size": size,
                    "sha256": digest,
                    "target": target,
                }
            )
        else:
            records.append(
                {
                    "path": relative,
                    "mode": mode,
                    "size": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    tree_sha256 = hashlib.sha256(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    ).hexdigest()
    manifest = {
        "schema_version": 1,
        "artifact_kind": "official_oprd_source",
        "repository": contract["official_source"]["repository"],
        "commit": actual_commit,
        "contract_sha256": contract_sha256(contract),
        "tracked_files": len(records),
        "tree_sha256": tree_sha256,
        "files": records,
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _prompt_rows(table: Any, prompt_key: str) -> Iterable[Any]:
    if prompt_key not in table.column_names:
        raise InputError(f"parquet is missing prompt column: {prompt_key}")
    for index, value in enumerate(table[prompt_key].to_pylist()):
        if value is None:
            raise InputError(f"null prompt at row {index}")
        if not isinstance(value, (str, list)):
            raise InputError(
                f"unsupported prompt type at row {index}: {type(value).__name__}"
            )
        yield value


def authenticate_dataset(
    source: pathlib.Path,
    manifest_path: pathlib.Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise InputError("pyarrow is required to authenticate the DAPO parquet") from exc

    source = source.resolve(strict=True)
    data_contract = contract["training_data"]
    git_blob = _run("git", "hash-object", str(source)).strip()
    if git_blob != data_contract["official_git_blob"]:
        raise InputError(
            f"DAPO Git blob mismatch: expected {data_contract['official_git_blob']}, "
            f"got {git_blob}"
        )
    table = parquet.read_table(source)
    if table.num_rows != data_contract["rows"]:
        raise InputError(
            f"DAPO row count mismatch: expected {data_contract['rows']}, got {table.num_rows}"
        )
    row_digest = hashlib.sha256()
    prompt_rows = list(_prompt_rows(table, data_contract["prompt_key"]))
    for index, prompt in enumerate(prompt_rows):
        row_digest.update(
            canonical_json_bytes({"index": index, "prompt": prompt}) + b"\n"
        )
    manifest = {
        "schema_version": 1,
        "artifact_kind": "official_dapo_parquet",
        "contract_sha256": contract_sha256(contract),
        "dataset_id": data_contract["dataset_id"],
        "dataset_revision": data_contract["dataset_revision"],
        "official_path": data_contract["official_path"],
        "official_git_blob": git_blob,
        "sha256": _sha256_file(source),
        "size": source.stat().st_size,
        "rows": table.num_rows,
        "columns": table.column_names,
        "schema": str(table.schema),
        "prompt_key": data_contract["prompt_key"],
        "row_order_sha256": row_digest.hexdigest(),
        "expected_updates": data_contract["expected_updates"],
        "expected_consumed_rows": data_contract["expected_consumed_rows"],
        "expected_dropped_tail_rows": data_contract["expected_dropped_tail_rows"],
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=pathlib.Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source")
    source.add_argument("--source", type=pathlib.Path, required=True)
    source.add_argument("--manifest", type=pathlib.Path, required=True)
    dataset = subparsers.add_parser("dataset")
    dataset.add_argument("--parquet", type=pathlib.Path, required=True)
    dataset.add_argument("--manifest", type=pathlib.Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    contract = load_contract(arguments.contract) if arguments.contract else load_contract()
    if arguments.command == "source":
        manifest = authenticate_source(arguments.source, arguments.manifest, contract)
    else:
        manifest = authenticate_dataset(arguments.parquet, arguments.manifest, contract)
    print(json.dumps({"ok": True, "manifest": str(arguments.manifest), "sha256": hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()}))


if __name__ == "__main__":
    main()
