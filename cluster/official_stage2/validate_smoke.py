#!/usr/bin/env python3
"""Validate and accept an official Stage-2 smoke or canary run."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any


class SmokeValidationError(RuntimeError):
    """Raised when smoke or canary evidence is incomplete or inconsistent."""


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_smoke(
    *,
    run: pathlib.Path,
    project: pathlib.Path,
    runtime_log: pathlib.Path,
    rows: int,
    world_size: int,
) -> dict[str, Any]:
    if rows not in (16, 80):
        raise SmokeValidationError("rows must be 16 or 80")
    if world_size not in (4, 8):
        raise SmokeValidationError("world size must be 4 or 8")
    expected_updates = rows // 8
    hydra = run / "actual-hydra.yaml"
    argv = run / "actual-python-argv.json"
    if not hydra.is_file() or not argv.is_file() or not runtime_log.is_file():
        raise SmokeValidationError("actual Hydra, argv, or runtime capture is missing")
    logs = sorted((project / "logs" / "terminal").glob("*.log"))
    if len(logs) != 1:
        raise SmokeValidationError(f"expected one official terminal log, found {len(logs)}")
    terminal_text = logs[0].read_text(encoding="utf-8", errors="replace")
    for marker in ("Traceback", "RayActorError", "CUDA out of memory", "NCCL error"):
        if marker in terminal_text:
            raise SmokeValidationError(f"terminal log contains failure marker: {marker}")
    runtime_text = runtime_log.read_text(encoding="utf-8", errors="replace")
    step_marker = f"training/global_step:{expected_updates}"
    if step_marker not in runtime_text:
        raise SmokeValidationError(f"runtime log is missing final step marker: {step_marker}")
    for marker in ("RayActorError", "ActorDiedError", "CUDA out of memory", "NCCL error"):
        if marker in runtime_text:
            raise SmokeValidationError(f"runtime log contains failure marker: {marker}")
    cleanup_warning = "RuntimeError: DataLoader worker ("
    cleanup_warning_count = sum(
        cleanup_warning in line and "is killed by signal: Killed" in line
        for line in runtime_text.splitlines()
    )
    traceback_count = runtime_text.count("Traceback (most recent call last):")
    if traceback_count != cleanup_warning_count:
        raise SmokeValidationError(
            "runtime traceback count is not explained by the known DataLoader shutdown warning: "
            f"tracebacks={traceback_count}, cleanup_warnings={cleanup_warning_count}"
        )
    steps = list(project.rglob(f"global_step_{expected_updates}"))
    if len(steps) != 1:
        raise SmokeValidationError(f"expected one final checkpoint, found {len(steps)}")
    actor = steps[0] / "actor"
    for kind in ("model", "optim", "extra_state"):
        pattern = f"{kind}_world_size_{world_size}_rank_*.pt"
        files = list(actor.glob(pattern))
        if len(files) != world_size:
            raise SmokeValidationError(
                f"expected {world_size} checkpoint shards for {pattern}, found {len(files)}"
            )
    if not (actor / "huggingface" / "config.json").is_file():
        raise SmokeValidationError("checkpoint Hugging Face config is missing")
    return {
        "status": "complete",
        "rows": rows,
        "expected_updates": expected_updates,
        "world_size": world_size,
        "topology_deviation_from_official": world_size != 8,
        "checkpoint": str(steps[0]),
        "checkpoint_shards_per_kind": world_size,
        "cleanup_warning_count": cleanup_warning_count,
        "actual_hydra_sha256": _sha256(hydra),
        "actual_argv_sha256": _sha256(argv),
        "runtime_log": str(runtime_log),
        "runtime_log_sha256": _sha256(runtime_log),
        "terminal_log": str(logs[0]),
        "terminal_log_sha256": _sha256(logs[0]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=pathlib.Path, required=True)
    parser.add_argument("--project-path", type=pathlib.Path, required=True)
    parser.add_argument("--runtime-log", type=pathlib.Path, required=True)
    parser.add_argument("--rows", type=int, choices=(16, 80), required=True)
    parser.add_argument("--world-size", type=int, choices=(4, 8), required=True)
    parser.add_argument("--write-marker", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = validate_smoke(
        run=arguments.run_directory.resolve(strict=True),
        project=arguments.project_path.resolve(strict=True),
        runtime_log=arguments.runtime_log.resolve(strict=True),
        rows=arguments.rows,
        world_size=arguments.world_size,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if arguments.write_marker:
        result_path = arguments.run_directory / "smoke-result.json"
        marker_path = arguments.run_directory / "COMPLETE"
        if result_path.exists() or marker_path.exists():
            raise SmokeValidationError("smoke result or completion marker already exists")
        result_path.write_text(
            json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        marker_path.write_text("OFFICIAL_STAGE2_SMOKE_COMPLETE\n", encoding="utf-8")


if __name__ == "__main__":
    main()
