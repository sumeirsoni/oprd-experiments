#!/usr/bin/env python3
"""Immutable OPRD evaluation generation artifacts and standalone official rescoring."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable, Mapping

CLUSTER_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

from oprd_atomic_directory import atomic_move_directory_noreplace

from contract import canonical_json_bytes


SAMPLES_FILENAME = "samples.jsonl"
MANIFEST_FILENAME = "manifest.json"
COMPLETE_FILENAME = "COMPLETE"
SCORER_VERSION = 1
REQUIRED_RECORD_FIELDS = frozenset(
    {
        "benchmark",
        "problem_index",
        "rollout_index",
        "request_id",
        "raw_prompt",
        "rendered_prompt",
        "response_text",
        "generated_token_count",
        "finish_reason",
        "stop_reason",
        "finish_metadata_source",
        "instruction_suffix_count",
        "stop_token_ids",
        "vllm_version",
        "sampling",
        "requested_seed",
    }
)
VALID_TRACKS = frozenset(
    {"published_offline_exact", "corrected_seeded_single_suffix"}
)


class EvaluationArtifactError(RuntimeError):
    """A generation artifact or score layer is invalid or incomplete."""


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: pathlib.Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise EvaluationArtifactError(f"refusing to overwrite file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
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


def _json_line(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _validate_sampling(sampling: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for field in ("temperature", "top_p", "max_tokens", "enable_thinking"):
        if sampling.get(field) != expected.get(field):
            raise EvaluationArtifactError(
                f"sampling field {field} mismatch: expected {expected.get(field)!r}, "
                f"got {sampling.get(field)!r}"
            )


def _validate_record(
    record: Mapping[str, Any],
    *,
    benchmark: str,
    samples_per_problem: int,
    sampling: Mapping[str, Any],
    track: str,
) -> tuple[int, int]:
    missing = REQUIRED_RECORD_FIELDS - set(record)
    if missing:
        raise EvaluationArtifactError(f"generation record is missing fields: {sorted(missing)}")
    if record["benchmark"] != benchmark:
        raise EvaluationArtifactError("generation record benchmark mismatch")
    problem_index = record["problem_index"]
    rollout_index = record["rollout_index"]
    if not isinstance(problem_index, int) or problem_index < 0:
        raise EvaluationArtifactError("problem_index must be a nonnegative integer")
    if not isinstance(rollout_index, int) or not 0 <= rollout_index < samples_per_problem:
        raise EvaluationArtifactError("rollout_index is outside the configured sample count")
    for field in ("request_id", "raw_prompt", "rendered_prompt", "response_text"):
        if not isinstance(record[field], str):
            raise EvaluationArtifactError(f"generation record {field} must be a string")
    token_count = record["generated_token_count"]
    if not isinstance(token_count, int) or token_count < 0:
        raise EvaluationArtifactError("generated_token_count must be nonnegative")
    if token_count > sampling["max_tokens"]:
        raise EvaluationArtifactError("generated output exceeds the configured token cap")
    finish_reason = record["finish_reason"]
    if not isinstance(finish_reason, str) or not finish_reason:
        raise EvaluationArtifactError("finish_reason must be preserved")
    if record["finish_metadata_source"] != "vllm":
        raise EvaluationArtifactError("official evaluation requires authoritative vLLM finish metadata")
    if not isinstance(record["instruction_suffix_count"], int) or record["instruction_suffix_count"] < 0:
        raise EvaluationArtifactError("instruction_suffix_count must be nonnegative")
    if not isinstance(record["stop_token_ids"], list) or not all(
        isinstance(token_id, int) for token_id in record["stop_token_ids"]
    ):
        raise EvaluationArtifactError("stop_token_ids must be preserved as integers")
    if not isinstance(record["vllm_version"], str) or not record["vllm_version"]:
        raise EvaluationArtifactError("vLLM version must be preserved")
    if not isinstance(record["sampling"], Mapping):
        raise EvaluationArtifactError("record sampling must be an object")
    _validate_sampling(record["sampling"], sampling)
    if track == "published_offline_exact" and record["requested_seed"] is not None:
        raise EvaluationArtifactError("published exact evaluation must not set request seeds")
    if track == "corrected_seeded_single_suffix":
        if not isinstance(record["requested_seed"], int):
            raise EvaluationArtifactError("corrected evaluation requires an integer request seed")
        if record["instruction_suffix_count"] != 1:
            raise EvaluationArtifactError("corrected evaluation requires exactly one instruction suffix")
    return problem_index, rollout_index


def write_generation_artifact(
    destination: pathlib.Path,
    *,
    benchmark: Mapping[str, Any],
    track: str,
    sampling: Mapping[str, Any],
    identity: Mapping[str, Any],
    problems: list[Mapping[str, Any]],
    records: Iterable[Mapping[str, Any]],
    samples_per_problem: int,
) -> dict[str, Any]:
    if track not in VALID_TRACKS:
        raise EvaluationArtifactError(f"unknown evaluation track: {track}")
    if destination.exists() or destination.is_symlink():
        raise EvaluationArtifactError(f"refusing to overwrite evaluation artifact: {destination}")
    benchmark_name = benchmark.get("name")
    if not isinstance(benchmark_name, str) or not benchmark_name:
        raise EvaluationArtifactError("benchmark name is required")
    if not isinstance(samples_per_problem, int) or samples_per_problem < 1:
        raise EvaluationArtifactError("samples_per_problem must be positive")
    if len(problems) < 1:
        raise EvaluationArtifactError("at least one benchmark problem is required")
    for index, problem in enumerate(problems):
        if problem.get("problem_index") != index:
            raise EvaluationArtifactError("benchmark problems must have contiguous indices")
        if not isinstance(problem.get("problem"), str):
            raise EvaluationArtifactError("benchmark problem text must be preserved")
        if "ground_truth" not in problem:
            raise EvaluationArtifactError("benchmark ground truth must be preserved")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    promoted = False
    try:
        seen = set()
        row_count = 0
        with (staging / SAMPLES_FILENAME).open("wb") as handle:
            for raw_record in records:
                record = dict(raw_record)
                key = _validate_record(
                    record,
                    benchmark=benchmark_name,
                    samples_per_problem=samples_per_problem,
                    sampling=sampling,
                    track=track,
                )
                if key in seen:
                    raise EvaluationArtifactError(f"duplicate generation sample: {key}")
                seen.add(key)
                handle.write(_json_line(record))
                row_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        expected = {
            (problem_index, rollout_index)
            for problem_index in range(len(problems))
            for rollout_index in range(samples_per_problem)
        }
        if seen != expected:
            missing = sorted(expected - seen)
            extra = sorted(seen - expected)
            raise EvaluationArtifactError(
                f"generation sample coverage mismatch: missing={missing[:10]}, extra={extra[:10]}"
            )
        samples_path = staging / SAMPLES_FILENAME
        manifest = {
            "schema_version": 1,
            "artifact_kind": "oprd_official_evaluation_generations",
            "status": "complete",
            "track": track,
            "benchmark": dict(benchmark),
            "sampling": dict(sampling),
            "identity": dict(identity),
            "samples_per_problem": samples_per_problem,
            "problems": problems,
            "samples": {
                "path": SAMPLES_FILENAME,
                "rows": row_count,
                "size": samples_path.stat().st_size,
                "sha256": _sha256_file(samples_path),
            },
        }
        manifest_payload = _json_line(manifest)
        (staging / MANIFEST_FILENAME).write_bytes(manifest_payload)
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        (staging / COMPLETE_FILENAME).write_text(
            f"manifest_sha256 {manifest_sha256}\n", encoding="ascii"
        )
        for path in staging.iterdir():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        descriptor = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        atomic_move_directory_noreplace(
            staging, destination, trusted_root=destination.parent
        )
        promoted = True
        return {
            "destination": str(destination),
            "manifest_sha256": manifest_sha256,
            "samples_sha256": manifest["samples"]["sha256"],
            "rows": row_count,
        }
    finally:
        if not promoted:
            shutil.rmtree(staging, ignore_errors=True)


def _complete_digest(path: pathlib.Path) -> str:
    parts = path.read_text(encoding="ascii").strip().split()
    if len(parts) != 2 or parts[0] != "manifest_sha256":
        raise EvaluationArtifactError("invalid COMPLETE marker")
    digest = parts[1]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise EvaluationArtifactError("invalid COMPLETE manifest digest")
    return digest


def load_generation_artifact(
    manifest_path: pathlib.Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    manifest_path = manifest_path.resolve(strict=True)
    directory = manifest_path.parent
    if manifest_path.name != MANIFEST_FILENAME:
        raise EvaluationArtifactError("manifest path must end in manifest.json")
    complete_path = directory / COMPLETE_FILENAME
    samples_path = directory / SAMPLES_FILENAME
    if not complete_path.is_file() or not samples_path.is_file():
        raise EvaluationArtifactError("evaluation generation artifact is incomplete")
    manifest_payload = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    if _complete_digest(complete_path) != manifest_sha256:
        raise EvaluationArtifactError("COMPLETE marker does not authenticate manifest")
    manifest = json.loads(manifest_payload)
    if manifest.get("status") != "complete":
        raise EvaluationArtifactError("evaluation manifest is not complete")
    samples_metadata = manifest["samples"]
    if samples_path.stat().st_size != samples_metadata["size"]:
        raise EvaluationArtifactError("samples JSONL size mismatch")
    if _sha256_file(samples_path) != samples_metadata["sha256"]:
        raise EvaluationArtifactError("samples JSONL SHA-256 mismatch")
    records = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise EvaluationArtifactError(
                    f"invalid samples JSONL at line {line_number}: {exc}"
                ) from exc
    if len(records) != samples_metadata["rows"]:
        raise EvaluationArtifactError("samples JSONL row count mismatch")
    seen = set()
    for record in records:
        key = _validate_record(
            record,
            benchmark=manifest["benchmark"]["name"],
            samples_per_problem=manifest["samples_per_problem"],
            sampling=manifest["sampling"],
            track=manifest["track"],
        )
        if key in seen:
            raise EvaluationArtifactError(f"duplicate generation sample: {key}")
        seen.add(key)
    expected_count = len(manifest["problems"]) * manifest["samples_per_problem"]
    if len(seen) != expected_count:
        raise EvaluationArtifactError("generation artifact does not cover every expected sample")
    return manifest, records, manifest_sha256


def _load_official_grader(
    official_source: pathlib.Path,
) -> tuple[Callable[[str, Any], Any], Callable[[str], Any]]:
    evaluation_directory = (
        official_source.resolve(strict=True) / "scripts" / "val" / "eval"
    )
    utils_path = evaluation_directory / "utils.py"
    if not utils_path.is_file():
        raise EvaluationArtifactError(f"official grader utilities are missing: {utils_path}")
    sys.path.insert(0, str(evaluation_directory))
    try:
        spec = importlib.util.spec_from_file_location("oprd_official_eval_utils", utils_path)
        if spec is None or spec.loader is None:
            raise EvaluationArtifactError("cannot load official grader module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(evaluation_directory))
    return module.grade_answer_verl, module.extract_answer


def _is_cap_hit(record: Mapping[str, Any], max_tokens: int) -> bool:
    if record["finish_reason"] == "length":
        return True
    if record["generated_token_count"] == max_tokens:
        return True
    if record["finish_reason"] in {"stop", "eos", "eos_token", "stop_token"}:
        return False
    raise EvaluationArtifactError(
        f"ambiguous finish reason for request {record['request_id']}: "
        f"{record['finish_reason']!r}"
    )


def _distinct_ngram(texts: Iterable[str], n: int = 4) -> float:
    unique = set()
    total = 0
    for text in texts:
        tokens = text.split()
        for index in range(max(0, len(tokens) - n + 1)):
            unique.add(tuple(tokens[index : index + n]))
            total += 1
    return len(unique) / total if total else 0.0


def score_generation_artifact(
    manifest_path: pathlib.Path,
    *,
    grader: Callable[[str, Any], Any],
    extractor: Callable[[str], Any],
    grader_identity: Mapping[str, Any],
    output_path: pathlib.Path,
) -> dict[str, Any]:
    manifest, records, manifest_sha256 = load_generation_artifact(manifest_path)
    problems = {problem["problem_index"]: problem for problem in manifest["problems"]}
    counts = Counter()
    per_problem = defaultdict(list)
    outcomes = []
    lengths = []
    texts = []
    for record in records:
        problem = problems[record["problem_index"]]
        correct = bool(grader(record["response_text"], problem["ground_truth"]))
        extracted = extractor(record["response_text"])
        cap_hit = _is_cap_hit(record, manifest["sampling"]["max_tokens"])
        bucket = (
            "cap_hit_correct"
            if cap_hit and correct
            else "cap_hit_incorrect"
            if cap_hit
            else "natural_finish_correct"
            if correct
            else "natural_finish_incorrect"
        )
        counts[bucket] += 1
        if extracted is None:
            counts["format_error"] += 1
        per_problem[record["problem_index"]].append(correct)
        lengths.append(record["generated_token_count"])
        texts.append(record["response_text"])
        outcomes.append(
            {
                "problem_index": record["problem_index"],
                "rollout_index": record["rollout_index"],
                "request_id": record["request_id"],
                "bucket": bucket,
                "correct": correct,
                "cap_hit": cap_hit,
                "format_error": extracted is None,
                "extracted_answer": extracted,
            }
        )
    total = len(records)
    correct_total = counts["natural_finish_correct"] + counts["cap_hit_correct"]
    best_total = sum(any(values) for values in per_problem.values())
    bucket_total = sum(
        counts[name]
        for name in (
            "natural_finish_correct",
            "natural_finish_incorrect",
            "cap_hit_correct",
            "cap_hit_incorrect",
        )
    )
    if bucket_total != total:
        raise EvaluationArtifactError("finish/correctness buckets are not exhaustive")
    result = {
        "schema_version": 1,
        "artifact_kind": "oprd_official_evaluation_score",
        "scorer_version": SCORER_VERSION,
        "generation_manifest_sha256": manifest_sha256,
        "generation_samples_sha256": manifest["samples"]["sha256"],
        "track": manifest["track"],
        "benchmark": manifest["benchmark"],
        "grader": dict(grader_identity),
        "n": total,
        "counts": {
            name: counts[name]
            for name in (
                "natural_finish_correct",
                "natural_finish_incorrect",
                "cap_hit_correct",
                "cap_hit_incorrect",
                "format_error",
            )
        },
        "metrics": {
            "avg_at_16": correct_total / total if total else 0.0,
            "best_at_16": best_total / len(per_problem) if per_problem else 0.0,
            "cap_hit_rate": (
                counts["cap_hit_correct"] + counts["cap_hit_incorrect"]
            )
            / total
            if total
            else 0.0,
            "format_error_rate": counts["format_error"] / total if total else 0.0,
            "average_output_length": sum(lengths) / len(lengths) if lengths else 0.0,
            "distinct_4gram": _distinct_ngram(texts),
        },
        "outcomes": outcomes,
    }
    _atomic_write(output_path, _json_line(result))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--manifest", type=pathlib.Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--manifest", type=pathlib.Path, required=True)
    score.add_argument("--official-source", type=pathlib.Path, required=True)
    score.add_argument("--official-commit", required=True)
    score.add_argument("--output", type=pathlib.Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "validate":
        manifest, records, manifest_sha256 = load_generation_artifact(arguments.manifest)
        print(
            json.dumps(
                {
                    "ok": True,
                    "manifest_sha256": manifest_sha256,
                    "rows": len(records),
                    "track": manifest["track"],
                },
                sort_keys=True,
            )
        )
        return
    grader, extractor = _load_official_grader(arguments.official_source)
    result = score_generation_artifact(
        arguments.manifest,
        grader=grader,
        extractor=extractor,
        grader_identity={
            "repository": "https://github.com/ShenzhiYang2000/OPRD",
            "commit": arguments.official_commit,
            "entry_point": "scripts/val/eval/utils.py:grade_answer_verl",
        },
        output_path=arguments.output,
    )
    print(json.dumps({"ok": True, "metrics": result["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
