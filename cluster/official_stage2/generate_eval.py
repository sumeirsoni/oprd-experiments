#!/usr/bin/env python3
"""Generate one immutable official OPRD evaluation track with eight vLLM workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import multiprocessing
import os
import pathlib
from typing import Any

import pandas as pd

from aime_eval import write_generation_artifact
from contract import load_contract


PROMPT_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
BENCHMARK_PATHS = {
    "aime24": "scripts/val/data/AIME24/test.parquet",
    "aime25": "scripts/val/data/AIME25/test.parquet",
    "aimo": "scripts/val/data/AIMO/test.parquet",
}


class GenerationError(RuntimeError):
    """Official evaluation generation failed or returned incomplete metadata."""


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_problems(path: pathlib.Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)
    problems = []
    for index in range(len(frame)):
        if "BRUMO25" in str(path) or "CMIMC25" in str(path) or "HMMT25" in str(path):
            prompt = frame.at[index, "problem"].strip()
            answer = frame.at[index, "answer"].strip()
        else:
            prompt = frame.at[index, "prompt"][0]["content"].strip()
            answer = frame.at[index, "reward_model"]["ground_truth"].strip()
        problems.append(
            {
                "problem_index": index,
                "problem": prompt,
                "ground_truth": answer,
            }
        )
    if not problems:
        raise GenerationError(f"benchmark contains no problems: {path}")
    return problems


def _render_user_prompt(raw_prompt: str, track: str) -> str:
    if track == "published_offline_exact":
        return f"{raw_prompt} {PROMPT_SUFFIX}"
    count = raw_prompt.count(PROMPT_SUFFIX)
    if count > 1:
        raise GenerationError("corrected evaluation source prompt already duplicates the suffix")
    return raw_prompt if count == 1 else f"{raw_prompt} {PROMPT_SUFFIX}"


def _seed(checkpoint_sha256: str, dataset_sha256: str, problem: int, rollout: int) -> int:
    payload = f"oprd-eval-seed-v1:{checkpoint_sha256}:{dataset_sha256}:{problem}:{rollout}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63 - 1)


def _worker(arguments: tuple[Any, ...]) -> list[dict[str, Any]]:
    (
        checkpoint,
        problems,
        rollout_ids,
        gpu_id,
        track,
        checkpoint_sha256,
        dataset_sha256,
        benchmark,
    ) = arguments
    os.environ.pop("ROCR_VISIBLE_DEVICES", None)
    os.environ.pop("HIP_VISIBLE_DEVICES", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=checkpoint,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=1,
    )
    tokenizer = llm.get_tokenizer()
    stop_token_ids = []
    for stop_token in ("<|im_end|>", "<|endoftext|>"):
        encoded = tokenizer.encode(stop_token, add_special_tokens=False)
        if encoded:
            stop_token_ids.append(encoded[0])
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": _render_user_prompt(problem["problem"], track)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for problem in problems
    ]
    records = []
    sampling_record = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 31744,
        "enable_thinking": False,
    }
    for rollout_index in rollout_ids:
        if track == "published_offline_exact":
            requested_seeds = [None] * len(problems)
            sampling = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=31744,
                stop_token_ids=stop_token_ids or None,
            )
        else:
            requested_seeds = [
                _seed(checkpoint_sha256, dataset_sha256, index, rollout_index)
                for index in range(len(problems))
            ]
            sampling = [
                SamplingParams(
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=31744,
                    stop_token_ids=stop_token_ids or None,
                    seed=seed,
                )
                for seed in requested_seeds
            ]
        outputs = llm.generate(rendered, sampling, use_tqdm=False)
        if len(outputs) != len(problems):
            raise GenerationError("vLLM returned the wrong number of evaluation outputs")
        for problem, rendered_prompt, request_output, requested_seed in zip(
            problems, rendered, outputs, requested_seeds
        ):
            if not request_output.finished or len(request_output.outputs) != 1:
                raise GenerationError("vLLM request did not finish with exactly one candidate")
            candidate = request_output.outputs[0]
            response_text = candidate.text
            records.append(
                {
                    "benchmark": benchmark,
                    "problem_index": problem["problem_index"],
                    "rollout_index": rollout_index,
                    "request_id": str(request_output.request_id),
                    "raw_prompt": problem["problem"],
                    "rendered_prompt": rendered_prompt,
                    "response_text": response_text,
                    "generated_token_count": len(candidate.token_ids),
                    "finish_reason": candidate.finish_reason,
                    "stop_reason": candidate.stop_reason,
                    "finish_metadata_source": "vllm",
                    "instruction_suffix_count": rendered_prompt.count(PROMPT_SUFFIX),
                    "stop_token_ids": stop_token_ids,
                    "vllm_version": importlib.metadata.version("vllm"),
                    "sampling": sampling_record,
                    "requested_seed": requested_seed,
                }
            )
    del llm
    return records


def generate(arguments: argparse.Namespace) -> dict[str, Any]:
    contract = load_contract(arguments.contract) if arguments.contract else load_contract()
    official_source = arguments.official_source.resolve(strict=True)
    benchmark_path = official_source / BENCHMARK_PATHS[arguments.benchmark]
    if not benchmark_path.is_file():
        raise GenerationError(f"benchmark parquet is missing: {benchmark_path}")
    problems = load_problems(benchmark_path)
    dataset_sha256 = _sha256_file(benchmark_path)
    rollout_chunks = [list(range(worker, 16, 8)) for worker in range(8)]
    worker_arguments = [
        (
            str(arguments.checkpoint.resolve(strict=True)),
            problems,
            rollout_chunks[worker],
            worker,
            arguments.track,
            arguments.checkpoint_sha256,
            dataset_sha256,
            arguments.benchmark,
        )
        for worker in range(8)
    ]
    context = multiprocessing.get_context("spawn")
    records = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=8, mp_context=context
    ) as executor:
        futures = [executor.submit(_worker, value) for value in worker_arguments]
        for future in concurrent.futures.as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda record: (record["problem_index"], record["rollout_index"]))
    model = contract["models"]["student"]
    return write_generation_artifact(
        arguments.destination,
        benchmark={
            "name": arguments.benchmark,
            "path": BENCHMARK_PATHS[arguments.benchmark],
            "dataset_sha256": dataset_sha256,
            "problem_count": len(problems),
        },
        track=arguments.track,
        sampling={
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 31744,
            "enable_thinking": False,
        },
        identity={
            "checkpoint": str(arguments.checkpoint.resolve(strict=True)),
            "checkpoint_sha256": arguments.checkpoint_sha256,
            "model_id": model["id"],
            "model_revision": model["revision"],
            "tokenizer_id": contract["models"]["tokenizer"]["id"],
            "tokenizer_revision": contract["models"]["tokenizer"]["revision"],
            "official_source_commit": contract["official_source"]["commit"],
        },
        problems=problems,
        records=records,
        samples_per_problem=16,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=pathlib.Path, default=None)
    parser.add_argument("--official-source", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--benchmark", choices=sorted(BENCHMARK_PATHS), required=True)
    parser.add_argument("--track", choices=sorted(("published_offline_exact", "corrected_seeded_single_suffix")), required=True)
    parser.add_argument("--destination", type=pathlib.Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if len(arguments.checkpoint_sha256) != 64:
        raise SystemExit("--checkpoint-sha256 must be a SHA-256 digest")
    print(generate(arguments))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
