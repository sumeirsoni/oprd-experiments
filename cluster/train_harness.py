"""Shared training harness for Tier 1/2 OPRD experiments.

Full on-policy loop: student rollout generation (vLLM in an isolated subprocess on CUDA,
transformers fallback elsewhere), frozen-teacher prefill over student rollouts, nine loss
arms, AdamW with token-budgeted gradient accumulation, periodic rollout refresh, AIME Avg@N
eval, and diagnostics.

Loss arms (adversarial review §5.5/§5.7; supervised on generated-region tokens only):
  oprd_bridge       Stage 2 of the OPRD-Bridge paper/official-code algorithm: consume one
                    explicit immutable Stage-1 artifact, then optimize squared MSE between
                    L2-normalized frozen bridge projections with student-only gradients.
  linear_map_mse    TinyBERT-style CONTRAST arm: full-rank map W: d_s->d_t, MSE in
                    teacher space, W trained JOINTLY with the student (the convention
                    that distinguishes it from oprd_bridge's frozen bridge). Warm-started
                    by closed-form least squares with both backbones frozen, so a failure
                    is attributable to joint training, not random init. Saved to
                    wmap_final.pt.
  minilmv2_selfrel  Projector-free: row-wise KL between teacher and student
                    cosine self-relation matrices (softmax over window tokens).
  ln_dist           Local-window top-k distance matching: teacher's top-k neighbor
                    cosine distances vs student distances to the SAME positions,
                    Huber loss on mean-normalized distance vectors. Dimension-free.
  soft_knn_kl       Soft neighbor-identity KL over teacher's top-k window
                    neighbors, temperature --tau. (Tier-0 M1: control arm only.)
  cka_gram          1 - linear CKA between centered Gram matrices per window.
  pkt_tau           Tier 1: soft_knn_kl with tau as the swept variable.
  opd_topk          POSITIVE CONTROL: plain on-policy distillation on outputs — sparse
                    KL toward the teacher's renormalized top-16 next-token distribution
                    over the student's own rollouts. No hidden states, no bridge. If this
                    arm cannot beat the untrained baseline, the protocol cannot distill
                    and no representation-arm result is interpretable.
  depth_gram        DEPTH-trajectory arm (partner idea, evidence/note-depth-trajectory.md):
                    per token, the pairwise cross-LAYER cosine Gram over the supervised
                    layers — how the token's state evolves with depth. Window-mean-centered
                    on both sides before MSE (tier-0 probe: raw depth-Grams are ~97%
                    token-independent; only the residual is token-specific). Dimension-free,
                    projector-free (cosines are within-model). Needs >=2 supervised layers;
                    intended with a dense band, e.g.
                    --supervised-layer-fracs 0.125,0.25,...,1.0. Nearest prior art is FDD
                    (ACL 2025, vocab-space trajectories) — cite/compare in any write-up.

Rollout-length protocol: the paper generates LONG rollouts and supervises only the last
k=2000 tokens. The validated local protocol uses --rollout-tokens 16000
--supervise-last-k 2000 --require-finished. Training on
short truncated rollouts (the old default) teaches never-concluding behavior and is the
suspected cause of the tier-2 answered_rate collapse.

vLLM runs as a SUBPROCESS (spawned per generation round): the main process offloads
teacher/student/optimizer to CPU first, so the engine gets a clean GPU, and engine
teardown is guaranteed by process exit. Per-request seeds keep Avg@N samples distinct.

VERIFICATION STATUS: loss arms and control flow were smoke-tested locally; the corrected
long-rollout protocol passed a 1.5B CUDA/vLLM/Slurm positive-control gate. External-artifact
Qwen3 OPRD Stage 2, all-layer CKA, and full-rank map memory remain unprofiled. Run the
targeted one-task smokes in cluster/README.md before any array.

Cross-tokenizer pairs (Phi-4-mini -> Qwen) are NOT implemented; the harness exits with
a clear error if teacher/student vocabs differ (tier2 sbatch skips that pair by default).
"""
import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from oprd_bridge import (
    ARTIFACT_TYPE,
    SCHEMA_VERSION as BRIDGE_SCHEMA_VERSION,
    BridgeBank,
    OPRDBridgeError,
    load_bridge_artifact,
    proportional_teacher_layer,
    read_bridge_manifest,
    resolve_layer_map,
    stage1_contract_sha256,
)

DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
AIME = {"aime24": ("Maxwell-Jia/AIME_2024", "train"),
        "aime25": ("MathArena/aime_2025", "train")}
INSTR = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
RELATIONAL = {"minilmv2_selfrel", "ln_dist", "soft_knn_kl", "pkt_tau", "cka_gram",
              "depth_gram"}
REL_WINDOW = 512  # relational losses cap the per-rollout window at this many tokens


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loss", required=True,
                   choices=["oprd_bridge", "linear_map_mse", "minilmv2_selfrel",
                            "ln_dist", "soft_knn_kl", "cka_gram", "pkt_tau", "opd_topk",
                            "depth_gram"])
    p.add_argument("--rollout-tokens", type=int, default=0,
                   help="training-rollout generation length; 0 = same as --max-tokens. "
                        "Paper protocol: long rollouts (>=8k) with --supervise-last-k 2000")
    p.add_argument("--supervise-last-k", type=int, default=0,
                   help="apply the loss only to the last K gen tokens (paper's mask); 0 = all")
    p.add_argument("--require-finished", action="store_true",
                   help="drop training rollouts that hit the --rollout-tokens cap (no natural "
                        "EOS): with --supervise-last-k, a capped trace's 'last K' is not truly "
                        "last, so supervising it is unfaithful to the paper's mask")
    p.add_argument("--bridge-rank", type=int, default=8,
                   help="oprd_bridge: expected rank of the immutable frozen bridge")
    p.add_argument("--bridge-loss-coef", type=float, default=10.0,
                   help="Stage 2 normalized-MSE coefficient (official launcher: 10)")
    p.add_argument("--bridge-artifact", default="",
                   help="completed immutable Stage-1 artifact required by oprd_bridge training")
    p.add_argument("--bridge-manifest-sha256", default="",
                   help="optional independently recorded expected artifact manifest SHA-256")
    p.add_argument("--bridge-stage1-contract-sha256", default="",
                   help="expected canonical Stage-1 contract digest; default is production")
    p.add_argument("--student-revision", default="",
                   help="immutable student model revision; required by oprd_bridge training")
    p.add_argument("--teacher-revision", default="",
                   help="immutable teacher model revision; required by oprd_bridge training")
    p.add_argument("--tokenizer", default="",
                   help="student-compatible tokenizer used by the bridge; default: --student")
    p.add_argument("--tokenizer-revision", default="",
                   help="immutable tokenizer revision; required by oprd_bridge training")
    p.add_argument("--linear-map-init-rows", type=int, default=512,
                   help="row cap for the expensive full-rank least-squares warm start")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--prompts", required=True,
                   help="HF dataset id or local .jsonl prompt source")
    p.add_argument("--dataset-revision", default="",
                   help="immutable HF dataset revision; ignored for local JSONL")
    p.add_argument("--dataset-split", default="train",
                   help="dataset split used for Stage-2 rollout prompts")
    p.add_argument("--prompt-key", default="",
                   help="exact prompt column; empty uses prompt/question/problem fallback")
    p.add_argument("--n-prompts", type=int, default=2000)
    layer_group = p.add_mutually_exclusive_group()
    layer_group.add_argument("--supervised-layers", default="",
                             help="student transformer layers: 'all' or comma-separated 1-based indices")
    layer_group.add_argument("--supervised-layer-fracs", default="",
                             help="backward-compatible fractional student depths; default 0.25,0.5")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-tokens", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=2000)
    p.add_argument("--rollout-refresh", type=int, default=100,
                   help="regenerate rollouts from current policy every N optimizer steps; 0 = never")
    p.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    p.add_argument("--gpu-mem-util", type=float, default=0.85,
                   help="vLLM worker gpu_memory_utilization (main process offloads first)")
    p.add_argument("--track-stable-rank", action="store_true")
    p.add_argument("--eval", default="", help="comma list of: aime24,aime25")
    p.add_argument("--eval-avg-at", type=int, default=16)
    p.add_argument("--eval-only", action="store_true",
                   help="skip training; load <out>/ckpt_final.pt and run --eval only")
    p.add_argument("--resume-from", default="",
                   help="load student weights from this ckpt (e.g. <out>/ckpt_last.pt) and "
                        "continue training; use with --start-step")
    p.add_argument("--start-step", type=int, default=0,
                   help="0-based optimizer step to resume from; --steps is the TOTAL target. "
                        "LR schedule is advanced to this index before the loop.")
    p.add_argument("--diagnostics", default="", help="comma list of: calibrated_mknn,cka,dist_n")
    p.add_argument("--run-label", default="", help="paper-facing arm label recorded in run metadata")
    p.add_argument("--result-namespace", default="", help="shared experiment namespace recorded in run metadata")
    p.add_argument("--source-revision", default="", help="exact source revision recorded in run metadata")
    p.add_argument("--out", required=True)
    return p.parse_args()


def _revision_kwargs(revision):
    if revision and not revision.startswith("local-"):
        return {"revision": revision}
    return {}


def _sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(path):
    root = pathlib.Path(path)
    digest = hashlib.sha256()
    for child in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = child.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(child)))
    return digest.hexdigest()


def _require_sha256(value, option):
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SystemExit(f"{option} must be a lowercase 64-character SHA-256 digest")


def _require_immutable_revision(identifier, revision, option):
    source = pathlib.Path(identifier).expanduser()
    if source.is_file():
        expected = f"local-file-sha256:{_sha256_file(source)}"
    elif source.is_dir():
        expected = f"local-tree-sha256:{_sha256_tree(source)}"
    else:
        expected = None
    if expected is not None:
        if revision != expected:
            raise SystemExit(
                f"{option} does not match the current local source: "
                f"actual={expected!r}, supplied={revision!r}"
            )
        return
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise SystemExit(
            f"{option} must be an immutable 40-character lowercase Hub commit SHA"
        )


def _validate_runtime_args(a):
    positive_integers = {
        "--n-prompts": a.n_prompts,
        "--batch-tokens": a.batch_tokens,
        "--bridge-rank": a.bridge_rank,
        "--linear-map-init-rows": a.linear_map_init_rows,
        "--topk": a.topk,
        "--max-tokens": a.max_tokens,
        "--eval-avg-at": a.eval_avg_at,
    }
    for option, value in positive_integers.items():
        if value < 1:
            raise SystemExit(f"{option} must be positive")
    if not a.eval_only and a.steps < 0:
        raise SystemExit("--steps must be nonnegative")
    if not a.eval_only and a.steps == 0 and (a.resume_from or a.start_step):
        raise SystemExit("--steps 0 is only valid for a fresh untrained baseline")
    for option, value in {
        "--rollout-tokens": a.rollout_tokens,
        "--supervise-last-k": a.supervise_last_k,
        "--rollout-refresh": a.rollout_refresh,
        "--start-step": a.start_step,
    }.items():
        if value < 0:
            raise SystemExit(f"{option} must be nonnegative")
    for option, value in {
        "--lr": a.lr,
        "--tau": a.tau,
        "--bridge-loss-coef": a.bridge_loss_coef,
    }.items():
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"{option} must be finite and positive")
    if not math.isfinite(a.gpu_mem_util) or not 0 < a.gpu_mem_util <= 1:
        raise SystemExit("--gpu-mem-util must be finite and within (0, 1]")
    if not a.dataset_split:
        raise SystemExit("--dataset-split must be non-empty")
    if a.dataset_revision:
        _require_immutable_revision(
            a.prompts, a.dataset_revision, "--dataset-revision"
        )


def _require_oprd_artifact_inputs(a):
    required = {
        "--bridge-artifact": a.bridge_artifact,
        "--bridge-manifest-sha256": a.bridge_manifest_sha256,
        "--student-revision": a.student_revision,
        "--teacher-revision": a.teacher_revision,
        "--tokenizer-revision": a.tokenizer_revision,
        "--dataset-revision": a.dataset_revision,
        "--prompt-key": a.prompt_key,
    }
    missing = [option for option, value in required.items() if not value]
    if missing:
        raise SystemExit(
            "oprd_bridge training requires explicit immutable inputs: " + ", ".join(missing)
        )
    _require_sha256(a.bridge_manifest_sha256, "--bridge-manifest-sha256")
    if a.bridge_stage1_contract_sha256:
        _require_sha256(
            a.bridge_stage1_contract_sha256,
            "--bridge-stage1-contract-sha256",
        )
    _require_immutable_revision(a.student, a.student_revision, "--student-revision")
    _require_immutable_revision(a.teacher, a.teacher_revision, "--teacher-revision")
    _require_immutable_revision(
        a.tokenizer or a.student,
        a.tokenizer_revision,
        "--tokenizer-revision",
    )


def _bridge_reference(manifest, manifest_sha256, artifact_path, stage2_identity):
    stage1 = manifest["stage1"]
    return {
        "artifact_type": manifest["artifact_type"],
        "schema_version": manifest["schema_version"],
        "manifest_sha256": manifest_sha256,
        "path": str(pathlib.Path(artifact_path).expanduser().resolve()),
        "stage2_identity": dict(stage2_identity),
        "identity": dict(manifest["identity"]),
        "bridge": dict(manifest["bridge"]),
        "stage1": {
            "selected_epoch": stage1.get("selected_epoch"),
            "selected_validation_cosine": stage1.get("selected_validation_cosine"),
        },
    }


def _validate_resume_bridge_reference(saved, current):
    if not isinstance(saved, dict):
        raise SystemExit(
            "OPRD resume requires an atomic training state with bridge artifact identity"
        )
    for field in (
        "artifact_type",
        "schema_version",
        "manifest_sha256",
        "stage2_identity",
        "identity",
        "bridge",
        "stage1",
    ):
        if saved.get(field) != current.get(field):
            raise SystemExit(
                f"OPRD resume bridge {field} mismatch: "
                f"checkpoint={saved.get(field)!r}, run={current.get(field)!r}"
            )


def _load_external_bridge(a, student_config, teacher_config, lmap):
    tokenizer_id = a.tokenizer or a.student
    _require_immutable_revision(a.student, a.student_revision, "--student-revision")
    _require_immutable_revision(a.teacher, a.teacher_revision, "--teacher-revision")
    _require_immutable_revision(
        tokenizer_id, a.tokenizer_revision, "--tokenizer-revision"
    )
    expected_contract_sha256 = (
        a.bridge_stage1_contract_sha256 or stage1_contract_sha256()
    )
    if not a.bridge_manifest_sha256:
        raise SystemExit("OPRD bridge loading requires --bridge-manifest-sha256")
    artifact_path = pathlib.Path(a.bridge_artifact).expanduser()
    try:
        manifest, manifest_sha256 = read_bridge_manifest(artifact_path)
        if manifest_sha256 != a.bridge_manifest_sha256:
            raise SystemExit(
                "bridge manifest SHA-256 mismatch: "
                f"artifact={manifest_sha256}, expected={a.bridge_manifest_sha256}"
            )
        bridge, stage1 = load_bridge_artifact(
            artifact_path,
            expected_student_model_id=a.student,
            expected_student_model_revision=a.student_revision,
            expected_teacher_model_id=a.teacher,
            expected_teacher_model_revision=a.teacher_revision,
            expected_tokenizer_id=tokenizer_id,
            expected_tokenizer_revision=a.tokenizer_revision,
            expected_stage1_contract_sha256=expected_contract_sha256,
            expected_layer_map=lmap,
            expected_rank=a.bridge_rank,
            expected_student_dim=student_config.hidden_size,
            expected_teacher_dim=teacher_config.hidden_size,
            expected_manifest_sha256=a.bridge_manifest_sha256,
            device="cpu",
        )
    except OPRDBridgeError as exc:
        raise SystemExit(f"invalid OPRD bridge artifact: {exc}") from exc
    reference = _bridge_reference(
        manifest,
        manifest_sha256,
        artifact_path,
        {
            "prompts": a.prompts,
            "dataset_revision": a.dataset_revision,
            "dataset_split": a.dataset_split,
            "prompt_key": a.prompt_key,
            "n_prompts": a.n_prompts,
            "seed": a.seed,
            "rollout_tokens": a.rollout_tokens,
            "max_tokens": a.max_tokens,
            "supervise_last_k": a.supervise_last_k,
            "require_finished": a.require_finished,
            "rollout_refresh": a.rollout_refresh,
            "batch_tokens": a.batch_tokens,
            "learning_rate": a.lr,
            "bridge_loss_coefficient": a.bridge_loss_coef,
            "backend": a.backend,
        },
    )
    if reference["artifact_type"] != ARTIFACT_TYPE:
        raise SystemExit(f"unsupported bridge artifact type: {reference['artifact_type']!r}")
    if reference["schema_version"] != BRIDGE_SCHEMA_VERSION:
        raise SystemExit(
            f"unsupported bridge artifact schema: {reference['schema_version']!r}"
        )
    return bridge, stage1, reference


def log(msg, out=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if out is not None:
        with open(out / "train.log", "a") as f:
            f.write(line + "\n")


def extract_boxed(text):
    m = list(re.finditer(r"\\boxed\{", text))
    if not m:
        return None
    s = m[-1].end()
    depth, i = 1, s
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[s:i - 1].strip()


def norm_ans(a):
    if a is None:
        return None
    a = str(a).strip().replace(" ", "").replace("\\!", "")
    a = re.sub(r"\.0+$", "", a)  # literal trailing .0 / .00 only — NOT rstrip charset
    return a or "0"


def load_prompt_texts(src, n, seed, *, revision="", split="train", prompt_key=""):
    source = pathlib.Path(src).expanduser()
    if source.suffix == ".jsonl" and source.is_file():
        with source.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        column = prompt_key or "prompt"
        if any(column not in row for row in rows):
            raise SystemExit(f"local prompt source lacks column {column!r}: {source}")
        values = [row[column] for row in rows]
    else:
        from datasets import load_dataset
        kwargs = {"split": split}
        if revision:
            kwargs["revision"] = revision
        dataset = load_dataset(src, **kwargs)
        if prompt_key:
            if prompt_key not in dataset.column_names:
                raise SystemExit(
                    f"prompt column {prompt_key!r} not found; columns={dataset.column_names}"
                )
            column = prompt_key
        else:
            try:
                column = next(
                    candidate
                    for candidate in ("prompt", "question", "problem")
                    if candidate in dataset.column_names
                )
            except StopIteration as exc:
                raise SystemExit(
                    f"no supported prompt column found; columns={dataset.column_names}"
                ) from exc
        values = [row[column] for row in dataset]
    texts = [
        value if isinstance(value, str) else value[0]["content"]
        for value in values
    ]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))[:n]
    return [texts[i] for i in idx]


def chat_ids(tok, prompt):
    if getattr(tok, "chat_template", None):
        out = tok.apply_chat_template([{"role": "user", "content": prompt + INSTR}],
                                      add_generation_prompt=True, tokenize=True)
        ids = out["input_ids"] if hasattr(out, "keys") else out
        return ids[0] if ids and isinstance(ids[0], list) else list(ids)
    return tok.encode("Problem:\n" + prompt + INSTR + "\n\nSolution:\n")


def hidden_forward(model, **kwargs):
    """Run the transformer backbone without materializing unused vocabulary logits."""
    backbone = getattr(model, "model", None)
    if backbone is None:
        return model(output_hidden_states=True, **kwargs)
    return backbone(output_hidden_states=True, **kwargs)


# ---------------------------------------------------------------- generation

def _worker_main(spec_path):
    """vLLM generation worker: runs in its own process so engine memory/teardown
    never touches the training process. Reads a JSON spec, writes gen_ids JSON."""
    spec = json.loads(pathlib.Path(spec_path).read_text())
    from vllm import LLM, SamplingParams
    try:
        from vllm import TokensPrompt
    except ImportError:
        from vllm.inputs import TokensPrompt
    llm = LLM(model=spec["model_dir"], dtype="bfloat16",
              gpu_memory_utilization=spec["gpu_mem_util"])
    prompts = [TokensPrompt(prompt_token_ids=p) for p in spec["prompt_ids"]]
    sps = [SamplingParams(temperature=0.6, top_p=0.95,
                          max_tokens=spec["max_tokens"], seed=s) for s in spec["seeds"]]
    outs = llm.generate(prompts, sps)
    gen = [list(o.outputs[0].token_ids) for o in outs]
    pathlib.Path(spec["out_json"]).write_text(json.dumps({"gen_ids": gen}))


def _move_optimizer(opt, device):
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)


def make_generator(student, teacher, opt, tok, a, out, aux_modules=None):
    """Returns gen(prompt_texts, seeds) -> [{prompt_ids, gen_ids}].
    seeds: one int per prompt (drives per-request vLLM seeds / hf global seeding)."""
    aux_modules = aux_modules if aux_modules is not None else []
    use_vllm = a.backend == "vllm" or (a.backend == "auto" and DEV == "cuda")
    if use_vllm:
        try:
            import vllm  # noqa: F401
        except ImportError:
            log("vllm not importable, falling back to hf generate", out)
            use_vllm = False

    def gen(prompt_texts, seeds, max_tokens):
        assert len(prompt_texts) == len(seeds)
        pid = [chat_ids(tok, p) for p in prompt_texts]
        if use_vllm:
            tmp = (out / "tmp_policy").resolve()
            student.save_pretrained(tmp, safe_serialization=True)
            tok.save_pretrained(tmp)
            # free the GPU for the worker: offload everything, restore after
            student.to("cpu")
            if teacher is not None:
                teacher.to("cpu")
            if opt is not None:
                _move_optimizer(opt, "cpu")
            for module in aux_modules:
                module.to("cpu")
            torch.cuda.empty_cache()
            # absolute paths: relative model_dir breaks if tmp_policy vanishes mid-flight
            # (and HF then mis-parses it as a hub repo id)
            spec = {"model_dir": str(tmp), "prompt_ids": pid, "seeds": list(seeds),
                    "max_tokens": max_tokens, "gpu_mem_util": a.gpu_mem_util,
                    "out_json": str((out / "worker_out.json").resolve())}
            spec_path = (out / "worker_spec.json").resolve()
            spec_path.write_text(json.dumps(spec))
            log(f"vllm worker: {len(pid)} prompts", out)
            r = subprocess.run([sys.executable, __file__, "--_worker", str(spec_path)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"vllm worker failed:\n{r.stderr[-4000:]}")
            gen_ids = json.loads((out / "worker_out.json").read_text())["gen_ids"]
            shutil.rmtree(tmp, ignore_errors=True)
            (out / "worker_out.json").unlink(missing_ok=True)
            student.to(DEV)
            if teacher is not None:
                teacher.to(DEV)
            if opt is not None:
                _move_optimizer(opt, DEV)
            for module in aux_modules:
                module.to(DEV)
            return [{"prompt_ids": pid[i], "gen_ids": gen_ids[i]} for i in range(len(pid))]
        # hf fallback (mps/cpu, or --backend hf): slow but exercises the same contract
        res = []
        student.eval()
        with torch.no_grad():
            for i, p in enumerate(pid):
                torch.manual_seed(seeds[i])  # per-sample seeding (global RNG drives sampling)
                ids = torch.tensor([p], device=DEV)
                seq = student.generate(ids, do_sample=True, temperature=0.6, top_p=0.95,
                                       max_new_tokens=max_tokens,
                                       pad_token_id=tok.eos_token_id)
                res.append({"prompt_ids": p, "gen_ids": seq[0, ids.shape[1]:].tolist()})
                if (i + 1) % 16 == 0:
                    log(f"  hf-gen {i+1}/{len(pid)}", out)
        student.train()
        return res

    return gen


@torch.no_grad()
def teacher_prefill(teacher, rollouts, layers, out, max_tok, last_k=0,
                    topk_targets=False, store_hidden=True):
    """Cache supervised teacher states and/or sparse output targets without unused logits."""
    entries = []
    for i, r in enumerate(rollouts):
        glen = min(len(r["gen_ids"]), max_tok)
        ids = torch.tensor([r["prompt_ids"] + r["gen_ids"][:glen]], device=DEV)
        gstart = len(r["prompt_ids"])
        off = max(0, glen - last_k) if last_k else 0
        e = {"off": off, "h": {}}
        if topk_targets:
            o = teacher(input_ids=ids, output_hidden_states=False, use_cache=False)
            lg = o.logits[0, gstart + off - 1:gstart + glen - 1].float()
            p = torch.softmax(lg, -1)
            tp, ti = p.topk(16, dim=-1)
            e["tk_p"] = (tp / tp.sum(-1, keepdim=True)).half().cpu()
            e["tk_ids"] = ti.int().cpu()
            del o, lg, p, tp, ti
        if store_hidden:
            h = hidden_forward(teacher, input_ids=ids, use_cache=False)
            e["h"] = {
                li: h.hidden_states[li][0, gstart + off:gstart + glen].bfloat16().cpu()
                for li in layers}
            del h
        entries.append(e)
        if DEV == "mps":
            torch.mps.empty_cache()
        if (i + 1) % 25 == 0:
            log(f"  teacher prefill {i+1}/{len(rollouts)}", out)
    return entries


# ---------------------------------------------------------------- loss arms

@torch.no_grad()
def collect_pool_states(student, rolls, tstates, lay_s, lmap, max_tok, out, cap=0,
                        seed=0):
    """Collect seeded aligned rows for the linear-map calibration warm start."""
    lengths = [max(0, min(len(r["gen_ids"]), max_tok) - ts["off"])
               for r, ts in zip(rolls, tstates)]
    total = sum(lengths)
    if total < 1:
        raise SystemExit("linear-map calibration pool has no supervised token rows")
    take = total if not cap else min(cap, total)
    chosen = (np.arange(total) if take == total else
              np.sort(np.random.default_rng(seed).choice(total, size=take, replace=False)))
    first = next(ts for ts, length in zip(tstates, lengths) if length)
    student_dim = student.config.hidden_size
    teacher_dim = first["h"][lmap[lay_s[0]]].shape[-1]
    S = {ls: torch.empty((take, student_dim), dtype=torch.bfloat16) for ls in lay_s}
    T = {ls: torch.empty((take, teacher_dim), dtype=torch.bfloat16) for ls in lay_s}
    rollout_slices = []
    student.eval()
    base = kept = 0
    for r, ts, length in zip(rolls, tstates, lengths):
        stop = base + length
        lo, hi = np.searchsorted(chosen, [base, stop])
        local = chosen[lo:hi] - base
        base = stop
        row_start, row_stop = kept, kept + len(local)
        rollout_slices.append((row_start, row_stop))
        kept = row_stop
        if not len(local):
            continue
        glen = min(len(r["gen_ids"]), max_tok)
        off = ts["off"]
        ids = torch.tensor([r["prompt_ids"] + r["gen_ids"][:glen]], device=DEV)
        gstart = len(r["prompt_ids"])
        o = hidden_forward(student, input_ids=ids, use_cache=False)
        idx = torch.as_tensor(local, dtype=torch.long)
        for ls in lay_s:
            hs = o.hidden_states[ls][0, gstart + off:gstart + glen].bfloat16().cpu()
            S[ls][row_start:row_stop].copy_(hs.index_select(0, idx))
            T[ls][row_start:row_stop].copy_(
                ts["h"][lmap[ls]][:length].index_select(0, idx))
        del o
        if DEV == "mps":
            torch.mps.empty_cache()
    student.train()
    action = "retained" if take == total else "sampled"
    log(f"linear-map calibration: {action} {take}/{total} tokens over {len(lay_s)} layers", out)
    return S, T, rollout_slices


def save_training_state(path, student, wmap, step, losses, bridge_reference=None):
    """Atomically save synchronized weight-level rescue state and loss history."""
    payload = {"version": 2, "step": step, "losses": list(losses),
               "student": {k: v.to(torch.bfloat16).cpu()
                           for k, v in student.state_dict().items()},
               "wmap": ({k: v.detach().cpu() for k, v in wmap.state_dict().items()}
                        if wmap is not None else None),
               "bridge_reference": bridge_reference}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def init_linear_map(wmap, S, T, out):
    """Closed-form least-squares warm start for the TinyBERT-style map (removes the
    random-init failure mode; joint training afterwards is the arm's defining choice)."""
    for ls in S:
        W = torch.linalg.lstsq(S[ls].float(), T[ls].float()).solution.T  # (d_t, d_s)
        wmap[str(ls)].weight.data.copy_(W.to(DEV))
    log(f"linear_map warm-started via least squares on {len(next(iter(S.values())))} rows "
        "(both backbones frozen)", out)


def cosine_sim(h):
    hn = F.normalize(h.float(), dim=-1)
    return hn @ hn.T


def loss_relational(arm, hs, ht, tau, topk):
    """hs (T,d_s), ht (T,d_t) — window = this rollout's gen region (cap REL_WINDOW)."""
    T = min(hs.shape[0], REL_WINDOW)
    hs, ht = hs[-T:], ht[-T:]  # LAST T of the supervised span (a capped span's tail is
    # the rollout's true tail under --require-finished); hs[:T] was a bug — it supervised
    # the first 512 of the last-2k window, where teacher-student divergence is lowest
    St, Ss = cosine_sim(ht), cosine_sim(hs)
    eye = torch.eye(T, device=hs.device, dtype=torch.bool)
    if arm == "minilmv2_selfrel":
        lt = torch.log_softmax(St.masked_fill(eye, -1e4), -1)
        ls = torch.log_softmax(Ss.masked_fill(eye, -1e4), -1)
        return F.kl_div(ls, lt, log_target=True, reduction="batchmean")
    k = min(topk, T - 1)
    nbr = St.masked_fill(eye, -1e4).topk(k, dim=-1).indices  # teacher-chosen neighbors
    st = St.gather(-1, nbr)
    ss = Ss.gather(-1, nbr)
    if arm in ("soft_knn_kl", "pkt_tau"):
        return F.kl_div(torch.log_softmax(ss / tau, -1),
                        torch.log_softmax(st / tau, -1),
                        log_target=True, reduction="batchmean")
    if arm == "ln_dist":
        dt, ds = (1 - st), (1 - ss)
        dt = dt / dt.mean().clamp_min(1e-6)
        ds = ds / ds.mean().clamp_min(1e-6)
        return F.huber_loss(ds, dt)
    raise ValueError(arm)


def linear_cka(X, Y):
    X = X.float() - X.float().mean(0)
    Y = Y.float() - Y.float().mean(0)
    xy = (X.T @ Y).norm() ** 2
    return xy / ((X.T @ X).norm() * (Y.T @ Y).norm()).clamp_min(1e-9)


def loss_depth_gram(s_list, t_list):
    """s_list/t_list: per-layer (T,d) states, same layer order, dims may differ across
    models. Per-token cross-layer cosine Gram, window-mean-centered on both sides
    (calibration: raw depth-Grams are ~97% token-independent, tier-0 probe), MSE on
    the residuals. Slices to the LAST REL_WINDOW tokens like the other relational arms."""
    T = min(s_list[0].shape[0], REL_WINDOW)

    def gram(hl):
        H = torch.stack([F.normalize(h[-T:].float(), dim=-1) for h in hl])  # (L,T,d)
        G = torch.einsum("ltd,mtd->tlm", H, H)
        iu = torch.triu_indices(len(hl), len(hl), offset=1)
        return G[:, iu[0], iu[1]]  # (T, L*(L-1)/2)

    gs, gt = gram(s_list), gram(t_list)
    return F.mse_loss(gs - gs.mean(0), gt - gt.mean(0))


def compute_loss(arm, s_states, t_states, layers, bridge, wmap, tau, topk,
                 bridge_loss_coef=1.0):
    if arm == "depth_gram":
        if len(layers) < 2:
            raise SystemExit("depth_gram needs >=2 supervised layers "
                             "(use a dense --supervised-layer-fracs band)")
        return loss_depth_gram([s_states[ls] for ls in layers],
                               [t_states[ls].to(s_states[layers[0]].device).float()
                                for ls in layers])
    per_layer = []
    for li in layers:
        hs = s_states[li]
        ht = t_states[li].to(hs.device).float()
        if arm == "oprd_bridge":
            a = F.normalize(bridge.project_student(li, hs), dim=-1)
            b = F.normalize(bridge.project_teacher(li, ht), dim=-1)
            per_layer.append(F.mse_loss(a, b))  # paper Eq. 14 / official normalized MSE
        elif arm == "linear_map_mse":
            per_layer.append(F.mse_loss(wmap[str(li)](hs), ht))
        elif arm == "cka_gram":
            T = min(hs.shape[0], REL_WINDOW)
            per_layer.append(1 - linear_cka(hs[-T:], ht[-T:]))
        else:
            per_layer.append(loss_relational(arm, hs, ht, tau, topk))
    loss = torch.stack(per_layer).mean()
    return loss * bridge_loss_coef if arm == "oprd_bridge" else loss


# ---------------------------------------------------------------- eval / diagnostics

def eval_aime(gen, tok, names, avg_at, seed, out, max_tokens):
    """One batched generation call per dataset (per-request seeds keep samples distinct).
    Reports truncation_rate (hit the cap with no boxed answer) so the token budget is
    empirically justified: if ~0, the cap did not bind and cannot explain low scores."""
    from datasets import load_dataset
    results = {}
    for name in names:
        repo, split = AIME[name]
        ds = load_dataset(repo, split=split)
        pcol = next(c for c in ["Problem", "problem", "question"] if c in ds.column_names)
        acol = next(c for c in ["Answer", "answer", "solution"] if c in ds.column_names)
        prompts, seeds = [], []
        for pi, row in enumerate(ds):
            prompts += [row[pcol]] * avg_at
            seeds += [seed * 1_000_000 + pi * 1000 + c for c in range(avg_at)]
        log(f"eval {name}: generating {len(prompts)} samples ({len(ds)}x{avg_at})", out)
        rolls = gen(prompts, seeds, max_tokens)
        correct, total, answered, truncated = 0, 0, 0, 0
        for pi, row in enumerate(ds):
            gt = norm_ans(row[acol])
            for c in range(avg_at):
                gids = rolls[pi * avg_at + c]["gen_ids"]
                pred = norm_ans(extract_boxed(tok.decode(gids)))
                correct += int(pred is not None and pred == gt)
                answered += int(pred is not None)
                truncated += int(pred is None and len(gids) >= max_tokens)
                total += 1
        results[name] = {"avg_at": avg_at, "accuracy": correct / max(total, 1), "n": total,
                         "answered_rate": answered / max(total, 1),
                         "truncation_rate": truncated / max(total, 1)}
        log(f"eval {name}: acc={results[name]['accuracy']:.4f} "
            f"answered={results[name]['answered_rate']:.3f} "
            f"truncated={results[name]['truncation_rate']:.3f}", out)
    return results


@torch.no_grad()
def diagnostics(names, s_pool, t_pool, rollout_texts):
    d = {}
    if "cka" in names:
        d["cka"] = float(linear_cka(s_pool, t_pool))
    if "calibrated_mknn" in names:
        n = min(2000, s_pool.shape[0])
        idx = torch.randperm(s_pool.shape[0])[:n]
        Ss, St = cosine_sim(s_pool[idx]), cosine_sim(t_pool[idx])
        eye = torch.eye(n, dtype=torch.bool)
        ks, kt = [S.masked_fill(eye, -1e4).topk(10, -1).indices for S in (Ss, St)]
        agree = torch.tensor([(np.intersect1d(ks[i], kt[i]).size / 10) for i in range(n)])
        d["mknn_k10"] = float(agree.mean())
        d["mknn_null"] = 10 / (n - 1)  # chance level
    if "dist_n" in names:
        grams, tot = set(), 0
        for t in rollout_texts:
            w = t.split()
            gs = list(zip(w, w[1:], w[2:], w[3:]))
            grams.update(gs)
            tot += len(gs)
        d["dist4"] = len(grams) / max(tot, 1)
    return d


def _run_metadata(a):
    return {
        "schema_version": 1,
        "run_kind": (
            "evaluation" if a.eval_only else
            "untrained_baseline" if a.steps == 0 else
            "training"
        ),
        "arm_label": a.run_label or a.loss,
        "result_namespace": a.result_namespace or None,
        "source_revision": a.source_revision or None,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    }


def _run_untrained_baseline(a, out, config_path, run_metadata):
    tok = AutoTokenizer.from_pretrained(
        a.tokenizer, **_revision_kwargs(a.tokenizer_revision)
    )
    student = AutoModelForCausalLM.from_pretrained(
        a.student, dtype=torch.float32, **_revision_kwargs(a.student_revision)
    ).to(DEV).eval()
    config_path.write_text(json.dumps(vars(a) | {"run_metadata": run_metadata}, indent=1))

    save_training_state(out / "train_state_last.pt", student, None, 0, [])
    torch.save(
        {key: value.to(torch.bfloat16).cpu() for key, value in student.state_dict().items()},
        out / "ckpt_final.pt",
    )
    res = {
        "run_metadata": run_metadata,
        "losses": [],
        "resumed_from": None,
        "start_step": 0,
    }
    (out / "results.json").write_text(json.dumps(res, indent=1))

    eval_names = [name for name in a.eval.split(",") if name]
    if eval_names:
        gen = make_generator(student, None, None, tok, a, out)
        res["eval"] = eval_aime(
            gen, tok, eval_names, a.eval_avg_at, a.seed, out, a.max_tokens
        )
        (out / "results.json").write_text(json.dumps(res, indent=1))
    log(f"DONE -> {out}/results.json", out)


# ---------------------------------------------------------------- main

def main():
    a = parse_args()
    a.tokenizer = a.tokenizer or a.student
    _validate_runtime_args(a)
    if a.loss == "oprd_bridge" and not a.eval_only and a.steps > 0:
        _require_oprd_artifact_inputs(a)
        a.bridge_stage1_contract_sha256 = (
            a.bridge_stage1_contract_sha256 or stage1_contract_sha256()
        )
    out = pathlib.Path(a.out)
    if not a.eval_only and not a.resume_from and out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"fresh training requires a new or empty --out directory: {out}"
        )
    out.mkdir(parents=True, exist_ok=True)
    invocation_config = ("eval_config.json" if a.eval_only else
                         "resume_config.json" if a.resume_from else "config.json")
    config_path = out / invocation_config
    run_metadata = _run_metadata(a)
    config_path.write_text(json.dumps(vars(a) | {"run_metadata": run_metadata}, indent=1))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    log(f"harness: loss={a.loss} device={DEV}", out)

    if not a.eval_only and a.steps == 0:
        _run_untrained_baseline(a, out, config_path, run_metadata)
        return

    student_config = AutoConfig.from_pretrained(
        a.student, **_revision_kwargs(a.student_revision)
    )
    teacher_config = AutoConfig.from_pretrained(
        a.teacher, **_revision_kwargs(a.teacher_revision)
    )
    nl_t = teacher_config.num_hidden_layers
    nl_s = student_config.num_hidden_layers
    try:
        lmap = resolve_layer_map(a.supervised_layers, a.supervised_layer_fracs, nl_s, nl_t)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    lay_s = list(lmap)
    lay_t = sorted(set(lmap.values()))

    bridge = None
    bridge_meta = None
    bridge_reference = None
    if a.loss == "oprd_bridge" and not a.eval_only:
        bridge, bridge_meta, bridge_reference = _load_external_bridge(
            a, student_config, teacher_config, lmap
        )
        log(
            "validated immutable bridge "
            f"manifest={bridge_reference['manifest_sha256']} before model generation",
            out,
        )

    config = vars(a) | {"resolved_student_layers": lay_s,
                        "resolved_teacher_layers": lay_t,
                        "resolved_layer_map": {str(k): v for k, v in lmap.items()},
                        "bridge_reference": bridge_reference,
                        "run_metadata": run_metadata}
    config_path.write_text(json.dumps(config, indent=1))
    log(f"supervised layer map={list(lmap.items())}", out)

    tok_t = AutoTokenizer.from_pretrained(
        a.teacher, **_revision_kwargs(a.teacher_revision)
    )
    tok = AutoTokenizer.from_pretrained(
        a.tokenizer, **_revision_kwargs(a.tokenizer_revision)
    )
    if tok_t.get_vocab() != tok.get_vocab():
        raise SystemExit("cross-tokenizer pair detected: token-aligned losses are undefined "
                         "here — this harness supports shared-vocab pairs only (see docstring).")

    teacher = AutoModelForCausalLM.from_pretrained(
        a.teacher, dtype=torch.bfloat16, **_revision_kwargs(a.teacher_revision)
    ).to(DEV).eval()
    student = AutoModelForCausalLM.from_pretrained(
        a.student, dtype=torch.float32, **_revision_kwargs(a.student_revision)
    )
    student.gradient_checkpointing_enable()
    student.config.use_cache = False
    student.to(DEV).train()

    wmap = None
    aux_modules = []
    params = list(student.parameters())
    if a.loss == "linear_map_mse":
        wmap = torch.nn.ModuleDict({
            str(ls): torch.nn.Linear(student.config.hidden_size,
                                     teacher.config.hidden_size, bias=False)
            for ls in lay_s}).to(DEV).float()
        params += list(wmap.parameters())
        aux_modules.append(wmap)
    if bridge is not None:
        bridge.to(DEV).float().freeze()
        aux_modules.append(bridge)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * s / max(a.steps, 1))))

    gen = make_generator(student, teacher, opt, tok, a, out, aux_modules)
    res = {"run_metadata": run_metadata}

    if a.eval_only:
        ckpt = out / "ckpt_final.pt"
        st = torch.load(ckpt, map_location="cpu")
        student.load_state_dict({k: v.to(torch.float32) for k, v in st.items()})
        if (out / "results.json").exists():  # merge, don't clobber the training results
            res = json.loads((out / "results.json").read_text())
        log(f"eval-only: loaded {ckpt}", out)
    else:
        start = a.start_step
        if start < 0 or start > a.steps or (a.steps and start == a.steps):
            raise SystemExit(f"--start-step {start} out of range for --steps {a.steps}")
        resume_ckpt = None
        resume_losses, resume_wmap = [], None
        if a.resume_from:
            resume_ckpt = pathlib.Path(a.resume_from)
            if not resume_ckpt.is_file():
                resume_ckpt = out / a.resume_from
            state_path = (
                resume_ckpt if resume_ckpt.name == "train_state_last.pt" else
                resume_ckpt.parent / "train_state_last.pt"
                if resume_ckpt.name in {"ckpt_last.pt", "ckpt_final.pt"} else None
            )
            if state_path is not None and state_path.is_file():
                state = torch.load(state_path, map_location="cpu")
                if state.get("step") != start:
                    raise SystemExit(f"resume state step={state.get('step')} but --start-step={start}")
                if a.loss == "oprd_bridge":
                    _validate_resume_bridge_reference(
                        state.get("bridge_reference"), bridge_reference
                    )
                st = state["student"]
                resume_losses = list(state.get("losses", []))
                resume_wmap = state.get("wmap")
                log(f"resume: loaded atomic training state from {state_path}", out)
            else:
                if a.loss == "oprd_bridge":
                    raise SystemExit(
                        "OPRD resume requires train_state_last.pt with bridge artifact identity"
                    )
                st = torch.load(resume_ckpt, map_location="cpu")
                log(f"resume: loaded legacy weights-only checkpoint from {resume_ckpt}", out)
            student.load_state_dict({k: v.to(torch.float32) for k, v in st.items()})
            log(f"resume: start-step={start}, total steps={a.steps}", out)
        elif start:
            raise SystemExit("--start-step > 0 requires --resume-from")
        if a.loss == "linear_map_mse" and resume_ckpt is not None:
            if resume_wmap is not None:
                wmap.load_state_dict(resume_wmap)
                log("resume: restored synchronized linear map from training state", out)
            else:
                map_name = "wmap_final.pt" if resume_ckpt.name == "ckpt_final.pt" else "wmap_last.pt"
                map_path = resume_ckpt.parent / map_name
                if not map_path.is_file():
                    raise SystemExit(f"linear_map_mse resume requires {map_path}")
                wmap.load_state_dict(torch.load(map_path, map_location="cpu"))
                log(f"resume: loaded legacy linear map from {map_path}", out)

        # advance LR schedule to match the resumed step index
        for _ in range(start):
            sched.step()

        prompts = load_prompt_texts(
            a.prompts,
            a.n_prompts,
            a.seed,
            revision=a.dataset_revision,
            split=a.dataset_split,
            prompt_key=a.prompt_key,
        )
        diag_names = [d for d in a.diagnostics.split(",") if d]
        need_teacher_hidden = (a.loss != "opd_topk" or a.track_stable_rank or bool(diag_names))

        rt = a.rollout_tokens or a.max_tokens  # training-rollout generation budget

        def refresh(step):
            n_pool = min(len(prompts), max(64, a.batch_tokens // 256))
            rng = np.random.default_rng(a.seed + step)
            sel = [prompts[i] for i in rng.permutation(len(prompts))[:n_pool]]
            seeds = [a.seed * 1_000_000 + step * 1000 + i for i in range(len(sel))]
            log(f"refresh @step {step}: generating {len(sel)} rollouts (<= {rt} tok)", out)
            rolls = gen(sel, seeds, rt)
            rolls = [r for r in rolls if len(r["gen_ids"]) >= 8]
            n_fin = sum(len(r["gen_ids"]) < rt for r in rolls)
            log(f"refresh @step {step}: {n_fin}/{len(rolls)} rollouts finished naturally", out)
            if a.require_finished:
                rolls = [r for r in rolls if len(r["gen_ids"]) < rt]
            if not rolls:
                return None, None  # caller keeps the previous pool
            tstates = teacher_prefill(teacher, rolls, lay_t, out, rt,
                                      last_k=a.supervise_last_k,
                                      topk_targets=(a.loss == "opd_topk"),
                                      store_hidden=need_teacher_hidden)
            return rolls, tstates

        rolls, tstates = refresh(start)
        if not rolls:
            raise SystemExit(
                "initial rollout pool is empty (all generations < 8 tokens, or none finished "
                "under --require-finished — raise --rollout-tokens or drop the flag)")
        if a.loss == "linear_map_mse" and resume_ckpt is None:
            S_pool, T_pool, _ = collect_pool_states(
                student, rolls, tstates, lay_s, lmap, rt, out,
                cap=a.linear_map_init_rows, seed=a.seed)
            init_linear_map(wmap, S_pool, T_pool, out)
            del S_pool, T_pool

        losses, ri = list(resume_losses), 0
        if not losses and (out / "results.json").exists():
            prev = json.loads((out / "results.json").read_text())
            losses = list(prev.get("losses", []))[:start]
        for step in range(start, a.steps):
            if a.rollout_refresh and step and step % a.rollout_refresh == 0:
                new_rolls, new_tstates = refresh(step)
                if new_rolls is None:
                    log(f"refresh @step {step}: pool empty, keeping previous rollouts", out)
                else:
                    rolls, tstates, ri = new_rolls, new_tstates, 0
            opt.zero_grad(set_to_none=True)
            toks_done, batch_losses = 0, []
            while toks_done < a.batch_tokens:
                r, ts = rolls[ri % len(rolls)], tstates[ri % len(tstates)]
                ri += 1
                glen = min(len(r["gen_ids"]), rt)
                off = ts["off"]
                sup = glen - off  # tokens actually supervised (last-k mask)
                ids = torch.tensor([r["prompt_ids"] + r["gen_ids"][:glen]], device=DEV)
                gstart = len(r["prompt_ids"])
                if a.loss == "opd_topk":
                    o = student(input_ids=ids)
                    lq = torch.log_softmax(
                        o.logits[0, gstart + off - 1:gstart + glen - 1].float(), -1)
                    tk_ids = ts["tk_ids"].to(DEV).long()
                    tk_p = ts["tk_p"].to(DEV).float()
                    loss = -(tk_p * lq.gather(-1, tk_ids)).sum(-1).mean()
                else:
                    o = hidden_forward(student, input_ids=ids, use_cache=False)
                    s_states = {ls: o.hidden_states[ls][0, gstart + off:gstart + glen].float()
                                for ls in lay_s}
                    t_states = {ls: ts["h"][lmap[ls]][:sup] for ls in lay_s}
                    loss = compute_loss(a.loss, s_states, t_states, lay_s, bridge, wmap,
                                        a.tau, a.topk, a.bridge_loss_coef)
                # weight by tokens the loss actually saw, clipped to the remaining budget
                eff = min(sup, REL_WINDOW) if a.loss in RELATIONAL else sup
                w = min(eff, a.batch_tokens - toks_done) / a.batch_tokens
                loss_value = float(loss.detach().item())
                if not math.isfinite(loss_value):
                    raise RuntimeError(
                        f"non-finite training loss at optimizer step {step + 1}"
                    )
                (loss * w).backward()
                batch_losses.append(loss_value)
                toks_done += eff
                del o
                if DEV == "mps":
                    torch.mps.empty_cache()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not bool(torch.isfinite(grad_norm).item()):
                raise RuntimeError(
                    f"non-finite gradient norm at optimizer step {step + 1}"
                )
            opt.step()
            sched.step()
            losses.append(float(np.mean(batch_losses)))
            if (step + 1) % 10 == 0 or step == start:
                log(f"step {step+1}/{a.steps} loss={losses[-1]:.5f} "
                    f"lr={sched.get_last_lr()[0]:.2e}", out)
            if (step + 1) % 100 == 0:
                save_training_state(
                    out / "train_state_last.pt", student, wmap,
                    step + 1, losses, bridge_reference=bridge_reference,
                )
                torch.save({k: v.to(torch.bfloat16).cpu()
                            for k, v in student.state_dict().items()}, out / "ckpt_last.pt")
                if wmap is not None:  # compatibility artifact; exact resume uses train_state_last
                    torch.save({k: v.detach().cpu() for k, v in wmap.state_dict().items()},
                               out / "wmap_last.pt")

        save_training_state(
            out / "train_state_last.pt", student, wmap,
            a.steps, losses, bridge_reference=bridge_reference,
        )
        torch.save({k: v.to(torch.bfloat16).cpu() for k, v in student.state_dict().items()},
                   out / "ckpt_final.pt")
        if wmap is not None:
            torch.save(wmap.state_dict(), out / "wmap_final.pt")
        res["losses"] = losses
        res["resumed_from"] = a.resume_from or None
        res["start_step"] = start
        res["student_layers"] = lay_s
        res["teacher_layers"] = lay_t
        res["layer_map"] = {str(k): v for k, v in lmap.items()}
        if bridge_meta is not None:
            res["bridge_stage1"] = bridge_meta
            res["bridge_reference"] = bridge_reference

        if a.track_stable_rank or diag_names:
            with torch.no_grad():
                sp, tp = [], []
                for r, ts in zip(rolls[:20], tstates[:20]):
                    glen = min(len(r["gen_ids"]), rt)
                    off = ts["off"]
                    ids = torch.tensor([r["prompt_ids"] + r["gen_ids"][:glen]], device=DEV)
                    o = hidden_forward(student, input_ids=ids, use_cache=False)
                    sp.append(o.hidden_states[lay_s[-1]][0, len(r["prompt_ids"]) + off:
                                                         len(r["prompt_ids"]) + glen].float().cpu())
                    tp.append(ts["h"][lmap[lay_s[-1]]][:glen - off].float())
                    del o
            if diag_names:
                res["diagnostics"] = diagnostics(diag_names, torch.cat(sp), torch.cat(tp),
                                                 [tok.decode(r["gen_ids"]) for r in rolls[:50]])
            if a.track_stable_rank:
                X = torch.cat(sp)
                sv = torch.linalg.svdvals(X - X.mean(0))
                res["stable_rank_final_layer"] = float((sv**2).sum() / (sv[0]**2).clamp_min(1e-9))

        # persist training results BEFORE eval so a walltime kill during eval loses only eval
        (out / "results.json").write_text(json.dumps(res, indent=1))

    eval_names = [e for e in a.eval.split(",") if e]
    if eval_names:
        student.eval()
        res["eval"] = eval_aime(gen, tok, eval_names, a.eval_avg_at, a.seed, out, a.max_tokens)
        (out / "results.json").write_text(json.dumps(res, indent=1))
    log(f"DONE -> {out}/results.json", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--_worker":
        _worker_main(sys.argv[2])
    else:
        main()
