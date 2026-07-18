"""Shared training harness for Tier 1/2 OPRD experiments.

Full on-policy loop: student rollout generation (vLLM in an isolated subprocess on CUDA,
transformers fallback elsewhere), frozen-teacher prefill over student rollouts, seven loss
arms, AdamW with token-budgeted gradient accumulation, periodic rollout refresh, AIME Avg@N
eval, and diagnostics.

Loss arms (adversarial review §5.5/§5.7; supervised on generated-region tokens only):
  oprd_bridge       OPRD-Bridge paper recipe, faithful two-phase (digest §1.3):
                    Stage 1 with BOTH backbones frozen — P_T = top-r teacher-covariance
                    PCA (r = --bridge-rank, paper default 8), P_S = linear map fit by
                    closed-form least squares (exact optimum of the paper's convex
                    Stage-1 objective); everything then frozen. Stage 2: per-token L2
                    distance between L2-normalized projections, student-only gradients.
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

Rollout-length protocol: the paper generates LONG rollouts and supervises only the last
k=2000 tokens. Reproduce with --rollout-tokens 12000 --supervise-last-k 2000. Training on
short truncated rollouts (the old default) teaches never-concluding behavior and is the
suspected cause of the tier-2 answered_rate collapse.

vLLM runs as a SUBPROCESS (spawned per generation round): the main process offloads
teacher/student/optimizer to CPU first, so the engine gets a clean GPU, and engine
teardown is guaranteed by process exit. Per-request seeds keep Avg@N samples distinct.

VERIFICATION STATUS: loss arms, training loop, refresh, eval plumbing, and --eval-only
smoke-tested end-to-end on Apple MPS (Qwen2.5-0.5B; hf backend). Loss math, gradient
flow, KL directions, and token alignment additionally verified by an independent
adversarial review (2026-07-13). The vLLM worker is written against current vLLM docs
(TokensPrompt API) but is UNTESTED on CUDA — run the 1-task smoke in cluster/README.md
(which now includes a rollout-refresh step) before submitting arrays.

Cross-tokenizer pairs (Phi-4-mini -> Qwen) are NOT implemented; the harness exits with
a clear error if teacher/student vocabs differ (tier2 sbatch skips that pair by default).
"""
import argparse
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = ("cuda" if torch.cuda.is_available()
       else "mps" if torch.backends.mps.is_available() else "cpu")
AIME = {"aime24": ("Maxwell-Jia/AIME_2024", "train"),
        "aime25": ("MathArena/aime_2025", "train")}
INSTR = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
RELATIONAL = {"minilmv2_selfrel", "ln_dist", "soft_knn_kl", "pkt_tau", "cka_gram"}
REL_WINDOW = 512  # relational losses cap the per-rollout window at this many tokens


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loss", required=True,
                   choices=["oprd_bridge", "linear_map_mse", "minilmv2_selfrel",
                            "ln_dist", "soft_knn_kl", "cka_gram", "pkt_tau", "opd_topk"])
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
                   help="oprd_bridge: rank r of the frozen PCA/projector bridge (paper default 8)")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--prompts", required=True,
                   help="HF dataset id (prompt/question column) or local .jsonl with 'prompt'")
    p.add_argument("--n-prompts", type=int, default=2000)
    p.add_argument("--supervised-layer-fracs", default="0.25,0.5")
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
    p.add_argument("--diagnostics", default="", help="comma list of: calibrated_mknn,cka,dist_n")
    p.add_argument("--out", required=True)
    return p.parse_args()


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


def load_prompt_texts(src, n, seed):
    if src.endswith(".jsonl"):
        rows = [json.loads(l) for l in open(src)]
        texts = [r["prompt"] for r in rows]
    else:
        from datasets import load_dataset
        ds = load_dataset(src, split="train")
        col = next(c for c in ["prompt", "question", "problem"] if c in ds.column_names)
        texts = [r[col] if isinstance(r[col], str) else r[col][0]["content"] for r in ds]
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


def make_generator(student, teacher, opt, tok, a, out):
    """Returns gen(prompt_texts, seeds) -> [{prompt_ids, gen_ids}].
    seeds: one int per prompt (drives per-request vLLM seeds / hf global seeding)."""
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
            tmp = out / "tmp_policy"
            student.save_pretrained(tmp, safe_serialization=True)
            tok.save_pretrained(tmp)
            # free the GPU for the worker: offload everything, restore after
            student.to("cpu"); teacher.to("cpu"); _move_optimizer(opt, "cpu")
            torch.cuda.empty_cache()
            spec = {"model_dir": str(tmp), "prompt_ids": pid, "seeds": list(seeds),
                    "max_tokens": max_tokens, "gpu_mem_util": a.gpu_mem_util,
                    "out_json": str(out / "worker_out.json")}
            spec_path = out / "worker_spec.json"
            spec_path.write_text(json.dumps(spec))
            log(f"vllm worker: {len(pid)} prompts", out)
            r = subprocess.run([sys.executable, __file__, "--_worker", str(spec_path)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"vllm worker failed:\n{r.stderr[-4000:]}")
            gen_ids = json.loads((out / "worker_out.json").read_text())["gen_ids"]
            shutil.rmtree(tmp, ignore_errors=True)
            (out / "worker_out.json").unlink(missing_ok=True)
            student.to(DEV); teacher.to(DEV); _move_optimizer(opt, DEV)
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
def teacher_prefill(teacher, rollouts, layers, out, max_tok, last_k=0, topk_targets=False):
    """Frozen-teacher pass over each rollout. Stores, for the SUPERVISED region only
    (last `last_k` gen tokens if set, else all): hidden states at `layers` (bf16 cpu;
    bf16 not fp16 because large-model activation outliers overflow fp16), plus, when
    topk_targets, the teacher's renormalized top-16 next-token distribution (opd_topk).
    Returns list of {"h": {layer: (sup,d_t)}, "off": int, "tk_ids", "tk_p"}."""
    entries = []
    for i, r in enumerate(rollouts):
        glen = min(len(r["gen_ids"]), max_tok)
        ids = torch.tensor([r["prompt_ids"] + r["gen_ids"][:glen]], device=DEV)
        gstart = len(r["prompt_ids"])
        off = max(0, glen - last_k) if last_k else 0
        o = teacher(input_ids=ids, output_hidden_states=True, use_cache=False)
        e = {"off": off,
             "h": {li: o.hidden_states[li][0, gstart + off:gstart + glen].bfloat16().cpu()
                   for li in layers}}
        if topk_targets:
            lg = o.logits[0, gstart + off - 1:gstart + glen - 1].float()
            p = torch.softmax(lg, -1)
            tp, ti = p.topk(16, dim=-1)
            e["tk_p"] = (tp / tp.sum(-1, keepdim=True)).half().cpu()
            e["tk_ids"] = ti.int().cpu()
        entries.append(e)
        del o
        if DEV == "mps":
            torch.mps.empty_cache()
        if (i + 1) % 25 == 0:
            log(f"  teacher prefill {i+1}/{len(rollouts)}", out)
    return entries


# ---------------------------------------------------------------- loss arms

@torch.no_grad()
def collect_pool_states(student, rolls, tstates, lay_s, lmap, max_tok, out, cap=16384):
    """Frozen-student prefill over the initial rollout pool, aligned token-for-token with
    the already-computed teacher states. Returns (S, T): dicts keyed by student layer,
    S[ls] (N,d_s) / T[ls] (N,d_t) float32 cpu, N ≤ cap. Used only for Stage-1 bridge
    fitting / projector init — both backbones frozen (paper Eq. 12-13 procedure)."""
    S = {ls: [] for ls in lay_s}
    T = {ls: [] for ls in lay_s}
    n = 0
    student.eval()
    for r, ts in zip(rolls, tstates):
        glen = min(len(r["gen_ids"]), max_tok)
        off = ts["off"]
        ids = torch.tensor([r["prompt_ids"] + r["gen_ids"][:glen]], device=DEV)
        gstart = len(r["prompt_ids"])
        o = student(input_ids=ids, output_hidden_states=True)
        for ls in lay_s:
            S[ls].append(o.hidden_states[ls][0, gstart + off:gstart + glen].float().cpu())
            T[ls].append(ts["h"][lmap[ls]][:glen - off].float())
        del o
        if DEV == "mps":
            torch.mps.empty_cache()
        n += glen - off
        if n >= cap:
            break
    student.train()
    log(f"stage-1 pool: {n} tokens over {len(lay_s)} layers", out)
    return ({ls: torch.cat(v)[:cap] for ls, v in S.items()},
            {ls: torch.cat(v)[:cap] for ls, v in T.items()})


def fit_bridge(S, T, rank, out):
    """OPRD-Bridge Stage 1 (both backbones frozen, then everything frozen):
    P_T = top-r principal directions of the teacher covariance (paper Eq. 12);
    P_S = linear map minimizing ||P_S h_S - P_T (h_T - mu)||^2 (paper Eq. 13) —
    solved in closed form by least squares (exact optimum of the paper's convex
    Stage-1 objective, in place of its 20-epoch AdamW loop)."""
    bridge = {}
    for ls in S:
        Ht, Hs = T[ls].double(), S[ls].double()
        mu = Ht.mean(0)
        C = (Ht - mu).T @ (Ht - mu) / max(Ht.shape[0] - 1, 1)
        evals, evecs = torch.linalg.eigh(C)
        P_T = evecs[:, -rank:].flip(-1).T.contiguous()          # (r, d_t)
        Y = (Ht - mu) @ P_T.T                                   # (N, r)
        P_S = torch.linalg.lstsq(Hs, Y).solution.T.contiguous()  # (r, d_s)
        resid = ((Hs @ P_S.T - Y) ** 2).mean().sqrt()
        bridge[ls] = (mu.float().to(DEV), P_T.float().to(DEV), P_S.float().to(DEV))
        log(f"bridge layer {ls}: r={rank}, stage-1 rms residual {resid:.4f}", out)
    return bridge


def init_linear_map(wmap, S, T, out):
    """Closed-form least-squares warm start for the TinyBERT-style map (removes the
    random-init failure mode; joint training afterwards is the arm's defining choice)."""
    for ls in S:
        W = torch.linalg.lstsq(S[ls].double(), T[ls].double()).solution.T  # (d_t, d_s)
        wmap[str(ls)].weight.data.copy_(W.float().to(DEV))
    log("linear_map warm-started via least squares (both backbones frozen)", out)


def cosine_sim(h):
    hn = F.normalize(h.float(), dim=-1)
    return hn @ hn.T


def loss_relational(arm, hs, ht, tau, topk):
    """hs (T,d_s), ht (T,d_t) — window = this rollout's gen region (cap REL_WINDOW)."""
    T = min(hs.shape[0], REL_WINDOW)
    hs, ht = hs[:T], ht[:T]
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


def compute_loss(arm, s_states, t_states, layers, bridge, wmap, tau, topk):
    per_layer = []
    for li in layers:
        hs = s_states[li]
        ht = t_states[li].to(hs.device).float()
        if arm == "oprd_bridge":
            mu, P_T, P_S = bridge[li]
            a = F.normalize(hs @ P_S.T, dim=-1)          # paper Eq. 14: L2-normalized
            b = F.normalize((ht - mu) @ P_T.T, dim=-1)   # projections, frozen bridge
            per_layer.append((a - b).norm(dim=-1).mean())
        elif arm == "linear_map_mse":
            per_layer.append(F.mse_loss(wmap[str(li)](hs), ht))
        elif arm == "cka_gram":
            T = min(hs.shape[0], REL_WINDOW)
            per_layer.append(1 - linear_cka(hs[:T], ht[:T]))
        else:
            per_layer.append(loss_relational(arm, hs, ht, tau, topk))
    return torch.stack(per_layer).mean()


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


# ---------------------------------------------------------------- main

def main():
    a = parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    log(f"harness: loss={a.loss} device={DEV}", out)

    tok_t = AutoTokenizer.from_pretrained(a.teacher)
    tok = AutoTokenizer.from_pretrained(a.student)
    if tok_t.get_vocab() != tok.get_vocab():
        raise SystemExit("cross-tokenizer pair detected: token-aligned losses are undefined "
                         "here — this harness supports shared-vocab pairs only (see docstring).")

    teacher = AutoModelForCausalLM.from_pretrained(a.teacher, dtype=torch.bfloat16).to(DEV).eval()
    student = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.float32)
    student.gradient_checkpointing_enable()
    student.config.use_cache = False
    student.to(DEV).train()

    fracs = [float(f) for f in a.supervised_layer_fracs.split(",")]
    nl_t, nl_s = teacher.config.num_hidden_layers, student.config.num_hidden_layers
    lmap = {}
    for f in fracs:  # student layer -> teacher layer; first frac wins on rounding collisions
        ls_ = max(1, round(f * nl_s))
        lmap.setdefault(ls_, max(1, round(f * nl_t)))
    lay_s = sorted(lmap)
    lay_t = sorted(set(lmap.values()))
    log(f"supervised layers student={lay_s} teacher={lay_t}", out)

    wmap = None
    params = list(student.parameters())
    if a.loss == "linear_map_mse":
        wmap = torch.nn.ModuleDict({
            str(ls): torch.nn.Linear(student.config.hidden_size,
                                     teacher.config.hidden_size, bias=False)
            for ls in lay_s}).to(DEV).float()
        params += list(wmap.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * s / max(a.steps, 1))))

    gen = make_generator(student, teacher, opt, tok, a, out)
    res = {}

    if a.eval_only:
        ckpt = out / "ckpt_final.pt"
        st = torch.load(ckpt, map_location="cpu")
        student.load_state_dict({k: v.to(torch.float32) for k, v in st.items()})
        if (out / "results.json").exists():  # merge, don't clobber the training results
            res = json.loads((out / "results.json").read_text())
        log(f"eval-only: loaded {ckpt}", out)
    else:
        prompts = load_prompt_texts(a.prompts, a.n_prompts, a.seed)

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
                                      topk_targets=(a.loss == "opd_topk"))
            return rolls, tstates

        rolls, tstates = refresh(0)
        if not rolls:
            raise SystemExit(
                "initial rollout pool is empty (all generations < 8 tokens, or none finished "
                "under --require-finished — raise --rollout-tokens or drop the flag)")
        bridge = None
        if a.loss in ("oprd_bridge", "linear_map_mse"):
            # Stage 1 (paper procedure): both backbones frozen, fit/init the projector(s)
            S_pool, T_pool = collect_pool_states(student, rolls, tstates, lay_s, lmap,
                                                 rt, out)
            if a.loss == "oprd_bridge":
                bridge = fit_bridge(S_pool, T_pool, a.bridge_rank, out)  # then frozen forever
            else:
                init_linear_map(wmap, S_pool, T_pool, out)  # warm start; trains jointly after
            del S_pool, T_pool

        losses, ri = [], 0
        for step in range(a.steps):
            if a.rollout_refresh and step and step % a.rollout_refresh == 0:
                new_rolls, new_tstates = refresh(step)
                if new_rolls is None:
                    log(f"refresh @step {step}: pool empty, keeping previous rollouts", out)
                else:
                    rolls, tstates, ri = new_rolls, new_tstates, 0
            opt.zero_grad(set_to_none=True)
            toks_done, batch_losses = 0, []
            while toks_done < a.batch_tokens:
                r, ts = rolls[ri % len(rolls)], tstates[ri % len(rolls)]
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
                    o = student(input_ids=ids, output_hidden_states=True)
                    s_states = {ls: o.hidden_states[ls][0, gstart + off:gstart + glen].float()
                                for ls in lay_s}
                    t_states = {ls: ts["h"][lmap[ls]][:sup] for ls in lay_s}
                    loss = compute_loss(a.loss, s_states, t_states, lay_s, bridge, wmap,
                                        a.tau, a.topk)
                # weight by tokens the loss actually saw, clipped to the remaining budget
                eff = min(sup, REL_WINDOW) if a.loss in RELATIONAL else sup
                w = min(eff, a.batch_tokens - toks_done) / a.batch_tokens
                (loss * w).backward()
                batch_losses.append(float(loss.item()))
                toks_done += eff
                del o
                if DEV == "mps":
                    torch.mps.empty_cache()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            losses.append(float(np.mean(batch_losses)))
            if (step + 1) % 10 == 0 or step == 0:
                log(f"step {step+1}/{a.steps} loss={losses[-1]:.5f} "
                    f"lr={sched.get_last_lr()[0]:.2e}", out)
            if (step + 1) % 100 == 0:
                torch.save({k: v.to(torch.bfloat16).cpu()
                            for k, v in student.state_dict().items()}, out / "ckpt_last.pt")

        torch.save({k: v.to(torch.bfloat16).cpu() for k, v in student.state_dict().items()},
                   out / "ckpt_final.pt")
        if wmap is not None:
            torch.save(wmap.state_dict(), out / "wmap_final.pt")
        res["losses"] = losses

        diag_names = [d for d in a.diagnostics.split(",") if d]
        if a.track_stable_rank or diag_names:
            with torch.no_grad():
                sp, tp = [], []
                for r, ts in zip(rolls[:20], tstates[:20]):
                    glen = min(len(r["gen_ids"]), rt)
                    off = ts["off"]
                    ids = torch.tensor([r["prompt_ids"] + r["gen_ids"][:glen]], device=DEV)
                    o = student(input_ids=ids, output_hidden_states=True)
                    sp.append(o.hidden_states[lay_s[-1]][0, len(r["prompt_ids"]) + off:
                                                         len(r["prompt_ids"]) + glen].float().cpu())
                    tp.append(ts["h"][lay_t[-1]][:glen - off].float())
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
