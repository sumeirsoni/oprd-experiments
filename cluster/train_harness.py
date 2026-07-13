"""Shared training harness for Tier 1/2 OPRD experiments.

Implements the full on-policy loop: student rollout generation (vLLM if available,
transformers fallback), frozen-teacher prefill over student rollouts, seven loss arms,
AdamW training with token-budgeted gradient accumulation, periodic rollout refresh,
AIME Avg@N eval, and diagnostics.

Loss arms (adversarial review §5.5/§5.7; supervised on generated-region tokens only):
  pca_mse           OPRD-Bridge recipe: PCA-project teacher states to student dim
                    (fit on the initial rollout pool), MSE on supervised layers.
  linear_map_mse    TinyBERT-style: learned linear map W: d_s->d_t, MSE in teacher
                    space; W trained jointly with the student.
  minilmv2_selfrel  Projector-free: row-wise KL between teacher and student
                    cosine self-relation matrices (softmax over window tokens).
  ln_dist           Local-window top-k distance matching: teacher's top-k neighbor
                    cosine distances vs student distances to the SAME positions,
                    Huber loss on mean-normalized distance vectors. Dimension-free.
  soft_knn_kl       Soft neighbor-identity KL over teacher's top-k window
                    neighbors, temperature --tau. (Tier-0 M1: control arm only.)
  cka_gram          1 - linear CKA between centered Gram matrices per window.
  pkt_tau           Tier 1: soft_knn_kl with tau as the swept variable.

VERIFICATION STATUS: loss arms + training loop + eval plumbing smoke-tested
end-to-end on Apple MPS with Qwen2.5-0.5B (see cluster/README.md). The vLLM
backend and multi-GPU-scale behavior are UNTESTED — smoke-test one task on the
cluster (--steps 5 --n-prompts 8 --rollout-refresh 0) before submitting arrays.

Cross-tokenizer pairs (Phi-4-mini -> Qwen) are NOT implemented: token-aligned
losses are undefined by construction there; the anchor-fingerprint OT bridge is
a separate stretch arm. The harness exits with a clear error if tokenizers differ.
"""
import argparse
import json
import math
import pathlib
import re
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loss", required=True,
                   choices=["pca_mse", "linear_map_mse", "minilmv2_selfrel",
                            "ln_dist", "soft_knn_kl", "cka_gram", "pkt_tau"])
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
    p.add_argument("--track-stable-rank", action="store_true")
    p.add_argument("--eval", default="", help="comma list of: aime24,aime25")
    p.add_argument("--eval-avg-at", type=int, default=16)
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
    a = str(a).strip().replace(" ", "").replace("\\!", "").rstrip(".0") or "0"
    return a


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

def generate_rollouts(student, tok, prompts, max_tokens, seed, backend, out):
    """Returns list of {prompt_ids, gen_ids}. Samples temp 0.6 / top_p 0.95."""
    use_vllm = backend == "vllm" or (backend == "auto" and DEV == "cuda")
    if use_vllm:
        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            log("vllm not importable, falling back to hf generate", out)
            use_vllm = False
    if use_vllm:
        tmp = out / "tmp_policy"
        student.save_pretrained(tmp, safe_serialization=True)
        tok.save_pretrained(tmp)
        llm = LLM(model=str(tmp), dtype="bfloat16", seed=seed, enforce_eager=False)
        sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=max_tokens, seed=seed)
        pid = [chat_ids(tok, p) for p in prompts]
        outs = llm.generate(prompt_token_ids=pid, sampling_params=sp)
        res = [{"prompt_ids": pid[i], "gen_ids": list(o.outputs[0].token_ids)}
               for i, o in enumerate(outs)]
        del llm
        torch.cuda.empty_cache()
        return res
    # hf fallback: generate from the in-memory student (works on mps/cpu; slow)
    res = []
    student.eval()
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = torch.tensor([chat_ids(tok, p)], device=DEV)
            genconf = dict(do_sample=True, temperature=0.6, top_p=0.95,
                           max_new_tokens=max_tokens, pad_token_id=tok.eos_token_id)
            seq = student.generate(ids, **genconf)
            res.append({"prompt_ids": ids[0].tolist(),
                        "gen_ids": seq[0, ids.shape[1]:].tolist()})
            if (i + 1) % 16 == 0:
                log(f"  hf-gen {i+1}/{len(prompts)}", out)
    student.train()
    return res


@torch.no_grad()
def teacher_prefill(teacher, rollouts, layers, out):
    """Frozen-teacher hidden states at `layers` over gen-region tokens; fp16 cpu."""
    states = []
    for i, r in enumerate(rollouts):
        ids = torch.tensor([r["prompt_ids"] + r["gen_ids"]], device=DEV)
        gstart = len(r["prompt_ids"])
        o = teacher(input_ids=ids, output_hidden_states=True, use_cache=False)
        states.append({li: o.hidden_states[li][0, gstart:].half().cpu() for li in layers})
        del o
        if DEV == "mps":
            torch.mps.empty_cache()
        if (i + 1) % 25 == 0:
            log(f"  teacher prefill {i+1}/{len(rollouts)}", out)
    return states


# ---------------------------------------------------------------- loss arms

def fit_pca(states, layers, d_s, out):
    """Per-layer PCA basis mapping teacher dim -> student dim, fit on ≤50k pooled tokens."""
    bases = {}
    for li in layers:
        pool = torch.cat([s[li] for s in states]).float()
        if pool.shape[0] > 50000:
            pool = pool[torch.randperm(pool.shape[0])[:50000]]
        mu = pool.mean(0)
        _, _, V = torch.linalg.svd(pool - mu, full_matrices=False)
        Vd = V[:d_s]
        if Vd.shape[0] < d_s:  # pool rank < student dim: only possible in tiny smoke runs
            Vd = torch.cat([Vd, torch.zeros(d_s - Vd.shape[0], Vd.shape[1])])
        bases[li] = (mu.to(DEV), Vd.T.to(DEV))  # (d_t,), (d_t, d_s)
    log(f"pca fit on {len(layers)} layers", out)
    return bases


def cosine_sim(h):
    hn = F.normalize(h.float(), dim=-1)
    return hn @ hn.T


def loss_relational(arm, hs, ht, tau, topk):
    """hs (T,d_s), ht (T,d_t) — window = this rollout's gen region (cap 512 tokens)."""
    T = min(hs.shape[0], 512)
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


def compute_loss(arm, s_states, t_states, layers, bases, wmap, tau, topk):
    per_layer = []
    for li in layers:
        hs = s_states[li]
        ht = t_states[li].to(hs.device).float()
        if arm == "pca_mse":
            mu, P = bases[li]
            per_layer.append(F.mse_loss(hs, (ht - mu) @ P))
        elif arm == "linear_map_mse":
            per_layer.append(F.mse_loss(wmap[str(li)](hs), ht))
        elif arm == "cka_gram":
            T = min(hs.shape[0], 512)
            per_layer.append(1 - linear_cka(hs[:T], ht[:T]))
        else:
            per_layer.append(loss_relational(arm, hs, ht, tau, topk))
    return torch.stack(per_layer).mean()


# ---------------------------------------------------------------- eval / diagnostics

@torch.no_grad()
def eval_aime(student, tok, names, avg_at, max_tokens, seed, backend, out):
    from datasets import load_dataset
    results = {}
    student.eval()
    for name in names:
        repo, split = AIME[name]
        ds = load_dataset(repo, split=split)
        pcol = next(c for c in ["Problem", "problem", "question"] if c in ds.column_names)
        acol = next(c for c in ["Answer", "answer", "solution"] if c in ds.column_names)
        correct, total = 0, 0
        for pi, row in enumerate(ds):
            prompts = [row[pcol]] * avg_at
            rolls = generate_rollouts(student, tok, prompts, max_tokens,
                                      seed * 1000 + pi, backend, out)
            gt = norm_ans(row[acol])
            for r in rolls:
                pred = norm_ans(extract_boxed(tok.decode(r["gen_ids"])))
                correct += int(pred is not None and pred == gt)
                total += 1
            log(f"  eval {name} problem {pi+1}/{len(ds)}", out)
        results[name] = {"avg_at": avg_at, "accuracy": correct / max(total, 1), "n": total}
    student.train()
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
        grams = set()
        tot = 0
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
    student = AutoModelForCausalLM.from_pretrained(a.student, dtype=torch.float32).to(DEV)
    student.gradient_checkpointing_enable()
    student.config.use_cache = False
    student.train()

    fracs = [float(f) for f in a.supervised_layer_fracs.split(",")]
    nl_t, nl_s = teacher.config.num_hidden_layers, student.config.num_hidden_layers
    lay_t = sorted({max(1, round(f * nl_t)) for f in fracs})
    lay_s = sorted({max(1, round(f * nl_s)) for f in fracs})
    lmap = dict(zip(lay_s, lay_t))  # student layer -> teacher layer
    log(f"supervised layers student={lay_s} teacher={lay_t}", out)

    prompts = load_prompt_texts(a.prompts, a.n_prompts, a.seed)

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
        opt, lambda s: min(1.0, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * s / a.steps)))

    def refresh(step):
        n_pool = min(len(prompts), max(64, a.batch_tokens // 256))
        sel = [prompts[i] for i in np.random.default_rng(a.seed + step).permutation(
            len(prompts))[:n_pool]]
        log(f"refresh @step {step}: generating {len(sel)} rollouts", out)
        rolls = generate_rollouts(student, tok, sel, a.max_tokens, a.seed + step, a.backend, out)
        rolls = [r for r in rolls if len(r["gen_ids"]) >= 8]
        tstates = teacher_prefill(teacher, rolls, lay_t, out)
        return rolls, tstates

    rolls, tstates = refresh(0)
    bases = None
    if a.loss == "pca_mse":
        bases_t = fit_pca(tstates, lay_t, student.config.hidden_size, out)
        bases = {ls: bases_t[lmap[ls]] for ls in lay_s}  # rekey by student layer

    losses, ri = [], 0
    for step in range(a.steps):
        if a.rollout_refresh and step and step % a.rollout_refresh == 0:
            rolls, tstates = refresh(step)
            ri = 0
        opt.zero_grad(set_to_none=True)
        toks_done, batch_losses = 0, []
        while toks_done < a.batch_tokens:
            r, ts = rolls[ri % len(rolls)], tstates[ri % len(rolls)]
            ri += 1
            glen = min(len(r["gen_ids"]), a.max_tokens)
            ids = torch.tensor([r["prompt_ids"] + r["gen_ids"][:glen]], device=DEV)
            gstart = len(r["prompt_ids"])
            o = student(input_ids=ids, output_hidden_states=True)
            s_states = {ls: o.hidden_states[ls][0, gstart:gstart + glen].float() for ls in lay_s}
            t_states = {lmap[ls]: ts[lmap[ls]][:glen] for ls in lay_s}
            loss = compute_loss(a.loss, s_states,
                                {ls: t_states[lmap[ls]] for ls in lay_s},
                                lay_s, bases, wmap, a.tau, a.topk)
            (loss * glen / a.batch_tokens).backward()
            batch_losses.append(float(loss.item()))
            toks_done += glen
            del o
            if DEV == "mps":
                torch.mps.empty_cache()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(float(np.mean(batch_losses)))
        if (step + 1) % 10 == 0 or step == 0:
            log(f"step {step+1}/{a.steps} loss={losses[-1]:.5f} lr={sched.get_last_lr()[0]:.2e}", out)
        if (step + 1) % 100 == 0:
            torch.save({k: v.to(torch.bfloat16).cpu() for k, v in student.state_dict().items()},
                       out / "ckpt_last.pt")

    torch.save({k: v.to(torch.bfloat16).cpu() for k, v in student.state_dict().items()},
               out / "ckpt_final.pt")
    res = {"losses": losses}

    diag_names = [d for d in a.diagnostics.split(",") if d]
    if diag_names:
        with torch.no_grad():
            sp, tp = [], []
            for r, ts in zip(rolls[:20], tstates[:20]):
                glen = min(len(r["gen_ids"]), a.max_tokens)
                ids = torch.tensor([r["prompt_ids"] + r["gen_ids"][:glen]], device=DEV)
                o = student(input_ids=ids, output_hidden_states=True)
                sp.append(o.hidden_states[lay_s[-1]][0, len(r["prompt_ids"]):].float().cpu())
                tp.append(ts[lay_t[-1]][:glen].float())
                del o
        res["diagnostics"] = diagnostics(diag_names, torch.cat(sp), torch.cat(tp),
                                         [tok.decode(r["gen_ids"]) for r in rolls[:50]])

    eval_names = [e for e in a.eval.split(",") if e]
    if eval_names:
        res["eval"] = eval_aime(student, tok, eval_names, a.eval_avg_at,
                                a.max_tokens, a.seed, a.backend, out)

    (out / "results.json").write_text(json.dumps(res, indent=1))
    log(f"DONE -> {out}/results.json", out)


if __name__ == "__main__":
    main()
