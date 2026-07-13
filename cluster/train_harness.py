"""Shared training-harness STUB for Tier 1/2 OPRD experiments (UNTESTED-ON-CLUSTER).

This file defines the CLI contract used by tier1_tau_sweep.sbatch and
tier2_bridge_bakeoff.sbatch, the loss registry with the exact formulation each
arm should implement, and a skeletal on-policy training loop. The heavy pieces
(vLLM rollout server, fused loss kernels) are marked TODO — they must be
implemented and smoke-tested on one GPU before submitting the arrays.

Loss arms (Tier 2 bake-off, adversarial review §5.7):
  pca_mse           OPRD-Bridge paper recipe: PCA-project teacher states to student
                    dim (fit offline on rollout pool), MSE on supervised layers.
  linear_map_mse    TinyBERT-style: learned linear map W: d_s->d_t, MSE in teacher
                    space; W trained jointly, no PCA.
  minilmv2_selfrel  MiniLMv2: projector-free — KL between teacher and student
                    token-token self-relation matrices (Q/K/V relations or hidden
                    cosine-similarity rows, softmax over tokens, temperature 1).
  ln_dist           LN-Dist: local-window top-k distance matching — for each token,
                    teacher's top-k neighbor distances (cosine, within rollout
                    window) matched to student's distances to the SAME token ids,
                    Huber loss. Dimension-free.
  soft_knn_kl       Soft neighbor-identity KL: softmax over similarities to top-k
                    window neighbors, teacher->student KL. tau from --tau.
                    (M2 verdict may demote this arm to control.)
  cka_gram          CKA-Gram: 1 - linear CKA between per-batch Gram matrices of
                    supervised-layer states.
  pkt_tau           Tier 1: same as soft_knn_kl but tau is the swept variable.

CLI (keep stable — sbatch scripts depend on it):
  --loss ARM --tau F --topk K --teacher REPO --student REPO
  --prompts HF_DATASET --n-prompts N --supervised-layer-fracs "0.5,0.75,1.0"
  --steps N --batch-tokens N --lr F --seed N --out DIR
  [--track-stable-rank] [--eval aime24,aime25 --eval-avg-at 16]
  [--diagnostics calibrated_mknn,cka,dist_n]
"""
import argparse
import json
import pathlib


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loss", required=True,
                   choices=["pca_mse", "linear_map_mse", "minilmv2_selfrel",
                            "ln_dist", "soft_knn_kl", "cka_gram", "pkt_tau"])
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--n-prompts", type=int, default=2000)
    p.add_argument("--supervised-layer-fracs", default="0.5,0.75,1.0",
                   help="set from M1 verdict: layers where null-space signal peaks")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch-tokens", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--track-stable-rank", action="store_true")
    p.add_argument("--eval", default="")
    p.add_argument("--eval-avg-at", type=int, default=16)
    p.add_argument("--diagnostics", default="")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=1))

    # TODO(cluster): implement before submitting arrays —
    # 1. Rollout generation: vLLM server for the student (on-policy), sampling
    #    temp 0.6 / top_p 0.95, <=2000 tok, prompts from --prompts (seed --seed).
    # 2. Teacher prefill over student rollouts (frozen, bf16), capture states at
    #    --supervised-layer-fracs; for cross-tokenizer (phi4mini pair) align via
    #    the bridge under test — no token-level alignment exists by construction.
    # 3. Loss registry per module docstring; supervise generated-region tokens only.
    # 4. AdamW, cosine schedule, --lr, --steps optimizer steps of --batch-tokens.
    # 5. Eval: AIME24/25 Avg@--eval-avg-at; diagnostics per --diagnostics.
    #    Seed budget per arm comes from the M4 power table in tier0-results.md.
    raise NotImplementedError(
        "train_harness.py is a stub — implement the TODOs and smoke-test on one "
        "GPU (e.g. --steps 5 --n-prompts 8) before submitting the sbatch arrays.")


if __name__ == "__main__":
    main()
