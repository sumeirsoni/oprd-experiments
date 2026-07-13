# Tier 0 results — kill-switch measurements (2026-07-11 overnight, M3 Max 36GB)

**Summary for partner (read this first):** We ran the four no-training measurements that decide which training arms deserve cluster time. Bottom line: **don't spend any allocation on identity-only relational arms** — on both model pairs, the token-neighborhood identity signal lives almost entirely in the part of the representation the output head already reads (row-space mKNN ≥ 99% of full), so plain on-policy distillation already sees it. OPRD's core variance argument **does** hold in practice (its gradient noise is ~2.2× lower than OPD's on identical rollouts), so OPRD-MSE stays the backbone. Whether geometric alignment actually *causes* correctness we could not resolve at this scale (and at rollout level the correlation trends weakly negative), so every geometric arm we hand you includes a functional-bridge variant as insurance. The power analysis says: design experiments to detect **≥2-point** effects — at realistic pass rates and seed correlation that's ~50–80 seeds/arm, while 1-point effects need 200–700 and are not worth the allocation.

## Verdict → Tier 1–2 consequence

| # | Measurement | Verdict | Consequence for Tier 1–2 |
|---|---|---|---|
| M1 | W_head row/null-space neighbor localization | **NO-GO** (both pairs): row-space carries ≥99% of mKNN signal at every swept layer | Identity-only arms **demoted to control**. Note: null-space carries near-equal signal at mid layers (info is diffuse, not head-hidden); mid band (25–50% depth) is where relational signal is richest → supervise there. |
| M2 | Which relational property survives calibration | Identity ranks first (both pairs), local-distance ≈ global second | LN-Dist stays a live arm; identity-only kept **only as control** (M1 overrides). All three families beat block-permutation null decisively. |
| M3 | Functional vs geometric alignment | **AMBIGUOUS** (both pairs): neither tracks correctness at this N; rollout-level trend weakly negative for geometry | Add **functional-bridge variant** to every geometric arm. Causal-inertness threat stands unresolved — do not claim geometry→function without Tier 1 evidence. |
| M4 | Gradient variance + power | **GO for OPRD-MSE**: relative variance 0.389 vs OPD 0.857; Theorem 1 holds as used | Seed budget from power table below: target 2-pt effects (~53–82 seeds/arm at pass 0.2–0.5, ρ=0.2). 1-pt effects are out of budget. |

## Deviations and skips (complete list)

1. First overnight attempt (7/11 ~02:05) kernel-panicked the host: mlx generation and MPS extraction ran concurrently and starved the kernel. Rerun executed strictly serially; measurements unaffected (all data regenerated or resumed from checkpoints).
2. cross_width token pool = **88,431 < the 100K floor** (Qwen3-1.7B-Base writes shorter rollouts at 2000-token cap). Recorded, not padded.
3. M3 same_arch grading matched only **2/100 rollouts** (boxed-answer extraction failed against dataset ground truth on R1-distill outputs) → same_arch rollout-level correctness correlations are unpowered/absent. cross_width graded 75/100.
4. M4 sequence cap halved 2000→1000 tokens per the OOM ladder (MPS backward peak exceeded the 24GB allocator cap); MPS watermark raised to 0.85 for M4 only. Gradient-variance comparison unaffected (both objectives measured on identical truncated rollouts).
5. M4 run on same_arch only (per plan — the 1.5B student).
6. Cross-tokenizer stretch pair (Phi-4-mini-reasoning) **not run** — cut per budget guard after the crash restart.
7. All measurements correlational; no activation-patching (named follow-up).

---

## Go/no-go verdict table

| Measurement | Pair | Verdict |
|---|---|---|
| M1 | same_arch | NO-GO for identity-only arms: mKNN(P_row) >= 90% of mKNN(full) at all swept layers - neighbor identity is output-readable, OPD already sees it. |
| M1 | cross_width | NO-GO for identity-only arms: mKNN(P_row) >= 90% of mKNN(full) at all swept layers - neighbor identity is output-readable, OPD already sees it. |
| M2 | same_arch | identity >= local-distance (calibrated z: {'global': 101.30004853585002, 'identity': 555.6248863536979, 'local_distance': 115.33769663171825}) - identity arm stays live. |
| M2 | cross_width | identity >= local-distance (calibrated z: {'global': 57.34122596651233, 'identity': 105.44491940990476, 'local_distance': 56.976623694301125}) - identity arm stays live. |
| M3 | same_arch | AMBIGUOUS: neither alignment tracks correctness at this N (low power or ceiling). |
| M3 | cross_width | AMBIGUOUS: neither alignment tracks correctness at this N (low power or ceiling). |
| M4 | same_arch | OPRD gradient variance materially lower than OPD (0.389 vs 0.857 relative variance) - Theorem 1's variance claim holds as used. |
| M4 | cross_width | SKIPPED (no result file) |

## M1 (same_arch)

- pool=198651 tokens, Ns=[1000, 10000, 50000, 100000], r90={'justrl': 1140, 'r1d15': 1140}

| layer frac | proj | N | mKNN k=5 | k=10 | k=30 |
|---|---|---|---|---|---|
| 0.25 | full | 1000 | 0.954 | 0.956 | 0.954 |
| 0.25 | full | 10000 | 0.937 | 0.944 | 0.948 |
| 0.25 | full | 50000 | 0.931 | 0.936 | 0.942 |
| 0.25 | full | 100000 | 0.924 | 0.930 | 0.937 |
| 0.25 | pca_r90 | 100000 | 0.925 | 0.931 | 0.939 |
| 0.25 | row_r90 | 1000 | 0.952 | 0.950 | 0.949 |
| 0.25 | row_r90 | 10000 | 0.938 | 0.942 | 0.943 |
| 0.25 | row_r90 | 50000 | 0.929 | 0.934 | 0.939 |
| 0.25 | row_r90 | 100000 | 0.922 | 0.927 | 0.935 |
| 0.25 | null_r90 | 1000 | 0.951 | 0.954 | 0.959 |
| 0.25 | null_r90 | 10000 | 0.936 | 0.940 | 0.946 |
| 0.25 | null_r90 | 50000 | 0.928 | 0.932 | 0.938 |
| 0.25 | null_r90 | 100000 | 0.921 | 0.926 | 0.934 |
| 0.25 | row_64 | 100000 | 0.879 | 0.886 | 0.897 |
| 0.25 | row_256 | 100000 | 0.911 | 0.918 | 0.926 |
| 0.5 | full | 1000 | 0.921 | 0.929 | 0.926 |
| 0.5 | full | 10000 | 0.902 | 0.908 | 0.910 |
| 0.5 | full | 50000 | 0.891 | 0.894 | 0.900 |
| 0.5 | full | 100000 | 0.882 | 0.886 | 0.893 |
| 0.5 | pca_r90 | 100000 | 0.885 | 0.890 | 0.899 |
| 0.5 | row_r90 | 1000 | 0.917 | 0.924 | 0.918 |
| 0.5 | row_r90 | 10000 | 0.897 | 0.902 | 0.903 |
| 0.5 | row_r90 | 50000 | 0.885 | 0.889 | 0.893 |
| 0.5 | row_r90 | 100000 | 0.876 | 0.881 | 0.886 |
| 0.5 | null_r90 | 1000 | 0.920 | 0.929 | 0.925 |
| 0.5 | null_r90 | 10000 | 0.898 | 0.903 | 0.907 |
| 0.5 | null_r90 | 50000 | 0.885 | 0.889 | 0.894 |
| 0.5 | null_r90 | 100000 | 0.877 | 0.881 | 0.887 |
| 0.5 | row_64 | 100000 | 0.804 | 0.805 | 0.804 |
| 0.5 | row_256 | 100000 | 0.860 | 0.863 | 0.866 |
| 0.75 | full | 1000 | 0.876 | 0.881 | 0.890 |
| 0.75 | full | 10000 | 0.841 | 0.843 | 0.841 |
| 0.75 | full | 50000 | 0.818 | 0.818 | 0.817 |
| 0.75 | full | 100000 | 0.806 | 0.807 | 0.807 |
| 0.75 | pca_r90 | 100000 | 0.810 | 0.812 | 0.815 |
| 0.75 | row_r90 | 1000 | 0.871 | 0.875 | 0.878 |
| 0.75 | row_r90 | 10000 | 0.838 | 0.838 | 0.832 |
| 0.75 | row_r90 | 50000 | 0.815 | 0.814 | 0.809 |
| 0.75 | row_r90 | 100000 | 0.804 | 0.802 | 0.799 |
| 0.75 | null_r90 | 1000 | 0.862 | 0.875 | 0.885 |
| 0.75 | null_r90 | 10000 | 0.828 | 0.832 | 0.833 |
| 0.75 | null_r90 | 50000 | 0.805 | 0.807 | 0.806 |
| 0.75 | null_r90 | 100000 | 0.791 | 0.794 | 0.794 |
| 0.75 | row_64 | 100000 | 0.682 | 0.673 | 0.663 |
| 0.75 | row_256 | 100000 | 0.774 | 0.769 | 0.759 |
| 1.0 | full | 1000 | 0.762 | 0.768 | 0.785 |
| 1.0 | full | 10000 | 0.725 | 0.724 | 0.712 |
| 1.0 | full | 50000 | 0.689 | 0.680 | 0.661 |
| 1.0 | full | 100000 | 0.672 | 0.663 | 0.644 |
| 1.0 | pca_r90 | 100000 | 0.681 | 0.675 | 0.662 |
| 1.0 | row_r90 | 1000 | 0.804 | 0.806 | 0.792 |
| 1.0 | row_r90 | 10000 | 0.762 | 0.762 | 0.748 |
| 1.0 | row_r90 | 50000 | 0.735 | 0.730 | 0.715 |
| 1.0 | row_r90 | 100000 | 0.721 | 0.715 | 0.701 |
| 1.0 | null_r90 | 1000 | 0.723 | 0.745 | 0.780 |
| 1.0 | null_r90 | 10000 | 0.679 | 0.680 | 0.678 |
| 1.0 | null_r90 | 50000 | 0.628 | 0.616 | 0.604 |
| 1.0 | null_r90 | 100000 | 0.607 | 0.594 | 0.580 |
| 1.0 | row_64 | 100000 | 0.485 | 0.461 | 0.431 |
| 1.0 | row_256 | 100000 | 0.655 | 0.639 | 0.611 |

row_ratio_min=0.9935382980235792, null_share_max=0.9962436296023063, fragility_slope=-0.020879963300243415

**Verdict:** NO-GO for identity-only arms: mKNN(P_row) >= 90% of mKNN(full) at all swept layers - neighbor identity is output-readable, OPD already sees it.

## M1 (cross_width)

- pool=88431 tokens, Ns=[1000, 10000, 50000, 88431], r90={'qwen3_4b': 2085, 'qwen3_17b': 1675}

| layer frac | proj | N | mKNN k=5 | k=10 | k=30 |
|---|---|---|---|---|---|
| 0.25 | full | 1000 | 0.793 | 0.782 | 0.718 |
| 0.25 | full | 10000 | 0.768 | 0.749 | 0.737 |
| 0.25 | full | 50000 | 0.746 | 0.737 | 0.722 |
| 0.25 | full | 88431 | 0.731 | 0.729 | 0.723 |
| 0.25 | pca_r90 | 88431 | 0.740 | 0.741 | 0.741 |
| 0.25 | row_r90 | 1000 | 0.789 | 0.781 | 0.707 |
| 0.25 | row_r90 | 10000 | 0.765 | 0.747 | 0.732 |
| 0.25 | row_r90 | 50000 | 0.743 | 0.734 | 0.717 |
| 0.25 | row_r90 | 88431 | 0.729 | 0.727 | 0.719 |
| 0.25 | null_r90 | 1000 | 0.761 | 0.736 | 0.674 |
| 0.25 | null_r90 | 10000 | 0.741 | 0.714 | 0.692 |
| 0.25 | null_r90 | 50000 | 0.718 | 0.705 | 0.678 |
| 0.25 | null_r90 | 88431 | 0.705 | 0.698 | 0.680 |
| 0.25 | row_64 | 88431 | 0.589 | 0.571 | 0.526 |
| 0.25 | row_256 | 88431 | 0.685 | 0.677 | 0.652 |
| 0.5 | full | 1000 | 0.685 | 0.686 | 0.678 |
| 0.5 | full | 10000 | 0.676 | 0.662 | 0.644 |
| 0.5 | full | 50000 | 0.658 | 0.650 | 0.623 |
| 0.5 | full | 88431 | 0.644 | 0.640 | 0.624 |
| 0.5 | pca_r90 | 88431 | 0.652 | 0.651 | 0.639 |
| 0.5 | row_r90 | 1000 | 0.678 | 0.675 | 0.665 |
| 0.5 | row_r90 | 10000 | 0.670 | 0.655 | 0.633 |
| 0.5 | row_r90 | 50000 | 0.653 | 0.643 | 0.613 |
| 0.5 | row_r90 | 88431 | 0.639 | 0.634 | 0.613 |
| 0.5 | null_r90 | 1000 | 0.636 | 0.632 | 0.619 |
| 0.5 | null_r90 | 10000 | 0.645 | 0.621 | 0.591 |
| 0.5 | null_r90 | 50000 | 0.624 | 0.608 | 0.568 |
| 0.5 | null_r90 | 88431 | 0.612 | 0.599 | 0.570 |
| 0.5 | row_64 | 88431 | 0.468 | 0.433 | 0.367 |
| 0.5 | row_256 | 88431 | 0.587 | 0.569 | 0.525 |
| 0.75 | full | 1000 | 0.677 | 0.688 | 0.677 |
| 0.75 | full | 10000 | 0.649 | 0.639 | 0.630 |
| 0.75 | full | 50000 | 0.628 | 0.621 | 0.601 |
| 0.75 | full | 88431 | 0.614 | 0.609 | 0.597 |
| 0.75 | pca_r90 | 88431 | 0.624 | 0.621 | 0.615 |
| 0.75 | row_r90 | 1000 | 0.673 | 0.680 | 0.663 |
| 0.75 | row_r90 | 10000 | 0.647 | 0.635 | 0.622 |
| 0.75 | row_r90 | 50000 | 0.626 | 0.617 | 0.593 |
| 0.75 | row_r90 | 88431 | 0.614 | 0.606 | 0.590 |
| 0.75 | null_r90 | 1000 | 0.647 | 0.644 | 0.640 |
| 0.75 | null_r90 | 10000 | 0.614 | 0.600 | 0.590 |
| 0.75 | null_r90 | 50000 | 0.584 | 0.572 | 0.551 |
| 0.75 | null_r90 | 88431 | 0.572 | 0.562 | 0.547 |
| 0.75 | row_64 | 88431 | 0.427 | 0.396 | 0.337 |
| 0.75 | row_256 | 88431 | 0.555 | 0.535 | 0.492 |
| 1.0 | full | 1000 | 0.680 | 0.691 | 0.665 |
| 1.0 | full | 10000 | 0.597 | 0.600 | 0.604 |
| 1.0 | full | 50000 | 0.564 | 0.555 | 0.527 |
| 1.0 | full | 88431 | 0.547 | 0.539 | 0.518 |
| 1.0 | pca_r90 | 88431 | 0.550 | 0.543 | 0.525 |
| 1.0 | row_r90 | 1000 | 0.685 | 0.700 | 0.657 |
| 1.0 | row_r90 | 10000 | 0.618 | 0.620 | 0.622 |
| 1.0 | row_r90 | 50000 | 0.590 | 0.583 | 0.553 |
| 1.0 | row_r90 | 88431 | 0.574 | 0.567 | 0.545 |
| 1.0 | null_r90 | 1000 | 0.611 | 0.619 | 0.626 |
| 1.0 | null_r90 | 10000 | 0.518 | 0.524 | 0.533 |
| 1.0 | null_r90 | 50000 | 0.475 | 0.466 | 0.443 |
| 1.0 | null_r90 | 88431 | 0.458 | 0.449 | 0.432 |
| 1.0 | row_64 | 88431 | 0.366 | 0.347 | 0.310 |
| 1.0 | row_256 | 88431 | 0.492 | 0.477 | 0.440 |

row_ratio_min=0.9895342040997949, null_share_max=0.9565155361471099, fragility_slope=-0.022371084488101236

**Verdict:** NO-GO for identity-only arms: mKNN(P_row) >= 90% of mKNN(full) at all swept layers - neighbor identity is output-readable, OPD already sees it.

## M2 (same_arch)


**layer frac 0.5** (raw / null mean / calibrated / sig):

| metric | raw | null | calibrated | significant |
|---|---|---|---|---|
| global_r | 0.9824 | 0.0237 | 0.9586 | True |
| identity_mknn | 0.8878 | 0.0075 | 0.8802 | True |
| local_spearman | 0.8819 | 0.0375 | 0.8444 | True |
| local_nl2 | 0.0330 | 0.1794 | -0.1464 | True |

**layer frac 1.0** (raw / null mean / calibrated / sig):

| metric | raw | null | calibrated | significant |
|---|---|---|---|---|
| global_r | 0.8904 | 0.0150 | 0.8754 | True |
| identity_mknn | 0.6201 | 0.0089 | 0.6113 | True |
| local_spearman | 0.5835 | 0.0366 | 0.5469 | True |
| local_nl2 | 0.1065 | 0.2915 | -0.1849 | True |

calibrated z (mid layer): {'global': 101.30004853585002, 'identity': 555.6248863536979, 'local_distance': 115.33769663171825}
note: block permutation (whole rollouts) as primary null; token-level exchangeability violated per ARH caveat

**Verdict:** identity >= local-distance (calibrated z: {'global': 101.30004853585002, 'identity': 555.6248863536979, 'local_distance': 115.33769663171825}) - identity arm stays live.

## M2 (cross_width)


**layer frac 0.5** (raw / null mean / calibrated / sig):

| metric | raw | null | calibrated | significant |
|---|---|---|---|---|
| global_r | 0.7311 | 0.0239 | 0.7072 | True |
| identity_mknn | 0.5778 | 0.0183 | 0.5596 | True |
| local_spearman | 0.5790 | 0.0686 | 0.5104 | True |
| local_nl2 | 0.1459 | 0.2152 | -0.0693 | True |

**layer frac 1.0** (raw / null mean / calibrated / sig):

| metric | raw | null | calibrated | significant |
|---|---|---|---|---|
| global_r | 0.6707 | 0.0122 | 0.6585 | True |
| identity_mknn | 0.4828 | 0.0109 | 0.4719 | True |
| local_spearman | 0.4366 | 0.0430 | 0.3935 | True |
| local_nl2 | 0.2754 | 0.3991 | -0.1237 | True |

calibrated z (mid layer): {'global': 57.34122596651233, 'identity': 105.44491940990476, 'local_distance': 56.976623694301125}
note: block permutation (whole rollouts) as primary null; token-level exchangeability violated per ARH caveat

**Verdict:** identity >= local-distance (calibrated z: {'global': 57.34122596651233, 'identity': 105.44491940990476, 'local_distance': 56.976623694301125}) - identity arm stays live.

## M3 (same_arch)

- n_tokens=198651, grading: {'method': 'dataset ground_truth vs \\boxed{} extraction', 'graded_rollouts': 2, 'total': 100}
- token correlations: {'geo0.5_vs_negKL': {'spearman': 0.03806073099844476, 'p': 1.3736052195374078e-64, 'n': 198651}, 'geo0.5_vs_top1': {'spearman': -0.0007087713620936273, 'p': 0.7520788341760901, 'n': 198651}, 'geo0.5_vs_pos': {'spearman': 0.017600967908887898, 'p': 4.3171576702144895e-15, 'n': 198651}, 'geo1.0_vs_negKL': {'spearman': -0.10964194487808687, 'p': 0.0, 'n': 198651}, 'geo1.0_vs_top1': {'spearman': -0.057306755034333655, 'p': 3.974144480730439e-144, 'n': 198651}, 'geo1.0_vs_pos': {'spearman': 0.0545904483404389, 'p': 5.923082419133369e-131, 'n': 198651}, 'KL_vs_pos': {'spearman': -0.008890771886160151, 'p': 7.410829605585248e-05, 'n': 198651}}
- rollout-level vs correctness: {'geo0.5': None, 'geo1.0': None, 'negKL': None, 'top1': None}
- region split: {'geo1.0': {'pre_decision_mean': 0.9339460730552673, 'answer_region_mean': 0.945128321647644}, 'KL': {'pre_decision_mean': 0.30494382977485657, 'answer_region_mean': 0.308298796415329}}
- logit-lens KL by student layer: {'7': 10.389582633972168, '14': 9.409713745117188, '21': 5.095975875854492, '28': 0.9811660647392273}

**Verdict:** AMBIGUOUS: neither alignment tracks correctness at this N (low power or ceiling).

## M3 (cross_width)

- n_tokens=88431, grading: {'method': 'dataset ground_truth vs \\boxed{} extraction', 'graded_rollouts': 75, 'total': 100}
- token correlations: {'geo0.5_vs_negKL': {'spearman': -0.156308433686491, 'p': 0.0, 'n': 88431}, 'geo0.5_vs_top1': {'spearman': -0.06254712486014341, 'p': 2.3038555256694508e-77, 'n': 88431}, 'geo0.5_vs_pos': {'spearman': -0.03896257905408635, 'p': 4.5949338618775525e-31, 'n': 88431}, 'geo1.0_vs_negKL': {'spearman': -0.16210935593437234, 'p': 0.0, 'n': 88431}, 'geo1.0_vs_top1': {'spearman': -0.08512199087993054, 'p': 7.205096168291648e-142, 'n': 88431}, 'geo1.0_vs_pos': {'spearman': -0.10043863463589785, 'p': 5.410110076449708e-197, 'n': 88431}, 'KL_vs_pos': {'spearman': -0.13577391535783115, 'p': 0.0, 'n': 88431}}
- rollout-level vs correctness: {'geo0.5': {'pointbiserial': -0.15839456279370084, 'p': 0.17469121248713274, 'n': 75}, 'geo1.0': {'pointbiserial': -0.12432497273278556, 'p': 0.2879036996602928, 'n': 75}, 'negKL': {'pointbiserial': 0.11437492794438583, 'p': 0.3285243785247554, 'n': 75}, 'top1': {'pointbiserial': 0.0578871600348261, 'p': 0.6217925680563327, 'n': 75}}
- region split: {'geo1.0': {'pre_decision_mean': 0.8601375222206116, 'answer_region_mean': 0.833162248134613}, 'KL': {'pre_decision_mean': 0.3905845582485199, 'answer_region_mean': 0.25136664509773254}}
- logit-lens KL by student layer: {'7': 13.612602233886719, '14': 10.61829948425293, '21': 3.970433235168457, '28': 3.3002023696899414}

**Verdict:** AMBIGUOUS: neither alignment tracks correctness at this N (low power or ceiling).

## M4 (same_arch)

- block layer 14, 46.8M params, 30 steps
- OPRD-MSE: {'mean_pairwise_cosine': 0.6102453470230103, 'relative_variance': 0.3893672227859497, 'mean_grad_norm': 2.9596855640411377}
- OPD-CE:   {'mean_pairwise_cosine': 0.11374960839748383, 'relative_variance': 0.8574491739273071, 'mean_grad_norm': 0.3092780113220215}

**Power analysis** (two-sample, 80% power, alpha=.05 two-sided, Avg@16, n_problems=30):

| pass rate | rho | effect (pts) | seeds/arm |
|---|---|---|---|
| 0.1 | 0.0 | 1 | 30 |
| 0.1 | 0.0 | 2 | 8 |
| 0.1 | 0.2 | 1 | 118 |
| 0.1 | 0.2 | 2 | 30 |
| 0.1 | 0.5 | 1 | 251 |
| 0.1 | 0.5 | 2 | 63 |
| 0.2 | 0.0 | 1 | 53 |
| 0.2 | 0.0 | 2 | 14 |
| 0.2 | 0.2 | 1 | 210 |
| 0.2 | 0.2 | 2 | 53 |
| 0.2 | 0.5 | 1 | 445 |
| 0.2 | 0.5 | 2 | 112 |
| 0.3 | 0.0 | 1 | 69 |
| 0.3 | 0.0 | 2 | 18 |
| 0.3 | 0.2 | 1 | 275 |
| 0.3 | 0.2 | 2 | 69 |
| 0.3 | 0.5 | 1 | 584 |
| 0.3 | 0.5 | 2 | 146 |
| 0.4 | 0.0 | 1 | 79 |
| 0.4 | 0.0 | 2 | 20 |
| 0.4 | 0.2 | 1 | 314 |
| 0.4 | 0.2 | 2 | 79 |
| 0.4 | 0.5 | 1 | 668 |
| 0.4 | 0.5 | 2 | 167 |
| 0.5 | 0.0 | 1 | 82 |
| 0.5 | 0.0 | 2 | 21 |
| 0.5 | 0.2 | 1 | 328 |
| 0.5 | 0.2 | 2 | 82 |
| 0.5 | 0.5 | 1 | 695 |
| 0.5 | 0.5 | 2 | 174 |

**Verdict:** OPRD gradient variance materially lower than OPD (0.389 vs 0.857 relative variance) - Theorem 1's variance claim holds as used.

## M4 (cross_width) - SKIPPED
