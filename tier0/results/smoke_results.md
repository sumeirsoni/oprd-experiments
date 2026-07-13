## Go/no-go verdict table

| Measurement | Pair | Verdict |
|---|---|---|
| M1 | smoke | NO-GO for identity-only arms: mKNN(P_row) >= 90% of mKNN(full) at all swept layers - neighbor identity is output-readable, OPD already sees it. |
| M2 | smoke | local-distance dominates identity (calibrated z: {'global': 16.511899969896618, 'identity': 14.931023827941013, 'local_distance': 16.00488516338294}) - identity-only arm demotes to control; LN-Dist leads. |
| M3 | smoke | AMBIGUOUS: neither alignment tracks correctness at this N (low power or ceiling). |
| M4 | smoke | OPRD gradient variance NOT materially lower than OPD (0.613 vs 0.512) - Theorem 1's practical relevance weakened (affects A1/C framing). |

## M1 (smoke)

- pool=2560 tokens, Ns=[1000, 2560], r90={'smoke_teacher': 710, 'smoke_student': 713}

| layer frac | proj | N | mKNN k=5 | k=10 | k=30 |
|---|---|---|---|---|---|
| 0.25 | full | 1000 | 0.953 | 0.949 | 0.955 |
| 0.25 | full | 2560 | 0.949 | 0.947 | 0.951 |
| 0.25 | pca_r90 | 2560 | 0.947 | 0.949 | 0.956 |
| 0.25 | row_r90 | 1000 | 0.948 | 0.950 | 0.947 |
| 0.25 | row_r90 | 2560 | 0.942 | 0.945 | 0.947 |
| 0.25 | null_r90 | 1000 | 0.934 | 0.927 | 0.928 |
| 0.25 | null_r90 | 2560 | 0.926 | 0.925 | 0.924 |
| 0.25 | row_64 | 2560 | 0.892 | 0.878 | 0.872 |
| 0.25 | row_256 | 2560 | 0.932 | 0.932 | 0.931 |
| 0.5 | full | 1000 | 0.955 | 0.954 | 0.949 |
| 0.5 | full | 2560 | 0.952 | 0.949 | 0.947 |
| 0.5 | pca_r90 | 2560 | 0.954 | 0.950 | 0.951 |
| 0.5 | row_r90 | 1000 | 0.952 | 0.947 | 0.940 |
| 0.5 | row_r90 | 2560 | 0.946 | 0.943 | 0.940 |
| 0.5 | null_r90 | 1000 | 0.920 | 0.925 | 0.925 |
| 0.5 | null_r90 | 2560 | 0.920 | 0.918 | 0.917 |
| 0.5 | row_64 | 2560 | 0.868 | 0.858 | 0.844 |
| 0.5 | row_256 | 2560 | 0.936 | 0.924 | 0.920 |
| 0.75 | full | 1000 | 0.935 | 0.936 | 0.936 |
| 0.75 | full | 2560 | 0.921 | 0.922 | 0.926 |
| 0.75 | pca_r90 | 2560 | 0.927 | 0.929 | 0.931 |
| 0.75 | row_r90 | 1000 | 0.931 | 0.933 | 0.926 |
| 0.75 | row_r90 | 2560 | 0.920 | 0.919 | 0.919 |
| 0.75 | null_r90 | 1000 | 0.913 | 0.915 | 0.921 |
| 0.75 | null_r90 | 2560 | 0.895 | 0.902 | 0.905 |
| 0.75 | row_64 | 2560 | 0.830 | 0.822 | 0.810 |
| 0.75 | row_256 | 2560 | 0.896 | 0.895 | 0.890 |
| 1.0 | full | 1000 | 0.890 | 0.885 | 0.889 |
| 1.0 | full | 2560 | 0.878 | 0.877 | 0.879 |
| 1.0 | pca_r90 | 2560 | 0.882 | 0.883 | 0.886 |
| 1.0 | row_r90 | 1000 | 0.916 | 0.911 | 0.909 |
| 1.0 | row_r90 | 2560 | 0.900 | 0.900 | 0.903 |
| 1.0 | null_r90 | 1000 | 0.845 | 0.845 | 0.861 |
| 1.0 | null_r90 | 2560 | 0.829 | 0.833 | 0.844 |
| 1.0 | row_64 | 2560 | 0.779 | 0.771 | 0.766 |
| 1.0 | row_256 | 2560 | 0.871 | 0.871 | 0.864 |

row_ratio_min=0.994070413835701, null_share_max=0.9778521216227661, fragility_slope=-0.012167322952336699

**Verdict:** NO-GO for identity-only arms: mKNN(P_row) >= 90% of mKNN(full) at all swept layers - neighbor identity is output-readable, OPD already sees it.

## M2 (smoke)


**layer frac 0.5** (raw / null mean / calibrated / sig):

| metric | raw | null | calibrated | significant |
|---|---|---|---|---|
| global_r | 0.9948 | 0.0821 | 0.9127 | True |
| identity_mknn | 0.9489 | 0.0829 | 0.8660 | True |
| local_spearman | 0.9564 | 0.1128 | 0.8436 | True |
| local_nl2 | 0.0278 | 0.3337 | -0.3059 | True |

**layer frac 1.0** (raw / null mean / calibrated / sig):

| metric | raw | null | calibrated | significant |
|---|---|---|---|---|
| global_r | 0.9802 | 0.0606 | 0.9196 | True |
| identity_mknn | 0.8777 | 0.0875 | 0.7903 | True |
| local_spearman | 0.8769 | 0.1228 | 0.7541 | True |
| local_nl2 | 0.0635 | 0.3662 | -0.3026 | True |

calibrated z (mid layer): {'global': 16.511899969896618, 'identity': 14.931023827941013, 'local_distance': 16.00488516338294}
note: block permutation (whole rollouts) as primary null; token-level exchangeability violated per ARH caveat

**Verdict:** local-distance dominates identity (calibrated z: {'global': 16.511899969896618, 'identity': 14.931023827941013, 'local_distance': 16.00488516338294}) - identity-only arm demotes to control; LN-Dist leads.

## M3 (smoke)

- n_tokens=2560, grading: {'method': 'dataset ground_truth vs \\boxed{} extraction', 'graded_rollouts': 0, 'total': 10}
- token correlations: {'geo0.5_vs_negKL': {'spearman': 0.1440604079884559, 'p': 2.413774278117994e-13, 'n': 2560}, 'geo0.5_vs_top1': {'spearman': 0.098857947369433, 'p': 5.395189314083517e-07, 'n': 2560}, 'geo0.5_vs_pos': {'spearman': 0.1662943320922434, 'p': 2.4788624475397492e-17, 'n': 2560}, 'geo1.0_vs_negKL': {'spearman': 0.09971110301315922, 'p': 4.3001839868317627e-07, 'n': 2560}, 'geo1.0_vs_top1': {'spearman': 0.04579632338220522, 'p': 0.020492015249000798, 'n': 2560}, 'geo1.0_vs_pos': {'spearman': 0.10181928079788777, 'p': 2.435446848909169e-07, 'n': 2560}, 'KL_vs_pos': {'spearman': -0.21441863537691802, 'p': 5.247540880990618e-28, 'n': 2560}}
- rollout-level vs correctness: {'geo0.5': None, 'geo1.0': None, 'negKL': None, 'top1': None}
- region split: {'geo1.0': {'pre_decision_mean': nan, 'answer_region_mean': 0.9863338470458984}, 'KL': {'pre_decision_mean': nan, 'answer_region_mean': 0.031083673238754272}}
- logit-lens KL by student layer: {'6': 15.93798542022705, '12': 14.724560737609863, '18': 8.603104591369629, '24': 0.1843043863773346}

**Verdict:** AMBIGUOUS: neither alignment tracks correctness at this N (low power or ceiling).

## M4 (smoke)

- block layer 12, 14.9M params, 3 steps
- OPRD-MSE: {'mean_pairwise_cosine': 0.2642539441585541, 'relative_variance': 0.6134232878684998, 'mean_grad_norm': 75.44612121582031}
- OPD-CE:   {'mean_pairwise_cosine': 0.2301994115114212, 'relative_variance': 0.5115578174591064, 'mean_grad_norm': 2.607215166091919}

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

**Verdict:** OPRD gradient variance NOT materially lower than OPD (0.613 vs 0.512) - Theorem 1's practical relevance weakened (affects A1/C framing).
