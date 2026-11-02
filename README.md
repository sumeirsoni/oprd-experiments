# On-policy representation distillation experiments

An experimental study of how a language model transfers knowledge to a smaller model. This repository compares output-level distillation with representation-based objectives, investigates where useful information appears in hidden states, and provides training and evaluation pipelines for GPU clusters.

**Status:** Research in progress. The positive control and preliminary ablations are complete. The initial cross-size comparison has a training-protocol confound, and the corrected experiment is not yet a completed benchmark result.

## Selected result: positive control

On a same-architecture 1.5B teacher-student pair, output-level on-policy distillation improved average reasoning accuracy under a fixed evaluation protocol:

| Model | AIME 2024 Avg@16 | AIME 2025 Avg@16 |
| --- | ---: | ---: |
| Untrained student | 28.33% | 22.92% |
| Output-level distillation | 48.89% | 33.68% |
| Mid-layer relational matching | 35.42% | 26.46% |

The trained rows are means over three training seeds; the untrained row is one baseline evaluation. Evaluations sampled 16 responses per problem with a 24,000-token limit. Completion and truncation rates also changed, so these scores alone do not isolate an improvement in reasoning conditional on completing an answer. This is a positive control for the training pipeline, not a demonstration that the proposed representation bridge works across model sizes. See [the full positive-control results](pc-gate-results.md) for individual seeds, rates, and limitations.

The earlier cross-size comparison used truncated training rollouts. Its poor scores cannot establish that the methods fail under a corrected protocol. See [the Tier 2 analysis](tier2-results.md) for the confound and [the archived evidence](oprd-experiment-results-3/README.md) for the later 24k reevaluation and provenance. The immutable bridge passed a single-H200 pilot; a separate implementation and execution contract for an official-code-faithful run are in development. No completed result from that run is claimed here.

## What is implemented

- A PyTorch training harness with student-generated rollouts, teacher supervision, multiple distillation losses, and vLLM generation.
- AIME evaluation with response-completion and truncation diagnostics, plus multi-seed experiment scripts.
- Slurm workflows for cluster setup, training, recovery, and validation, including an independently prepared cross-size bridge.
- Representation probes and ablations for examining how hidden-state geometry relates to model outputs.

## Explore the work

| Path | Contents |
| --- | --- |
| [Positive-control results](pc-gate-results.md) | Evaluation protocol, per-seed scores, completion rates, and caveats |
| [Cross-size experiment](tier2-results.md) | Initial results and the rollout-length confound |
| [Representation probes](tier0-results.md) and [layer ablations](tier3-results.md) | Measurements, controls, and limitations |
| [Cluster guide](cluster/README.md) | Environment, experiment configuration, and Slurm workflows |
| [Evidence archive](oprd-experiment-results-3/README.md) | Historical run data, logs, provenance, and verification |
| [Tier 0 scripts](tier0/scripts/) | Rollout generation, hidden-state extraction, and probe code |

This work investigates [On-Policy Representation Distillation](https://arxiv.org/abs/2606.06021); it is an independent experimental project, not the official implementation. Large hidden-state artifacts and model checkpoints are not included. The historical checkpoints described in the archive were deleted after their hashes were recorded, so exact weight-level reevaluation of those runs is unavailable.

## Verify archived results

The archive includes a standard-library check for its file hashes and published tables:

```bash
cd oprd-experiment-results-3
python3 build_manifest.py --check
```

GPU runs require the model weights, datasets, compatible CUDA hardware, and the setup described in the [cluster guide](cluster/README.md). Some scripts encode UMD Nexus paths and scheduling assumptions; adapt them to your environment before submitting jobs.
