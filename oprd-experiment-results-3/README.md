# Historical Nexus run evidence archive

This archive was pulled from `/cmlscratch/ssoni11/oprd-experiments` on 2026-07-29 before considering removal of the old scratch checkout.

It preserves:

- non-checkpoint contents of every historical run directory;
- all historical Slurm logs from the old checkout;
- the exact historical training and submission scripts that remained on scratch;
- SHA-256 hashes and byte sizes for every checkpoint and learned map that remained on scratch;
- a machine-generated reconciliation of the primary historical Tier-2 and positive-control tables.

The large `.pt` and `.safetensors` files are deliberately not copied into Git. Their identities are recorded in `checkpoints-tier2.sha256`, `checkpoints-pc.sha256`, `checkpoint-sizes.tsv`, and `manifest.json`.

`nexus-old-checkout/cluster/runs/` contains the latest Nexus versions of `config.json`, `results.json`, `train.log`, and other small run evidence. In particular, these Tier-2 results contain the 24k reevaluation that superseded the older copies in `oprd-experiment-results-2/`.

Run `python3 build_manifest.py --check` from this directory to recompute all archived hashes and verify that the raw results reproduce the published tables within their documented rounding.

After separate explicit approval, the 30 hash-verified checkpoint and learned-map files were deleted from scratch on 2026-07-29. This reclaimed 105,768,125,334 bytes and left all run directories, raw reported metrics, configurations, logs, historical scripts, and recorded checkpoint identities intact. Exact weight-level reevaluation of those historical runs is no longer possible from Nexus scratch.
