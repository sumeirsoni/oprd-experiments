#!/usr/bin/env python3
import argparse
import hashlib
import json
import pathlib
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
SNAPSHOT = ROOT / "nexus-old-checkout"
RUNS = SNAPSHOT / "cluster" / "runs"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_hash_file(path):
    values = {}
    for line in path.read_text().splitlines():
        digest, absolute = line.split(None, 1)
        values[absolute.strip()] = digest
    return values


def parse_sizes(path):
    values = {}
    for line in path.read_text().splitlines():
        size, absolute = line.split("\t", 1)
        values[absolute] = int(size)
    return values


def run_results(prefix):
    values = {}
    for path in sorted(RUNS.glob(f"{prefix}*/results.json")):
        values[path.parent.name] = json.loads(path.read_text())
    return values


def tier2_reconciliation():
    grouped = {}
    for name, result in run_results("t2_qwen3_4b_to_1p7b_").items():
        arm = name.removeprefix("t2_qwen3_4b_to_1p7b_")
        if arm.endswith(("_s0", "_s1", "_s2")):
            arm = arm.rsplit("_s", 1)[0]
        grouped.setdefault(arm, []).append(result["eval"])

    expected = {
        "untrained_baseline": (3.12, 2.50, 0.76, 0.19),
        "minilmv2_selfrel": (2.71, 0.42, 0.63, 0.30),
        "cka_gram": (0.69, 0.00, 0.53, 0.24),
        "soft_knn_kl": (0.21, 0.07, 0.39, 0.41),
        "ln_dist": (0.07, 0.00, 0.28, 0.51),
        "oprd_bridge": (0.00, 0.00, 0.18, 0.41),
        "linear_map_mse": (0.00, 0.00, 0.11, 0.83),
    }
    rows = {}
    for arm, evaluations in sorted(grouped.items()):
        aime24 = [100 * value["aime24"]["accuracy"] for value in evaluations]
        aime25 = [100 * value["aime25"]["accuracy"] for value in evaluations]
        answered = [value["aime24"]["answered_rate"] for value in evaluations]
        truncated = [value["aime24"]["truncation_rate"] for value in evaluations]
        actual = (
            statistics.mean(aime24),
            statistics.mean(aime25),
            statistics.mean(answered),
            statistics.mean(truncated),
        )
        published = expected[arm]
        rows[arm] = {
            "runs": len(evaluations),
            "aime24_percent_mean": actual[0],
            "aime24_percent_sd": statistics.stdev(aime24) if len(aime24) > 1 else None,
            "aime25_percent_mean": actual[1],
            "aime25_percent_sd": statistics.stdev(aime25) if len(aime25) > 1 else None,
            "aime24_answered_rate_mean": actual[2],
            "aime24_truncation_rate_mean": actual[3],
            "published_rounded": {
                "aime24_percent": published[0],
                "aime25_percent": published[1],
                "answered_rate": published[2],
                "truncation_rate": published[3],
            },
            "matches_published_rounding": (
                round(actual[0], 2) == published[0]
                and round(actual[1], 2) == published[1]
                and round(actual[2], 2) == published[2]
                and round(actual[3], 2) == published[3]
            ),
        }
    return rows


def positive_control_reconciliation():
    expected = {
        "pc_justRL_to_r1_1p5b_untrained_baseline": (28.33, 22.92, 0.717, 0.281),
        "pc_justRL_to_r1_1p5b_opd_topk_s0": (47.92, 33.12, 0.969, 0.031),
        "pc_justRL_to_r1_1p5b_opd_topk_s1": (50.83, 33.96, 0.938, 0.062),
        "pc_justRL_to_r1_1p5b_opd_topk_s2": (47.92, 33.96, 0.935, 0.065),
        "pc_justRL_to_r1_1p5b_minilmv2_selfrel_s0": (35.21, 27.92, 0.906, 0.094),
        "pc_justRL_to_r1_1p5b_minilmv2_selfrel_s1": (34.38, 25.62, 0.902, 0.098),
        "pc_justRL_to_r1_1p5b_minilmv2_selfrel_s2": (36.67, 25.83, 0.898, 0.102),
    }
    rows = {}
    for name, result in sorted(run_results("pc_justRL_to_r1_1p5b_").items()):
        evaluation = result["eval"]
        actual = (
            100 * evaluation["aime24"]["accuracy"],
            100 * evaluation["aime25"]["accuracy"],
            evaluation["aime24"]["answered_rate"],
            evaluation["aime24"]["truncation_rate"],
        )
        published = expected[name]
        rows[name] = {
            "aime24_percent": actual[0],
            "aime25_percent": actual[1],
            "aime24_answered_rate": actual[2],
            "aime24_truncation_rate": actual[3],
            "published_rounded": {
                "aime24_percent": published[0],
                "aime25_percent": published[1],
                "answered_rate": published[2],
                "truncation_rate": published[3],
            },
            "matches_published_rounding": (
                round(actual[0], 2) == published[0]
                and round(actual[1], 2) == published[1]
                and round(actual[2], 3) == published[2]
                and round(actual[3], 3) == published[3]
            ),
        }
    return rows


def build_reconciliation():
    data = {
        "schema_version": 1,
        "tier2_primary_24k": tier2_reconciliation(),
        "positive_control_24k": positive_control_reconciliation(),
    }
    data["all_rows_match_published_rounding"] = all(
        row["matches_published_rounding"]
        for group in (data["tier2_primary_24k"], data["positive_control_24k"])
        for row in group.values()
    )
    return data


def build_manifest():
    checkpoint_hashes = {}
    for name in ("checkpoints-tier2.sha256", "checkpoints-pc.sha256"):
        checkpoint_hashes.update(parse_hash_file(ROOT / name))
    checkpoint_sizes = parse_sizes(ROOT / "checkpoint-sizes.tsv")
    if checkpoint_hashes.keys() != checkpoint_sizes.keys():
        raise SystemExit("checkpoint hash and size inventories do not cover the same files")

    checkpoints = []
    scratch_prefix = "/cmlscratch/ssoni11/oprd-experiments/"
    for absolute in sorted(checkpoint_hashes):
        if not absolute.startswith(scratch_prefix):
            raise SystemExit(f"unexpected checkpoint path: {absolute}")
        checkpoints.append({
            "path": absolute.removeprefix(scratch_prefix),
            "bytes": checkpoint_sizes[absolute],
            "sha256": checkpoint_hashes[absolute],
            "contents_archived": False,
        })

    excluded = {ROOT / "manifest.json"}
    archived_files = []
    for path in sorted(value for value in ROOT.rglob("*") if value.is_file()):
        if path in excluded:
            continue
        archived_files.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    family_sizes = {}
    for prefix, label in (("t1_", "tier1"), ("t2_", "tier2"), ("pc_", "positive_control")):
        paths = [path for path in RUNS.iterdir() if path.is_dir() and path.name.startswith(prefix)]
        family_sizes[label] = {
            "run_directories": len(paths),
            "archived_non_checkpoint_bytes": sum(
                path.stat().st_size for run in paths for path in run.rglob("*") if path.is_file()
            ),
            "checkpoint_bytes_not_archived": sum(
                value["bytes"] for value in checkpoints
                if pathlib.PurePosixPath(value["path"]).parts[2].startswith(prefix)
            ),
        }

    return {
        "schema_version": 1,
        "source_root": "/cmlscratch/ssoni11/oprd-experiments",
        "snapshot_date_utc": "2026-07-29",
        "archived_files": archived_files,
        "checkpoints": checkpoints,
        "families": family_sizes,
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    reconciliation = build_reconciliation()
    if not reconciliation["all_rows_match_published_rounding"]:
        raise SystemExit("raw results do not reproduce the published rounded tables")

    if args.check:
        expected_reconciliation = json.loads((ROOT / "reconciliation.json").read_text())
        if reconciliation != expected_reconciliation:
            raise SystemExit("reconciliation.json is stale")
        expected_manifest = json.loads((ROOT / "manifest.json").read_text())
        actual_manifest = build_manifest()
        if actual_manifest != expected_manifest:
            raise SystemExit("manifest.json is stale")
        print("archive verification passed")
        return

    write_json(ROOT / "reconciliation.json", reconciliation)
    write_json(ROOT / "manifest.json", build_manifest())


if __name__ == "__main__":
    main()
