import json
import pathlib
import sys
import tempfile
import unittest

CLUSTER_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(CLUSTER_DIR / "official_stage2"))
import aime_eval


class AuditableEvaluationTests(unittest.TestCase):
    sampling = {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 16,
        "enable_thinking": False,
    }
    benchmark = {
        "name": "aime24",
        "dataset_id": "fixture/aime24",
        "dataset_revision": "1" * 40,
        "dataset_sha256": "2" * 64,
    }
    identity = {
        "checkpoint_sha256": "3" * 64,
        "model_id": "org/student",
        "model_revision": "4" * 40,
        "tokenizer_id": "org/student",
        "tokenizer_revision": "4" * 40,
        "official_source_commit": "5" * 40,
    }
    problems = [
        {"problem_index": 0, "problem": "p0", "ground_truth": "7"},
        {"problem_index": 1, "problem": "p1", "ground_truth": "9"},
    ]

    @staticmethod
    def grader(response, ground_truth):
        return f"\\boxed{{{ground_truth}}}" in response

    @staticmethod
    def extractor(response):
        marker = "\\boxed{"
        if marker not in response:
            return None
        return response.split(marker)[-1].split("}")[0]

    def record(
        self,
        problem_index,
        rollout_index,
        *,
        response,
        finish_reason,
        token_count,
        seed,
        suffix_count=1,
    ):
        return {
            "benchmark": "aime24",
            "problem_index": problem_index,
            "rollout_index": rollout_index,
            "request_id": f"{problem_index}-{rollout_index}",
            "raw_prompt": f"p{problem_index}",
            "rendered_prompt": f"rendered p{problem_index}",
            "response_text": response,
            "generated_token_count": token_count,
            "finish_reason": finish_reason,
            "stop_reason": None,
            "finish_metadata_source": "vllm",
            "instruction_suffix_count": suffix_count,
            "stop_token_ids": [1, 2],
            "vllm_version": "0.test",
            "sampling": dict(self.sampling),
            "requested_seed": seed,
        }

    def corrected_records(self):
        return [
            self.record(0, 0, response="ok \\boxed{7}", finish_reason="stop", token_count=5, seed=10),
            self.record(0, 1, response="wrong \\boxed{8}", finish_reason="stop", token_count=6, seed=11),
            self.record(1, 0, response="late \\boxed{9}", finish_reason="length", token_count=16, seed=20),
            self.record(1, 1, response="unfinished", finish_reason="length", token_count=16, seed=21),
        ]

    def test_generation_artifact_is_complete_checksummed_and_rescorable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            destination = root / "eval" / "aime24"
            result = aime_eval.write_generation_artifact(
                destination,
                benchmark=self.benchmark,
                track="corrected_seeded_single_suffix",
                sampling=self.sampling,
                identity=self.identity,
                problems=self.problems,
                records=self.corrected_records(),
                samples_per_problem=2,
            )
            self.assertEqual(result["rows"], 4)
            manifest, records, manifest_sha256 = aime_eval.load_generation_artifact(
                destination / "manifest.json"
            )
            self.assertEqual(manifest_sha256, result["manifest_sha256"])
            self.assertEqual(len(records), 4)
            self.assertEqual(manifest["samples"]["sha256"], result["samples_sha256"])

            score_path = destination / "score-v1.json"
            score = aime_eval.score_generation_artifact(
                destination / "manifest.json",
                grader=self.grader,
                extractor=self.extractor,
                grader_identity={"commit": "5" * 40, "entry_point": "fixture"},
                output_path=score_path,
            )
            self.assertEqual(
                score["counts"],
                {
                    "natural_finish_correct": 1,
                    "natural_finish_incorrect": 1,
                    "cap_hit_correct": 1,
                    "cap_hit_incorrect": 1,
                    "format_error": 1,
                },
            )
            self.assertEqual(score["metrics"]["avg_at_16"], 0.5)
            self.assertEqual(score["metrics"]["best_at_16"], 1.0)
            self.assertEqual(score["metrics"]["cap_hit_rate"], 0.5)
            self.assertTrue(score_path.is_file())
            persisted = json.loads(score_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["generation_manifest_sha256"], manifest_sha256)

    def test_artifact_rejects_missing_duplicate_and_wrong_track_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            missing = self.corrected_records()[:-1]
            with self.assertRaisesRegex(aime_eval.EvaluationArtifactError, "coverage"):
                aime_eval.write_generation_artifact(
                    root / "missing",
                    benchmark=self.benchmark,
                    track="corrected_seeded_single_suffix",
                    sampling=self.sampling,
                    identity=self.identity,
                    problems=self.problems,
                    records=missing,
                    samples_per_problem=2,
                )

            duplicate = self.corrected_records()
            duplicate[-1] = dict(duplicate[0])
            with self.assertRaisesRegex(aime_eval.EvaluationArtifactError, "duplicate"):
                aime_eval.write_generation_artifact(
                    root / "duplicate",
                    benchmark=self.benchmark,
                    track="corrected_seeded_single_suffix",
                    sampling=self.sampling,
                    identity=self.identity,
                    problems=self.problems,
                    records=duplicate,
                    samples_per_problem=2,
                )

            wrong_suffix = self.corrected_records()
            wrong_suffix[0] = dict(wrong_suffix[0], instruction_suffix_count=2)
            with self.assertRaisesRegex(aime_eval.EvaluationArtifactError, "exactly one"):
                aime_eval.write_generation_artifact(
                    root / "suffix",
                    benchmark=self.benchmark,
                    track="corrected_seeded_single_suffix",
                    sampling=self.sampling,
                    identity=self.identity,
                    problems=self.problems,
                    records=wrong_suffix,
                    samples_per_problem=2,
                )

    def test_published_exact_requires_unseeded_requests_and_preserves_suffix_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            records = [
                dict(record, requested_seed=None, instruction_suffix_count=2)
                for record in self.corrected_records()
            ]
            aime_eval.write_generation_artifact(
                root / "published",
                benchmark=self.benchmark,
                track="published_offline_exact",
                sampling=self.sampling,
                identity=self.identity,
                problems=self.problems,
                records=records,
                samples_per_problem=2,
            )
            seeded = list(records)
            seeded[0] = dict(seeded[0], requested_seed=1)
            with self.assertRaisesRegex(aime_eval.EvaluationArtifactError, "must not set"):
                aime_eval.write_generation_artifact(
                    root / "seeded",
                    benchmark=self.benchmark,
                    track="published_offline_exact",
                    sampling=self.sampling,
                    identity=self.identity,
                    problems=self.problems,
                    records=seeded,
                    samples_per_problem=2,
                )

    def test_tampering_and_ambiguous_finish_reason_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            destination = root / "artifact"
            aime_eval.write_generation_artifact(
                destination,
                benchmark=self.benchmark,
                track="corrected_seeded_single_suffix",
                sampling=self.sampling,
                identity=self.identity,
                problems=self.problems,
                records=self.corrected_records(),
                samples_per_problem=2,
            )
            with (destination / "samples.jsonl").open("ab") as handle:
                handle.write(b"tampered\n")
            with self.assertRaisesRegex(aime_eval.EvaluationArtifactError, "size mismatch"):
                aime_eval.load_generation_artifact(destination / "manifest.json")

            ambiguous = self.record(
                0,
                0,
                response="no answer",
                finish_reason="unknown",
                token_count=3,
                seed=1,
            )
            with self.assertRaisesRegex(aime_eval.EvaluationArtifactError, "ambiguous"):
                aime_eval._is_cap_hit(ambiguous, 16)


if __name__ == "__main__":
    unittest.main()
