import argparse
import contextlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import build_oprd_bridge as builder
from oprd_bridge import BridgeBank


class CorpusContractTests(unittest.TestCase):
    def test_smoke_contract_preserves_real_geometry_and_generation(self):
        smoke = builder._smoke_contract()
        self.assertEqual(smoke["attempted_prompts"], 8)
        self.assertEqual(smoke["optimizer"]["epochs"], 2)
        self.assertEqual(smoke["generation"]["max_new_tokens"], 4096)
        self.assertEqual(smoke["student"]["num_layers"], 28)
        self.assertEqual(smoke["teacher"]["num_layers"], 36)
        self.assertEqual(smoke["response_positions"]["k"], 2000)
        self.assertEqual(smoke["paper"]["arxiv_id"], "2606.06021v4")
        self.assertEqual(
            smoke["official_code"]["commit"],
            "a63236c990b1bfa9551e9d020719490d3ecd93e8",
        )

    def test_response_window_keeps_empty_and_final_two_thousand(self):
        empty = {"prompt_token_ids": [1, 2], "output_token_ids": []}
        self.assertEqual(builder._response_window(empty), (2, 0))

        long = {
            "prompt_token_ids": [1, 2, 3],
            "output_token_ids": list(range(2500)),
        }
        self.assertEqual(builder._response_window(long), (503, 2000))

    def test_public_prompt_format_has_no_local_math_instruction(self):
        class FakeTokenizer:
            def __init__(self):
                self.calls = []

            def apply_chat_template(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                if kwargs["tokenize"]:
                    return [10, 11, 12]
                return "<user>original prompt</user><assistant>"

        tokenizer = FakeTokenizer()
        raw = "original prompt"
        self.assertEqual(builder._format_prompt_ids(tokenizer, raw), [10, 11, 12])
        self.assertEqual(
            builder._format_prompt_text(tokenizer, raw),
            "<user>original prompt</user><assistant>",
        )
        for messages, kwargs in tokenizer.calls:
            self.assertEqual(messages, [{"role": "user", "content": raw}])
            self.assertTrue(kwargs["add_generation_prompt"])
            self.assertFalse(kwargs["enable_thinking"])

    def test_prompt_format_rejects_tokenizers_that_cannot_disable_thinking(self):
        class UnsupportedTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                raise TypeError("unexpected keyword argument 'enable_thinking'")

        with self.assertRaisesRegex(builder.BuildError, "enable_thinking=False"):
            builder._format_prompt_ids(UnsupportedTokenizer(), "prompt")
        with self.assertRaisesRegex(builder.BuildError, "enable_thinking=False"):
            builder._format_prompt_text(UnsupportedTokenizer(), "prompt")

    def test_vllm_identity_always_uses_the_contract_tokenizer(self):
        contract = {
            "identity": {
                "student_model_id": "/models/student",
                "student_model_revision": "local-tree-sha256:student",
                "tokenizer_id": "/models/tokenizer",
                "tokenizer_revision": "local-tree-sha256:tokenizer",
            }
        }
        kwargs = builder._vllm_identity_kwargs(contract)
        self.assertEqual(kwargs["model"], "/models/student")
        self.assertEqual(kwargs["tokenizer"], "/models/tokenizer")
        self.assertNotIn("revision", kwargs)
        self.assertNotIn("tokenizer_revision", kwargs)

    def test_supplied_local_revision_must_match_current_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary) / "dataset.jsonl"
            source.write_text('{"prompt": "p"}\n')
            expected = f"local-file-sha256:{builder._sha256_file(source)}"
            self.assertEqual(
                builder._resolve_source_revision(
                    str(source), "dataset", expected
                ),
                expected,
            )
            source.write_text('{"prompt": "changed"}\n')
            with self.assertRaisesRegex(builder.BuildError, "does not match local file"):
                builder._resolve_source_revision(
                    str(source), "dataset", expected
                )

    def test_source_provenance_hashes_atomic_directory_helper(self):
        helper = pathlib.Path(builder.__file__).resolve().with_name(
            "oprd_atomic_directory.py"
        )
        provenance = builder._source_provenance()
        self.assertEqual(
            provenance["source_files_sha256"][helper.name],
            builder._sha256_file(helper),
        )

    def test_prompt_selection_recovers_either_half_of_atomic_pair(self):
        prompts = [
            {
                "attempt_index": 0,
                "dataset_index": 7,
                "source_row": {"prompt": "p"},
                "raw_prompt": "p",
            }
        ]
        metadata = {
            "dataset_id": "dataset",
            "dataset_revision": "revision",
            "dataset_split": "train",
            "dataset_fingerprint": "fingerprint",
            "rows": 10,
            "features": {"prompt": "string"},
            "selected_dataset_indices_sha256": "digest",
        }
        with mock.patch.object(
            builder, "_sample_prompts", return_value=(prompts, metadata)
        ):
            with tempfile.TemporaryDirectory() as temporary:
                paths = builder.Paths(pathlib.Path(temporary))
                builder._atomic_write_jsonl(paths.prompts, prompts)
                actual_prompts, actual_metadata = (
                    builder._load_or_commit_prompt_selection(paths, {})
                )
                self.assertEqual(actual_prompts, prompts)
                self.assertEqual(actual_metadata, metadata)
                self.assertEqual(builder._load_json(paths.corpus_metadata), metadata)

            with tempfile.TemporaryDirectory() as temporary:
                paths = builder.Paths(pathlib.Path(temporary))
                enriched = {**metadata, "runtime": {"python": "test"}}
                builder._atomic_write_json(paths.corpus_metadata, enriched)
                actual_prompts, actual_metadata = (
                    builder._load_or_commit_prompt_selection(paths, {})
                )
                self.assertEqual(actual_prompts, prompts)
                self.assertEqual(actual_metadata, enriched)
                self.assertEqual(builder._load_jsonl(paths.prompts), prompts)

    def test_prompt_selection_rejects_orphan_mismatch(self):
        prompts = [
            {
                "attempt_index": 0,
                "dataset_index": 7,
                "source_row": {"prompt": "p"},
                "raw_prompt": "p",
            }
        ]
        metadata = {
            "dataset_id": "dataset",
            "dataset_revision": "revision",
            "dataset_split": "train",
            "selected_dataset_indices_sha256": "digest",
        }
        with mock.patch.object(
            builder, "_sample_prompts", return_value=(prompts, metadata)
        ):
            with tempfile.TemporaryDirectory() as temporary:
                paths = builder.Paths(pathlib.Path(temporary))
                changed = [{**prompts[0], "raw_prompt": "changed"}]
                builder._atomic_write_jsonl(paths.prompts, changed)
                with self.assertRaisesRegex(builder.BuildError, "deterministic sampling"):
                    builder._load_or_commit_prompt_selection(paths, {})

            with tempfile.TemporaryDirectory() as temporary:
                paths = builder.Paths(pathlib.Path(temporary))
                builder._atomic_write_json(
                    paths.corpus_metadata,
                    {**metadata, "dataset_revision": "changed"},
                )
                with self.assertRaisesRegex(builder.BuildError, "deterministic sampling"):
                    builder._load_or_commit_prompt_selection(paths, {})

    def test_attempt_record_requires_matching_incremental_journal(self):
        contract = {
            "identity": {
                "student_model_id": "student",
                "student_model_revision": "student-revision",
                "tokenizer_id": "tokenizer",
                "tokenizer_revision": "tokenizer-revision",
            }
        }
        row = {
            "attempt_index": 0,
            "dataset_index": 9,
            "source_row": {"prompt": "p", "answer": "42"},
            "raw_prompt": "p",
        }
        provenance = builder._attempt_provenance(
            contract, row, "formatted", [1, 2, 3]
        )
        record = {
            **row,
            "formatted_prompt": "formatted",
            "prompt_token_ids": [1, 2, 3],
            "output_token_ids": [],
            "completion_length": 0,
            "response_text": "",
            "finish_reason": "stop",
            "stop_reason": None,
            "request_finished": True,
            "generation_seed": provenance["generation_seed"],
            "attempt_provenance_sha256": provenance["sha256"],
            "hit_token_cap": False,
            "generation": builder.FIXED_CONTRACT["generation"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "attempt.json"
            builder._atomic_write_json(path, record)
            self.assertFalse(builder._valid_attempt_record(path, provenance, None))
            journal = {
                "attempt_index": 0,
                "attempt_provenance_sha256": provenance["sha256"],
                "record_sha256": builder._sha256_file(path),
            }
            self.assertTrue(builder._valid_attempt_record(path, provenance, journal))
            record["raw_prompt"] = "changed"
            builder._atomic_write_json(path, record)
            self.assertFalse(builder._valid_attempt_record(path, provenance, journal))

    def test_epoch_batches_match_public_padded_token_geometry(self):
        lengths = [40_000, 20_000, 16_000, 8_000, 7_000, 6_000]
        records = {
            index: {"prompt_token_ids": list(range(length)), "output_token_ids": []}
            for index, length in enumerate(lengths)
        }
        batches = builder._epoch_rollout_batches(records, list(reversed(records)))
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(flattened, list(range(len(lengths))))
        self.assertEqual(batches[0], [0])
        for batch in batches:
            self.assertLessEqual(len(batch), 4)
            max_length = max(len(builder._full_ids(records[index])) for index in batch)
            self.assertLessEqual(len(batch) * max_length, 65_536)
        self.assertEqual(
            batches,
            builder._epoch_rollout_batches(records, list(reversed(records))),
        )


class StreamedMathTests(unittest.TestCase):
    def test_atomic_safetensors_rejects_nonfinite_without_touching_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            absent = root / "absent.safetensors"
            with self.assertRaisesRegex(builder.BuildError, "non-finite tensor"):
                builder._atomic_save_safetensors(
                    absent, {"value": torch.tensor([float("nan")])}
                )
            self.assertFalse(absent.exists())

            existing = root / "existing.safetensors"
            builder._atomic_save_safetensors(
                existing, {"value": torch.tensor([1.0])}
            )
            original = existing.read_bytes()
            with self.assertRaisesRegex(builder.BuildError, "non-finite tensor"):
                builder._atomic_save_safetensors(
                    existing, {"value": torch.tensor([float("inf")])}
                )
            self.assertEqual(existing.read_bytes(), original)

    def test_load_pca_rejects_nonfinite_tensors(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temporary:
            paths = builder.Paths(pathlib.Path(temporary))
            paths.teacher.mkdir(parents=True)
            width = builder.FIXED_CONTRACT["teacher"]["hidden_size"]
            rank = builder.FIXED_CONTRACT["rank"]
            save_file(
                {
                    "mean.1": torch.full((width,), float("nan")),
                    "components.1": torch.zeros(rank, width),
                },
                str(paths.pca_tensors),
            )
            builder._atomic_write_json(paths.pca_metadata, {})
            with mock.patch.object(builder, "_layer_map", return_value={1: 1}):
                with self.assertRaisesRegex(builder.BuildError, "non-finite tensor"):
                    builder._load_pca(paths)

    def test_tensor_shard_requires_matching_journal_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "shard.safetensors"
            metadata = {"attempt_index": "0", "input_fingerprint": "abc"}
            builder._atomic_save_safetensors(
                path,
                {"targets": torch.zeros(0, 2, 3, dtype=torch.bfloat16)},
                metadata=metadata,
            )
            journal = {
                "sha256": builder._sha256_file(path),
                "tensor_name": "targets",
                "shape": [0, 2, 3],
                "dtype": "bfloat16",
                "metadata": metadata,
            }
            self.assertTrue(
                builder._valid_committed_tensor_shard(
                    path,
                    journal,
                    tensor_name="targets",
                    expected_shape=(0, 2, 3),
                    expected_dtype=torch.bfloat16,
                    expected_metadata=metadata,
                )
            )
            self.assertFalse(
                builder._valid_committed_tensor_shard(
                    path,
                    None,
                    tensor_name="targets",
                    expected_shape=(0, 2, 3),
                    expected_dtype=torch.bfloat16,
                    expected_metadata=metadata,
                )
            )

    def test_covariance_pca_matches_direct_solution(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=(64, 9)).astype(np.float16)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "matrix.f16"
            matrix = np.memmap(path, mode="w+", dtype=np.float16, shape=values.shape)
            matrix[:] = values
            matrix.flush()
            mean, components, eigenvalues = builder._fit_covariance_pca(
                matrix, rows=64, rank=3, pca_device="cpu", chunk_rows=11
            )

        direct = torch.from_numpy(values.astype(np.float32))
        direct_mean = direct.mean(0)
        covariance = (direct - direct_mean).T @ (direct - direct_mean) / 63
        direct_values, direct_vectors = torch.linalg.eigh(covariance)
        direct_components = direct_vectors[:, -3:].flip(1).T
        self.assertTrue(torch.allclose(mean, direct_mean, atol=2e-4, rtol=2e-4))
        self.assertTrue(
            torch.allclose(
                torch.tensor(eigenvalues),
                direct_values[-3:].flip(0),
                atol=2e-3,
                rtol=2e-3,
            )
        )
        self.assertTrue(
            torch.allclose(
                components.T @ components,
                direct_components.T @ direct_components,
                atol=2e-3,
                rtol=2e-3,
            )
        )

    def test_training_state_rejects_nonfinite_optimizer_or_projector_values(self):
        builder._require_finite_training_state(
            {"projector": torch.ones(2), "optimizer": {"step": 1.0}},
            context="test",
        )
        with self.assertRaisesRegex(builder.BuildError, "non-finite tensor"):
            builder._require_finite_training_state(
                {"optimizer": {"exp_avg": torch.tensor([float("inf")])}},
                context="test",
            )
        with self.assertRaisesRegex(builder.BuildError, "non-finite scalar"):
            builder._require_finite_training_state(
                {"loss_sum": float("nan")}, context="test"
            )

    def test_grouped_adamw_step_matches_monolithic_all_layer_step(self):
        torch.manual_seed(23)
        layers = list(range(1, 29))
        full = BridgeBank(layers, student_dim=3, teacher_dim=4, rank=2)
        grouped = []
        for group_index in range(7):
            group_layers = layers[group_index * 4 : (group_index + 1) * 4]
            bank = BridgeBank(group_layers, student_dim=3, teacher_dim=4, rank=2)
            with torch.no_grad():
                for layer in group_layers:
                    bank.student_projectors[str(layer)].weight.copy_(
                        full.student_projectors[str(layer)].weight
                    )
            grouped.append(bank)

        hidden = {layer: torch.randn(13, 3) for layer in layers}
        targets = {layer: torch.randn(13, 2) for layer in layers}
        full_optimizer = torch.optim.AdamW(
            full.student_projectors.parameters(), lr=1e-4, weight_decay=0.0
        )
        group_optimizers = [
            torch.optim.AdamW(
                bank.student_projectors.parameters(), lr=1e-4, weight_decay=0.0
            )
            for bank in grouped
        ]

        full_optimizer.zero_grad(set_to_none=True)
        full_loss = torch.stack(
            [
                F.mse_loss(full.project_student(layer, hidden[layer]), targets[layer])
                for layer in layers
            ]
        ).mean()
        full_loss.backward()
        full_optimizer.step()

        for bank, optimizer in zip(grouped, group_optimizers):
            optimizer.zero_grad(set_to_none=True)
            group_hidden = torch.stack([hidden[layer] for layer in bank.layers], dim=1)
            group_targets = torch.stack([targets[layer] for layer in bank.layers], dim=1)
            contribution, _ = builder._group_objective(
                bank, bank.layers, group_hidden, group_targets
            )
            contribution.backward()
            optimizer.step()

        for bank in grouped:
            for layer in bank.layers:
                self.assertTrue(
                    torch.allclose(
                        bank.student_projectors[str(layer)].weight,
                        full.student_projectors[str(layer)].weight,
                        atol=1e-8,
                        rtol=1e-7,
                    )
                )


class RestartAndSelectionTests(unittest.TestCase):
    def test_failed_phase_rejects_changed_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = builder.Paths(pathlib.Path(temporary))
            builder._begin_phase(paths, "phase", "first")
            builder._fail_phase(paths, "phase", RuntimeError("interrupted"))
            with self.assertRaises(builder.BuildError):
                builder._begin_phase(paths, "phase", "changed")

    def test_global_selection_uses_earliest_epoch_on_tie(self):
        table = [
            {"epoch": 1, "validation_cosine": 0.8},
            {"epoch": 2, "validation_cosine": 0.9},
            {"epoch": 3, "validation_cosine": 0.9},
        ]
        self.assertEqual(builder._selected_epoch(table), 2)

    def test_global_selection_skips_nonfinite_and_rejects_all_nonfinite(self):
        table = [
            {"epoch": 1, "validation_cosine": float("nan")},
            {"epoch": 2, "validation_cosine": 0.7},
            {"epoch": 3, "validation_cosine": float("inf")},
        ]
        self.assertEqual(builder._selected_epoch(table), 2)
        with self.assertRaises(builder.BuildError):
            builder._selected_epoch(
                [{"epoch": 1, "validation_cosine": float("nan")}]
            )

    def test_global_epoch_table_marks_corrupt_epoch_invalid_but_keeps_others(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = builder.Paths(pathlib.Path(temporary))
            epoch_count = builder.FIXED_CONTRACT["optimizer"]["epochs"]
            group_count = builder.FIXED_CONTRACT["student_groups"]["count"]
            for group_index in range(group_count):
                metrics = {
                    "epochs": [
                        {
                            "epoch": epoch,
                            "train_mse": 1.0,
                            "validation_mse": 1.0,
                            "validation_cosine": epoch / 100,
                            "validation_cosine_sum": epoch / 100,
                            "validation_cosine_count": 1,
                        }
                        for epoch in range(1, epoch_count + 1)
                    ]
                }
                if group_index == 3:
                    metrics["epochs"][1]["validation_cosine"] = float("nan")
                path = paths.group(group_index) / "metrics.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(metrics))

            table = builder._global_epoch_table(paths)
            self.assertFalse(table[1]["valid"])
            self.assertIsNone(table[1]["validation_cosine"])
            self.assertEqual(table[1]["invalid_groups"][0]["group"], 3)
            self.assertEqual(builder._selected_epoch(table), epoch_count)
            builder._canonical_json_bytes(table)

    def test_global_selection_rejects_table_with_no_valid_epoch(self):
        with self.assertRaises(builder.BuildError):
            builder._selected_epoch(
                [
                    {"epoch": 1, "valid": False, "validation_cosine": None},
                    {"epoch": 2, "valid": False, "validation_cosine": None},
                ]
            )

    def test_strict_cache_cleanup_propagates_removal_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            tree = root / "cache"
            journal = root / "journal.json"
            tree.mkdir()
            (tree / "shard").write_text("data")
            journal.write_text("{}")
            with mock.patch.object(
                builder.shutil, "rmtree", side_effect=OSError("busy")
            ):
                with self.assertRaisesRegex(OSError, "busy"):
                    builder._remove_tree_and_journal(tree, journal)
            self.assertTrue(tree.exists())
            self.assertFalse(journal.exists())

    def test_completed_phase_helper_rejects_corrupt_previous_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = builder.Paths(pathlib.Path(temporary))
            output = paths.group(0) / "complete.json"
            output.parent.mkdir(parents=True)
            output.write_text("valid")
            state = builder._empty_state()
            state["phases"]["train-student-group-0"] = {
                "status": "complete",
                "outputs": [
                    {
                        "path": builder._path_label(paths.root, output),
                        "sha256": builder._sha256_tree(output),
                    }
                ],
            }
            builder._atomic_write_json(paths.state, state)
            builder._require_completed_phase_output(
                paths, "train-student-group-0"
            )
            output.write_text("corrupt")
            with self.assertRaisesRegex(builder.BuildError, "checksum failed"):
                builder._require_completed_phase_output(
                    paths, "train-student-group-0"
                )

    def test_verify_checks_required_outputs_before_reusing_completed_phase(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = builder.Paths(pathlib.Path(temporary))
            with mock.patch.object(builder, "_read_contract", return_value={}), mock.patch.object(
                builder,
                "_require_completed_phase_outputs",
                side_effect=builder.BuildError("phase output checksum failed"),
            ), mock.patch.object(builder, "_verify_input_fingerprint") as fingerprint:
                with self.assertRaisesRegex(builder.BuildError, "checksum failed"):
                    builder.command_verify(
                        argparse.Namespace(
                            work_dir=str(paths.root),
                            max_pca_orthogonality_error=1e-4,
                        )
                    )
                fingerprint.assert_not_called()

    def test_verify_fingerprint_includes_orthogonality_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = builder.Paths(pathlib.Path(temporary))
            artifact = paths.root / "artifact"
            artifact.mkdir(parents=True)
            for path in (
                paths.contract,
                paths.rollouts,
                paths.split,
                paths.pca_tensors,
                paths.targets_index,
                paths.selection,
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(path.name)
            contract = {"artifact_dir": str(artifact)}
            first = builder._verify_input_fingerprint(paths, contract, 1e-4)
            second = builder._verify_input_fingerprint(paths, contract, 1e-3)
            self.assertNotEqual(first, second)

    def test_required_phase_outputs_reject_missing_or_changed_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = builder.Paths(pathlib.Path(temporary))
            state = builder._empty_state()
            output_paths = {}
            for phase_name in builder._required_phase_names(include_publish=False):
                output = paths.root / f"{phase_name}.txt"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(phase_name)
                output_paths[phase_name] = output
                state["phases"][phase_name] = {
                    "status": "complete",
                    "outputs": [
                        {
                            "path": builder._path_label(paths.root, output),
                            "sha256": builder._sha256_tree(output),
                        }
                    ],
                }
            builder._atomic_write_json(paths.state, state)
            checked = builder._require_completed_phase_outputs(
                paths, include_publish=False
            )
            self.assertEqual(
                checked, builder._required_phase_names(include_publish=False)
            )

            output_paths["fit-teacher-pca"].write_text("changed")
            with self.assertRaisesRegex(builder.BuildError, "checksum failed"):
                builder._require_completed_phase_outputs(
                    paths, include_publish=False
                )

            output_paths["fit-teacher-pca"].write_text("fit-teacher-pca")
            state = builder._read_state(paths)
            del state["phases"]["materialize-teacher-targets"]
            builder._atomic_write_json(paths.state, state)
            with self.assertRaisesRegex(builder.BuildError, "not complete"):
                builder._require_completed_phase_outputs(
                    paths, include_publish=False
                )

    def test_publish_recovers_after_rename_before_phase_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = builder.Paths(pathlib.Path(temporary) / "work")
            artifact = pathlib.Path(temporary) / "artifact"
            paths.root.mkdir(parents=True)
            contract = {
                "schema_version": builder.SCHEMA_VERSION,
                "artifact_dir": str(artifact),
                "stage1_contract_sha256": "a" * 64,
                "identity": {
                    "student_model_id": "org/student",
                    "student_model_revision": "1" * 40,
                    "teacher_model_id": "org/teacher",
                    "teacher_model_revision": "2" * 40,
                    "tokenizer_id": "org/tokenizer",
                    "tokenizer_revision": "3" * 40,
                },
            }
            builder._atomic_write_json(paths.contract, contract)
            builder._atomic_write_jsonl(paths.rollouts, [{"attempt_index": 0}])
            builder._atomic_write_json(paths.split, {"train_indices": [0]})
            builder._atomic_write_json(paths.corpus_metadata, {"source": "test"})
            builder._atomic_write_json(paths.pca_metadata, {"sampled_rows": 1})
            paths.pca_tensors.parent.mkdir(parents=True, exist_ok=True)
            paths.pca_tensors.write_bytes(b"pca")
            paths.targets.mkdir(parents=True)
            (paths.targets / "target").write_bytes(b"target")
            builder._atomic_write_json(paths.targets_index, {"entries": []})
            for group_index in range(
                builder.FIXED_CONTRACT["student_groups"]["count"]
            ):
                group = paths.group(group_index)
                group.mkdir(parents=True)
                builder._atomic_write_json(
                    group / "metrics.json", {"group": group_index}
                )
                builder._atomic_write_json(
                    group / "complete.json", {"group": group_index}
                )
                epochs = group / "epochs"
                epochs.mkdir()
                (epochs / "epoch_01.safetensors").write_bytes(b"checkpoint")

            bridge = BridgeBank([1], student_dim=4, teacher_dim=6, rank=2)
            with torch.no_grad():
                bridge.student_projectors["1"].weight.copy_(
                    torch.arange(8, dtype=torch.float32).reshape(2, 4)
                )
                bridge.set_teacher_projection(
                    1,
                    torch.arange(6, dtype=torch.float32),
                    torch.arange(12, dtype=torch.float32).reshape(2, 6),
                )
            bridge.freeze()
            table = [{"epoch": 1, "validation_cosine": 0.75, "valid": True}]
            expected = {
                "expected_student_model_id": "org/student",
                "expected_student_model_revision": "1" * 40,
                "expected_teacher_model_id": "org/teacher",
                "expected_teacher_model_revision": "2" * 40,
                "expected_tokenizer_id": "org/tokenizer",
                "expected_tokenizer_revision": "3" * 40,
                "expected_stage1_contract_sha256": "a" * 64,
                "expected_corpus_id": builder._corpus_id(paths),
                "expected_layer_map": {1: 1},
                "expected_rank": 2,
                "expected_student_dim": 4,
                "expected_teacher_dim": 6,
            }
            patches = (
                mock.patch.object(builder, "_read_contract", return_value=contract),
                mock.patch.object(
                    builder, "_require_completed_phase_outputs", return_value=[]
                ),
                mock.patch.object(builder, "_require_rollouts", return_value=([], {})),
                mock.patch.object(builder, "_global_epoch_table", return_value=table),
                mock.patch.object(builder, "_selected_epoch", return_value=1),
                mock.patch.object(
                    builder, "_build_published_bridge", return_value=bridge
                ),
                mock.patch.object(
                    builder, "_artifact_expected_kwargs", return_value=expected
                ),
                mock.patch.object(builder, "_layer_map", return_value={1: 1}),
            )
            with contextlib.ExitStack() as stack:
                for patch in patches:
                    stack.enter_context(patch)
                exact_artifact = stack.enter_context(
                    mock.patch.object(
                        builder,
                        "_assert_exact_artifact",
                        wraps=builder._assert_exact_artifact,
                    )
                )
                with mock.patch.object(
                    builder, "_finish_phase", side_effect=RuntimeError("after rename")
                ):
                    with self.assertRaisesRegex(RuntimeError, "after rename"):
                        builder.command_publish(
                            argparse.Namespace(work_dir=str(paths.root))
                        )
                self.assertEqual(exact_artifact.call_count, 0)
                self.assertTrue((artifact / "COMPLETE").is_file())
                self.assertEqual(
                    builder._read_state(paths)["phases"]["publish"]["status"],
                    "failed",
                )
                with mock.patch.object(
                    builder, "_runtime_versions", return_value={"changed": True}
                ):
                    with self.assertRaisesRegex(
                        builder.BuildError, "publish runtime changed"
                    ):
                        builder.command_publish(
                            argparse.Namespace(work_dir=str(paths.root))
                        )
                with mock.patch.object(
                    builder, "_source_provenance", return_value={"changed": True}
                ):
                    builder.command_publish(
                        argparse.Namespace(work_dir=str(paths.root))
                    )
                self.assertEqual(exact_artifact.call_count, 1)

            state = builder._read_state(paths)
            self.assertEqual(state["phases"]["publish"]["status"], "complete")
            self.assertTrue((artifact / "COMPLETE").is_file())

    def test_runtime_parser_rejects_nonpositive_validation_chunk(self):
        args = argparse.Namespace(batch_rows=0)
        with self.assertRaises(builder.BuildError):
            builder._validate_runtime_args(args)


if __name__ == "__main__":
    unittest.main()
