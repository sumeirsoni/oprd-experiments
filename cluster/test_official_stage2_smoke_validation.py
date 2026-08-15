import pathlib
import sys
import tempfile
import unittest

CLUSTER = pathlib.Path(__file__).resolve().parent
OFFICIAL = CLUSTER / "official_stage2"
if str(OFFICIAL) not in sys.path:
    sys.path.insert(0, str(OFFICIAL))

import validate_smoke


class OfficialStage2SmokeValidationTests(unittest.TestCase):
    def _evidence(self, root: pathlib.Path):
        run = root / "run"
        project = run / "output"
        runtime = run / "official-runtime.log"
        run.mkdir()
        (run / "actual-hydra.yaml").write_text("trainer:\n  n_gpus_per_node: 4\n")
        (run / "actual-python-argv.json").write_text("[]\n")
        terminal = project / "logs" / "terminal" / "run.log"
        terminal.parent.mkdir(parents=True)
        terminal.write_text("training metrics were emitted to the runtime stream\n")
        step = project / "experiment" / "global_step_10" / "actor"
        step.mkdir(parents=True)
        for kind in ("model", "optim", "extra_state"):
            for rank in range(4):
                (step / f"{kind}_world_size_4_rank_{rank}.pt").write_bytes(b"x")
        (step / "huggingface").mkdir()
        (step / "huggingface" / "config.json").write_text("{}\n")
        return run, project, runtime

    def test_accepts_final_step_from_runtime_with_known_shutdown_warning(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, project, runtime = self._evidence(pathlib.Path(temporary))
            runtime.write_text(
                "training/global_step:10\n"
                "Traceback (most recent call last):\n"
                "RuntimeError: DataLoader worker (pid 1) is killed by signal: Killed.\n"
            )
            result = validate_smoke.validate_smoke(
                run=run, project=project, runtime_log=runtime, rows=80, world_size=4
            )
            self.assertEqual(result["expected_updates"], 10)
            self.assertEqual(result["cleanup_warning_count"], 1)
            self.assertTrue(result["topology_deviation_from_official"])

    def test_rejects_unexplained_runtime_traceback(self):
        with tempfile.TemporaryDirectory() as temporary:
            run, project, runtime = self._evidence(pathlib.Path(temporary))
            runtime.write_text(
                "training/global_step:10\n"
                "Traceback (most recent call last):\n"
                "RuntimeError: unexpected failure\n"
            )
            with self.assertRaisesRegex(
                validate_smoke.SmokeValidationError, "not explained"
            ):
                validate_smoke.validate_smoke(
                    run=run,
                    project=project,
                    runtime_log=runtime,
                    rows=80,
                    world_size=4,
                )


if __name__ == "__main__":
    unittest.main()
