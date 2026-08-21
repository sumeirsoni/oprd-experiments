"""Fail-closed task-0 dataloader resume instrumentation.

Activated only when OPRD_RESUME_DATALOADER_EVIDENCE is set by the dedicated launcher.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


evidence_raw = os.environ.get("OPRD_RESUME_DATALOADER_EVIDENCE")
if evidence_raw:
    import torch
    from torchdata.stateful_dataloader import StatefulDataLoader

    evidence = pathlib.Path(evidence_raw).absolute()
    authorized = pathlib.Path(os.environ["OPRD_RESUME_DATA_PATH"]).resolve(strict=True)
    expected_sha256 = os.environ["OPRD_RESUME_DATA_SHA256"]
    expected_state = json.loads(os.environ["OPRD_RESUME_DATA_STATE"])
    real_torch_load = torch.load
    real_load_state_dict = StatefulDataLoader.load_state_dict
    observed: dict[str, object] = {}

    def authenticated_torch_load(path, *args, **kwargs):
        value = real_torch_load(path, *args, **kwargs)
        try:
            resolved = pathlib.Path(path).resolve(strict=True)
        except (TypeError, OSError):
            return value
        if resolved == authorized:
            if _sha256(resolved) != expected_sha256:
                raise RuntimeError("authorized resume data.pt digest changed")
            if not isinstance(value, dict):
                raise RuntimeError("authorized resume data.pt is not a state dictionary")
            actual = {
                "snapshot_step": value.get("_snapshot_step"),
                "samples_yielded": value.get("samples_yielded"),
                "sampler_iter_yielded": value.get("_sampler_iter_yielded"),
                "num_workers": value.get("_num_workers"),
                "iterator_finished": value.get("_iterator_finished"),
            }
            if actual != expected_state:
                raise RuntimeError(f"authorized resume data.pt state changed: {actual!r}")
            observed.clear()
            observed.update({"object_id": id(value), "state": actual})
        return value

    def authenticated_load_state_dict(self, state_dict):
        result = real_load_state_dict(self, state_dict)
        if observed.get("object_id") == id(state_dict):
            record = {
                "status": "dataloader_state_restored",
                "data_path": str(authorized),
                "data_sha256": expected_sha256,
                "state": observed["state"],
            }
            payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            evidence.parent.mkdir(parents=True, exist_ok=True)
            temporary = evidence.parent / f".{evidence.name}.tmp-{os.getpid()}"
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.link(temporary, evidence)
                directory_fd = os.open(evidence.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
            observed.clear()
        return result

    torch.load = authenticated_torch_load
    StatefulDataLoader.load_state_dict = authenticated_load_state_dict
