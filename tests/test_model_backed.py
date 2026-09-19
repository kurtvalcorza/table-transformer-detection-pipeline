"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): the zero-shot
evaluation and new heads on synthetic 'table' photos, a one-epoch unfreeze of the last decoder layer, and the
artifact round trip with head, box-head and decoder tensors. Skipped when the weights are absent."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import torch
from PIL import Image

from table_transformer_detection_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    POLICY_FROZEN,
    WEIGHTS_FILE,
    TableTransformerDetectionPipeline,
)

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHTS_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)


def _photo(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    canvas = np.full((240, 320, 3), 225, dtype=np.uint8)
    x0, y0 = 40 + seed * 7 % 60, 30 + seed * 11 % 50
    x1, y1 = x0 + 150, y0 + 120
    canvas[y0:y1, x0:x1] = 240
    for row in range(y0 + 8, y1 - 8, 16):  # ruled lines: something table-like
        canvas[row : row + 2, x0 + 6 : x1 - 6] = rng.integers(20, 60, size=3, dtype=np.uint8)
    for col in range(x0 + 8, x1 - 8, 40):
        canvas[y0 + 6 : y1 - 6, col : col + 2] = 40
    return {
        "id": f"s{seed:02d}",
        "image": Image.fromarray(canvas),
        "boxes": [[float(x0), float(y0), float(x1), float(y1)]],
    }


RECORDS = [_photo(i) for i in range(12)]


@pytest.fixture(scope="module")
def pipe():
    return TableTransformerDetectionPipeline.from_pretrained(device="cpu")


def test_zero_shot_evaluation_is_finite_and_labelled(pipe):
    metrics = pipe.evaluate_zero_shot(RECORDS[9:])
    assert (
        metrics["n_images"] == 3 and metrics["adapted"] is False and metrics["policy"].startswith("zero-shot")
    )
    assert 0.0 <= metrics["ap50"] <= 1.0 and 0.0 <= metrics["mean_best_iou"] <= 1.0


def test_new_heads_then_one_epoch_unfreeze_and_artifact_round_trip(pipe, tmp_path):
    result = pipe.adapt(RECORDS[:9], RECORDS[9:], head_steps=60, trainable_layers=1, epochs=1, lr=1e-4)
    assert result["classes"] == ["nutrition-table"] and result["history"][0]["stage"].startswith("new heads")
    assert result["n_trainable_head"] == 256 * 2 + 2 + 132_612 and result["n_trainable_layers"] == 1_578_752
    assert result["n_total"] == 28_799_431 and len(result["history"]) == 2 and result["best_epoch"] in (0, 1)
    assert (result["policy"] == POLICY_FROZEN) == (result["best_epoch"] == 0)
    metrics = pipe.evaluate(RECORDS[9:])
    assert metrics["n_images"] == 3 and metrics["adapted"] is True and metrics["policy"] == result["policy"]
    assert metrics["loss"] == pytest.approx(result["history"][result["best_epoch"]]["val"]["loss"], abs=1e-4)
    detections = pipe.detect_adapted(RECORDS[0]["image"], threshold=0.05)
    assert detections["classes"] == ["nutrition-table"] and all(
        d["label"] == "nutrition-table" for d in detections["detections"]
    )
    base = pipe.detect(RECORDS[0]["image"], threshold=0.05)
    assert all(
        d["label"] in ("table", "table rotated") for d in base["detections"]
    )  # the PubTables heads still answer
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert "head.weight" in manifest["tensors"] and "bbox_head.layers.0.weight" in manifest["tensors"]
    assert any(t.startswith("model.decoder.layers.5.") for t in manifest["tensors"]) == (
        result["best_epoch"] > 0
    )
    reloaded = TableTransformerDetectionPipeline.from_artifact(artifact, device="cpu")
    assert reloaded.predict_boxes(RECORDS[:2]) == pipe.predict_boxes(RECORDS[:2])
    assert reloaded.adapter["best_epoch"] == result["best_epoch"] and reloaded.classes == result["classes"]


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_policy(pipe, tmp_path):
    """A manifest that claims the frozen policy but carries decoder tensors, records another layer count than
    the tensors it lists, or whose payload carries an extra head tensor is refused before any tensor is
    applied."""
    import shutil

    from safetensors.torch import load_file, save_file

    result = pipe.adapt(RECORDS[:9], None, head_steps=40, trainable_layers=1, epochs=1, lr=1e-4)
    assert result["policy"].startswith("unfrozen")
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert any(t.startswith("model.decoder.layers.5.") for t in manifest["tensors"])
    claims_frozen = tmp_path / "claims_frozen"
    shutil.copytree(artifact, claims_frozen)
    adapter = {**manifest["adapter"], "policy": POLICY_FROZEN, "trainable_layers": 0}
    (claims_frozen / "manifest.json").write_text(json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded policy"):
        TableTransformerDetectionPipeline.from_artifact(claims_frozen, device="cpu")
    other_layers = tmp_path / "other_layers"
    shutil.copytree(artifact, other_layers)
    adapter = {
        **manifest["adapter"],
        "policy": "unfrozen last 2 decoder layers + new heads",
        "trainable_layers": 2,
    }
    (other_layers / "manifest.json").write_text(json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded policy"):
        TableTransformerDetectionPipeline.from_artifact(other_layers, device="cpu")
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["head.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    size = (extra / "adapter.safetensors").stat().st_size
    files = [{**manifest["files"][0], "bytes": size, "sha256": digest}]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        TableTransformerDetectionPipeline.from_artifact(extra, device="cpu")


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe):
    """A failure inside the unfreeze leaves the base exactly as it was, frozen, with no heads or adapter."""
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(RECORDS[:9], None, head_steps=40, trainable_layers=1, epochs=2, lr=1e-4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert pipe._head is None and not any(p.requires_grad for p in pipe._model.parameters())
