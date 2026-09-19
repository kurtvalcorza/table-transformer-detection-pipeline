"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): the zero-shot
evaluation and new heads on synthetic 'table' photos, a one-epoch unfreeze of the last decoder layer, and the
artifact round trip with head, box-head and decoder tensors. Skipped when the weights are absent."""

from __future__ import annotations

import json

import numpy as np
import pytest
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
