"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from table_transformer_detection_pipeline import (
    DETECTION_THRESHOLD,
    INPUT_SCHEMA,
    LABELS,
    MAX_DETECTIONS,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)

REFERENCES = [[70.0, 330.0, 780.0, 660.0], [70.0, 760.0, 430.0, 990.0]]


def _image(width: int = 850, height: int = 1100) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def _result(detections: list[dict]) -> dict:
    return {"detections": detections, "threshold": DETECTION_THRESHOLD, "width": 850, "height": 1100}


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), names=["page.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["labels"] == list(LABELS)
    assert manifest["schema"]["max_detections"] == MAX_DETECTIONS
    assert manifest["inputs"] == [{"id": "page.png", "mode": "RGB", "size": [850, 1100]}]
    assert manifest["threshold"] == DETECTION_THRESHOLD
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_explicit_threshold() -> None:
    manifest = validate_inputs(_image(), threshold=0.5)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["threshold"] == 0.5


def test_validate_inputs_rejects_like_detect() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(_image(MAX_IMAGE_SIDE + 1, 64))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8))
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        validate_inputs("not an image")
    with pytest.raises(ValueError, match="threshold"):
        validate_inputs(_image(), threshold=1.5)
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_image(), names=["a", "b"])


def test_evaluation_report_not_measurable_without_reference_boxes() -> None:
    detection = {"box": [75.0, 342.0, 692.0, 646.0], "label": "table", "score": 0.997}
    report = evaluation_report(_result([detection]))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_detections"] == 1
    assert "box_iou" in report["needs"]
    assert report["baselines"] == []
    assert report["threshold"] == DETECTION_THRESHOLD
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_reference_boxes() -> None:
    detections = [
        {"box": [75.0, 782.0, 361.0, 974.0], "label": "table", "score": 0.998},
        {"box": [70.0, 330.0, 780.0, 660.0], "label": "table", "score": 0.997},
    ]
    report = evaluation_report(_result(detections), REFERENCES, sample_kind="synthetic")
    assert report["verdict"] == "sample-sanity"
    assert report["sample_kind"] == "synthetic"
    assert [metric["id"] for metric in report["metrics"]] == ["box_iou", "box_iou"]
    assert [metric["reference"] for metric in report["metrics"]] == ["table-0", "table-1"]
    assert report["metrics"][0]["value"] == pytest.approx(1.0)
    assert report["metrics"][0]["matched_label"] == "table"
    assert 0.0 < report["metrics"][1]["value"] < 1.0
    assert all(metric["estimation"] for metric in report["metrics"])


def test_evaluation_report_handles_zero_detections() -> None:
    report = evaluation_report(_result([]), REFERENCES)
    assert report["n_detections"] == 0
    assert [metric["value"] for metric in report["metrics"]] == [0.0, 0.0]
    assert [metric["matched_label"] for metric in report["metrics"]] == [None, None]
