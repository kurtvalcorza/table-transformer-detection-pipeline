"""Offline tests for the box-labelled dataset contract, the pinned Open Food Facts record table and reader,
the seeded product-level split, the AP / operating-point metrics with the fixed-box prior, BYOD loaders
(directory and zip), the DETR set loss and matcher on tensors, artifact-manifest rejections and adapt()
argument validation. No model library beyond torch tensors is loaded; the corpus is served through an injected
fetcher of small synthetic JPEGs."""

# ruff: noqa: E501  -- assertion lines are kept on one line

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
import torch
from PIL import Image

from table_transformer_detection_pipeline import (
    ARTIFACT_FORMAT,
    CLASS_NAME,
    DECODER_LAYERS,
    DETECTION_THRESHOLD,
    MODEL_ID,
    MODEL_REVISION,
    NUM_QUERIES,
    SAMPLE_RECORDS,
    SAMPLE_SPLIT,
    WEIGHT_SHA256,
    TableTransformerDetectionPipeline,
    average_precision,
    build_sample_dataset,
    check_split_disjoint,
    dataset_digest,
    detection_metrics,
    fetch_sample_dataset,
    fixed_box_prior,
    image_digest,
    load_byod_dataset,
    prior_baseline,
    split_dataset,
    split_summary,
    validate_dataset,
    write_dataset_csv,
)
from table_transformer_detection_pipeline import pipeline as pl
from table_transformer_detection_pipeline import samples as sm
from table_transformer_detection_pipeline.samples import fetch_corpus, read_corpus

W, H = 96, 72


def _photo(seed: int, size=(W, H), boxes=None) -> tuple[Image.Image, list[list[float]]]:
    """A synthetic product photo: a dark rectangle (the 'table') on a light background."""
    rng = np.random.default_rng(seed)
    canvas = np.full((size[1], size[0], 3), 220, dtype=np.uint8)
    boxes = boxes or [[10 + seed % 5, 8 + seed % 3, 60 + seed % 7, 50 + seed % 4]]
    for x0, y0, x1, y1 in boxes:
        canvas[int(y0) : int(y1), int(x0) : int(x1)] = rng.integers(20, 60, size=3, dtype=np.uint8)
    return Image.fromarray(canvas), [[float(v) for v in b] for b in boxes]


def _jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


def _records(n=12):
    out = []
    for i in range(n):
        image, boxes = _photo(i)
        out.append({"id": f"r{i:02d}", "image": image, "boxes": boxes, "barcode": f"b{i:03d}"})
    return out


def _pipeline_without_model():
    return TableTransformerDetectionPipeline(lambda image, threshold: [], "cpu")


def _pin(monkeypatch, n=12):
    """Replace the pinned table with synthetic photos via an injected fetcher (files keyed by URL path)."""
    files = {}
    table = []
    for i in range(n):
        image, boxes = _photo(i)
        data = _jpeg(image)
        path = f"000/000/000/{i:04d}/1.jpg"
        files[sm.image_url(path)] = data
        norm = ";".join(f"{y0 / H:.6f},{x0 / W:.6f},{y1 / H:.6f},{x1 / W:.6f}" for x0, y0, x1, y1 in boxes)
        table.append(
            (
                f"{i:013d}_1",
                f"{i:013d}",
                path,
                W,
                H,
                len(data),
                hashlib.sha256(data).hexdigest(),
                norm,
                CLASS_NAME,
            )
        )
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", tuple(table))
    return files


# --- pinned table and reader --------------------------------------------------------------------------


def test_pinned_record_table_is_complete_and_traceable():
    assert len(SAMPLE_RECORDS) == 121
    assert all(
        len(r) == 9 and len(r[6]) == 64 and r[5] > 10_000 and r[3] >= 332 and r[4] >= 133
        for r in SAMPLE_RECORDS
    )
    assert len({r[0] for r in SAMPLE_RECORDS}) == 121 and len({r[2] for r in SAMPLE_RECORDS}) == 121
    assert len({r[1] for r in SAMPLE_RECORDS}) == 119  # two products have two photographs
    assert all(
        1 <= len(r[7].split(";")) <= 3 and len(r[7].split(";")) == len(r[8].split(";"))
        for r in SAMPLE_RECORDS
    )
    assert all(
        all(0.0 <= float(v) <= 1.0 for chunk in r[7].split(";") for v in chunk.split(","))
        for r in SAMPLE_RECORDS
    )
    assert sm.image_url(SAMPLE_RECORDS[0][2]).startswith("https://static.openfoodfacts.org/images/products/")
    assert sum(r[5] for r in SAMPLE_RECORDS) == sm.CORPUS_BYTES
    assert sum(SAMPLE_SPLIT.values()) == 119 and set(SAMPLE_SPLIT) == {"train", "validation", "test"}


def test_fetch_corpus_verifies_each_file_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    files = _pin(monkeypatch)
    calls = []

    def fetcher(url):
        calls.append(url)
        return files[url]

    corpus = fetch_corpus(cache_dir=tmp_path, fetcher=fetcher)
    assert len(corpus) == 12 and fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == corpus
    assert len(calls) == 12 and all(u.startswith(sm.CORPUS_BASE_URL) for u in calls)
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_read_corpus_decodes_records_with_boxes_and_provenance(tmp_path, monkeypatch, forbid_model_imports):
    files = _pin(monkeypatch)
    corpus = read_corpus(fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: files[url]))
    first = corpus[0]
    assert first["id"] == f"{0:013d}_1" and first["barcode"] == f"{0:013d}" and first["image"].size == (W, H)
    assert first["boxes"][0] == pytest.approx([10.0, 8.0, 60.0, 50.0], abs=0.01) and first["categories"] == [
        CLASS_NAME
    ]
    assert (
        first["source_size"] == [W, H]
        and first["dropped_boxes"] == 0
        and first["image_url"].endswith("/1.jpg")
    )
    with pytest.raises(ValueError, match="missing"):
        read_corpus({})


def test_read_corpus_downscales_large_originals_and_drops_slivers(monkeypatch):
    big, _ = _photo(1, size=(2000, 1500))
    data = _jpeg(big)
    boxes = "0.100000,0.100000,0.500000,0.600000;0.500000,0.500000,0.500500,0.500500"
    monkeypatch.setattr(
        sm,
        "SAMPLE_RECORDS",
        (
            (
                "big_1",
                "b",
                "p/1.jpg",
                2000,
                1500,
                len(data),
                hashlib.sha256(data).hexdigest(),
                boxes,
                f"{CLASS_NAME};{CLASS_NAME}",
            ),
        ),
    )
    record = read_corpus({"big_1": data})[0]
    assert record["image"].size == (1280, 960) and record["dropped_boxes"] == 1 and len(record["boxes"]) == 1
    assert record["boxes"][0] == pytest.approx([0.1 * 1280, 0.1 * 960, 0.6 * 1280, 0.5 * 960], abs=0.01)
    with pytest.raises(ValueError, match="served image is"):
        read_corpus({"big_1": _jpeg(_photo(1, size=(20, 15))[0])})


def test_sample_split_is_by_product_seeded_and_disjoint(tmp_path, monkeypatch, forbid_model_imports):
    files = _pin(monkeypatch)
    corpus = read_corpus(fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: files[url]))
    corpus.append({**corpus[0], "id": "extra", "image": _photo(99)[0]})  # a second photo of product 0
    sizes = {"train": 8, "validation": 2, "test": 2}
    splits = build_sample_dataset(corpus, seed=42, sizes=sizes)
    assert sum(check_split_disjoint(splits).values()) == 13
    summary = split_summary(splits)
    assert summary["train"]["products"] == 8 and summary["train"]["boxes"] >= 8
    assert splits["train"][0]["id"] == "train-000" and "source_id" in splits["train"][0]
    assert (
        build_sample_dataset(corpus, seed=42, sizes=sizes)["test"][0]["source_id"]
        == splits["test"][0]["source_id"]
    )
    assert (
        build_sample_dataset(corpus, seed=7, sizes=sizes)["test"][0]["source_id"]
        != splits["test"][0]["source_id"]
    )
    with pytest.raises(ValueError, match="only"):
        build_sample_dataset(corpus, sizes={"train": 12, "validation": 1, "test": 1})
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": splits["train"], "test": [splits["train"][0]]})
    with pytest.raises(ValueError, match="photographs in both"):
        check_split_disjoint(
            {
                "train": [{**splits["train"][0], "barcode": "same"}],
                "test": [{**splits["test"][0], "barcode": "same"}],
            }
        )


def test_fetch_sample_dataset_end_to_end_with_injected_fetcher(tmp_path, monkeypatch, forbid_model_imports):
    files = _pin(monkeypatch)
    splits = fetch_sample_dataset(
        cache_dir=tmp_path, fetcher=lambda url: files[url], sizes={"train": 8, "validation": 2, "test": 2}
    )
    manifests = {name: validate_dataset(part, min_records=1) for name, part in splits.items()}
    assert manifests["train"]["n_records"] == 8 and manifests["train"]["n_boxes"] == 8
    assert len({m["digest"] for m in manifests.values()}) == 3


# --- dataset contract ----------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(tmp_path, forbid_model_imports):
    records = _records()
    report = validate_dataset(records)
    assert (
        report["n_records"] == 12
        and report["n_boxes"] == 12
        and report["boxes_per_image"] == {"min": 1, "max": 1}
    )
    assert (
        report["image_side"] == {"min": W, "max": W}
        and 0 < report["box_area_fraction"]["min"] <= report["box_area_fraction"]["max"] < 1
    )
    assert report["model_id"] == MODEL_ID and report["digest"] == dataset_digest(records)
    records[0]["image"].save(tmp_path / "p.jpg")
    from_path = validate_dataset(
        [{"id": "p", "image": tmp_path / "p.jpg", "boxes": records[0]["boxes"]}, *records[1:]]
    )
    assert from_path["records"][0]["image"].size == (W, H)
    bad = [
        ({**records[0], "id": "bad id"}, "id must match"),
        ({**records[0], "image": "nope.jpg"}, "image file not found"),
        ({**records[0], "image": Image.new("RGB", (pl.MAX_IMAGE_SIDE + 1, 20))}, "image side"),
        ({**records[0], "boxes": []}, "boxes must be a list"),
        ({**records[0], "boxes": [[1, 2, 3]]}, "four values"),
        ({**records[0], "boxes": [[10, 10, 5, 20]]}, "x0 < x1"),
        ({**records[0], "boxes": [[0, 0, W + 1, 10]]}, "inside"),
        ({**records[0], "boxes": [[10, 10, 12, 30]]}, "smaller than"),
        ({"id": "x", "image": records[0]["image"]}, "missing 'boxes'"),
    ]
    for record, message in bad:
        with pytest.raises(ValueError, match=message):
            validate_dataset([record, *records[1:]])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], records[0], *records[2:]])
    with pytest.raises(ValueError, match="8..2000"):
        validate_dataset(records[:3])
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"})


def test_split_dataset_groups_by_product_deduplicates_and_is_seeded(forbid_model_imports):
    records = _records(20)
    duplicated = [*records, {**records[0], "id": "dup", "barcode": "other"}]
    splits = split_dataset(duplicated, val_fraction=0.2, test_fraction=0.2, seed=1)
    assert sum(len(part) for part in splits.values()) == 20
    check_split_disjoint(splits)
    assert {
        r["id"] for r in split_dataset(duplicated, val_fraction=0.2, test_fraction=0.2, seed=1)["test"]
    } == {r["id"] for r in splits["test"]}
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)


# --- metrics and the prior -------------------------------------------------------------------------------


def test_average_precision_operating_points_and_prior(forbid_model_imports):
    refs = [[[0, 0, 10, 10], [50, 50, 60, 60]], [[20, 20, 40, 40]]]
    perfect = [
        [{"box": [0, 0, 10, 10], "score": 0.9}, {"box": [50, 50, 60, 60], "score": 0.8}],
        [{"box": [20, 20, 40, 40], "score": 0.95}],
    ]
    assert average_precision(perfect, refs, 0.5) == 1.0 and average_precision(perfect, refs, 0.95) == 1.0
    ranked = [
        [
            {"box": [0, 0, 10, 10], "score": 0.9},
            {"box": [70, 70, 80, 80], "score": 0.85},
            {"box": [50, 50, 60, 60], "score": 0.3},
        ],
        [{"box": [20, 20, 40, 40], "score": 0.95}],
    ]
    ap = average_precision(ranked, refs, 0.5)
    assert 0.8 < ap < 1.0  # one false positive ranked above the third true positive
    metrics = detection_metrics(ranked, refs, threshold=DETECTION_THRESHOLD)
    assert metrics["n_reference_boxes"] == 3 and metrics["ap50"] == ap and metrics["map"] <= metrics["ap50"]
    assert metrics["recall_at_threshold"] == pytest.approx(2 / 3) and metrics["precision_at_threshold"] == 1.0
    assert (
        metrics["operating_points"]["0.5"]["recall"] == pytest.approx(2 / 3)
        and metrics["operating_points"]["0.5"]["detections"] == 3
    )
    assert metrics["mean_best_iou"] == 1.0 and len(metrics["per_image"]) == 2
    with pytest.raises(ValueError, match="prediction lists"):
        detection_metrics(perfect[:1], refs, threshold=0.9)
    with pytest.raises(ValueError, match="4-value"):
        average_precision([[{"box": [1, 2], "score": 0.5}]], [[[0, 0, 1, 1]]])
    records = _records(8)
    prior = fixed_box_prior(records)
    assert len(prior) == 4 and 0 < prior[0] < prior[2] <= 1 and 0 < prior[1] < prior[3] <= 1
    baseline = prior_baseline(records[:6], records[6:], threshold=0.9)
    assert (
        baseline["n_images"] == 2
        and 0.0 <= baseline["ap50"] <= 1.0
        and "fixed-box prior" in baseline["baseline"]
    )


# --- BYOD ------------------------------------------------------------------------------------------------


def test_byod_directory_and_zip_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records(4)
    folder = tmp_path / "byod"
    folder.mkdir()
    with open(folder / "boxes.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "file", "x_min", "y_min", "x_max", "y_max", "group"]
        )
        writer.writeheader()
        for record in records:
            record["image"].save(folder / f"{record['id']}.jpg")
            for box in record["boxes"]:
                writer.writerow(
                    {
                        "id": record["id"],
                        "file": f"{record['id']}.jpg",
                        "x_min": box[0],
                        "y_min": box[1],
                        "x_max": box[2],
                        "y_max": box[3],
                        "group": record["barcode"],
                    }
                )
    loaded = load_byod_dataset(folder)
    assert (
        [r["id"] for r in loaded] == [r["id"] for r in records]
        and loaded[0]["group"] == "b000"
        and loaded[0]["boxes"] == records[0]["boxes"]
    )
    archive = tmp_path / "byod.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in folder.iterdir():
            zf.write(path, f"inner/{path.name}")
    from_zip = load_byod_dataset(archive)
    assert (
        from_zip[1]["image"].size == records[1]["image"].size and from_zip[1]["boxes"] == records[1]["boxes"]
    )
    assert image_digest(from_zip[1]["image"]) == image_digest(loaded[1]["image"])
    assert not list(tmp_path.glob("inner"))
    exported = write_dataset_csv(records, tmp_path / "out" / "train.csv")
    rows = list(csv.DictReader(exported.read_text(encoding="utf-8").splitlines()))
    assert (
        rows[0]["group"] == "b000"
        and rows[0]["category"] == CLASS_NAME
        and float(rows[0]["x_max"]) == records[0]["boxes"][0][2]
    )
    with pytest.raises(ValueError, match="boxes.csv"):
        load_byod_dataset(tmp_path / "missing")


# --- the DETR set loss on tensors, adapt() guards, artifact refusals --------------------------------------


def test_matcher_and_set_loss_prefer_the_right_query():
    pipe = _pipeline_without_model()
    targets = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]])
    boxes = torch.full((NUM_QUERIES, 4), 0.5)
    boxes[3] = torch.tensor([0.5, 0.5, 0.2, 0.2])
    boxes[7] = torch.tensor([0.2, 0.2, 0.1, 0.1])
    boxes[:, 2:] = boxes[:, 2:].clamp_min(0.05)
    logits = torch.zeros(NUM_QUERIES, 2)
    logits[3, 0] = 3.0
    logits[7, 0] = 3.0
    pairs = pipe._match(torch.softmax(logits, dim=-1), boxes, targets)
    assert sorted(pairs) == [(3, 0), (7, 1)]
    good = float(pipe._loss(logits, boxes, targets))
    bad_logits = logits.clone()
    bad_logits[3, 0], bad_logits[7, 0] = -3.0, -3.0
    assert good < float(pipe._loss(bad_logits, boxes, targets))
    giou = pipe._giou(targets, targets)
    assert torch.allclose(torch.diagonal(giou), torch.ones(2), atol=1e-6) and giou[0, 1] < 0.5
    many = torch.rand(6, 4) * 0.4 + 0.3
    assert (
        len(pipe._match(torch.softmax(torch.zeros(NUM_QUERIES, 2), dim=-1), boxes, many)) == 6
    )  # greedy path


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    with pytest.raises(ValueError, match="head_steps"):
        pipe.adapt(_records(), head_steps=0)
    with pytest.raises(ValueError, match="head_lr"):
        pipe.adapt(_records(), head_lr=2.0)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), epochs=-1)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), lr=1.0)
    with pytest.raises(ValueError, match="trainable_layers"):
        pipe.adapt(_records(), trainable_layers=DECODER_LAYERS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(_records())
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    with pytest.raises(ValueError, match="no adapted heads"):
        pipe.detect_adapted(_records()[0]["image"])


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["bbox_head.layers.0.weight", "head.bias", "head.weight"],
        "adapter": {"classes": [CLASS_NAME], "policy": pl.POLICY_FROZEN},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)
