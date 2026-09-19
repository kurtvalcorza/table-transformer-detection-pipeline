"""Detection metrics for the adaptation contract, implemented here with no external scorer.

Boxes are ``[x_min, y_min, x_max, y_max]`` in pixels of the image they belong to. A prediction is a list of
``{"box", "score"}`` entries per image (every DETR query, not just those above a threshold — average
precision ranks them all); the reference is the list of ground-truth boxes per image.

- **AP@t** — average precision at IoU threshold *t*: predictions of all images are ranked by score, each is a
  true positive when it overlaps a not-yet-matched reference box of its image with IoU ≥ *t* (greedy in score
  order), the precision-recall curve is made monotone from the right and the area under it is summed over the
  recall steps (the VOC 2010+ / COCO "all-points" convention, one class).
- **mAP** — the mean of AP@t over t = 0.50, 0.55, …, 0.95 (the COCO primary metric, one class).
- **operating-point recall / precision** — at the pipeline's `DETECTION_THRESHOLD`, how many reference boxes
  are found (IoU ≥ 0.5) and how many surviving detections are true positives.
- **mean best IoU** — for every reference box, the IoU of its best-overlapping prediction, averaged.

The **fixed-box prior** frames the numbers: one box per image at the mean normalised reference box of the
training split, with a constant score — what "the table is usually here" alone buys.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .pipeline import box_iou

IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
MATCH_IOU = 0.5

METRIC_DEFINITIONS = {
    "ap50": "average precision at IoU 0.50 over all ranked predictions of all images (one class)",
    "ap75": "average precision at IoU 0.75",
    "map": "mean of the average precision at IoU 0.50, 0.55, ..., 0.95 (the COCO primary metric, one class)",
    "recall_at_threshold": (
        "fraction of reference boxes matched (IoU >= 0.5) by a detection at or above the operating threshold"
    ),
    "precision_at_threshold": (
        "fraction of detections at or above the operating threshold that match a reference box"
    ),
    "mean_best_iou": "mean over reference boxes of the IoU of their best-overlapping prediction, any score",
    "operating_points": "recall / precision / surviving detections at the operating threshold and at 0.5",
}


def _check_pairs(
    predictions: Sequence[Sequence[Mapping[str, Any]]], references: Sequence[Sequence[Sequence[float]]]
) -> None:
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} prediction lists for {len(references)} reference lists")
    if not references:
        raise ValueError("at least one image is required")
    for preds in predictions:
        for p in preds:
            if "box" not in p or "score" not in p or len(p["box"]) != 4:
                raise ValueError("each prediction needs a 4-value 'box' and a 'score'")


def average_precision(
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    references: Sequence[Sequence[Sequence[float]]],
    iou_threshold: float = MATCH_IOU,
) -> float:
    """All-points average precision for one class over a list of images."""
    _check_pairs(predictions, references)
    n_ref = sum(len(r) for r in references)
    if n_ref == 0:
        raise ValueError("at least one reference box is required")
    ranked = sorted(
        (
            (float(p["score"]), i, [float(v) for v in p["box"]])
            for i, preds in enumerate(predictions)
            for p in preds
        ),
        key=lambda t: -t[0],
    )
    matched = [np.zeros(len(r), dtype=bool) for r in references]
    tp = np.zeros(len(ranked))
    for k, (_score, i, box) in enumerate(ranked):
        best, best_j = 0.0, -1
        for j, ref in enumerate(references[i]):
            if matched[i][j]:
                continue
            iou = box_iou(box, ref)
            if iou > best:
                best, best_j = iou, j
        if best >= iou_threshold and best_j >= 0:
            matched[i][best_j] = True
            tp[k] = 1.0
    if not ranked:
        return 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    recall = cum_tp / n_ref
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    # monotone envelope from the right, area under the stepwise curve
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate([[0.0], recall])
    return float(np.sum((recall[1:] - recall[:-1]) * precision))


def _operating_point(
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    references: Sequence[Sequence[Sequence[float]]],
    threshold: float,
) -> dict[str, Any]:
    """Recall / precision of the detections at or above `threshold` (greedy IoU >= MATCH_IOU matching)."""
    n_ref = sum(len(r) for r in references)
    found = kept = true_kept = 0
    per_image = []
    for preds, refs in zip(predictions, references, strict=True):
        surviving = [p for p in preds if float(p["score"]) >= threshold]
        kept += len(surviving)
        used: set[int] = set()
        image_found = 0
        for p in sorted(surviving, key=lambda q: -float(q["score"])):
            best, best_j = 0.0, -1
            for j, ref in enumerate(refs):
                if j in used:
                    continue
                iou = box_iou(p["box"], ref)
                if iou > best:
                    best, best_j = iou, j
            if best >= MATCH_IOU and best_j >= 0:
                used.add(best_j)
                image_found += 1
                true_kept += 1
        found += image_found
        per_image.append(
            {"n_reference": len(refs), "n_detections_at_threshold": len(surviving), "found": image_found}
        )
    return {
        "threshold": float(threshold),
        "recall": found / n_ref if n_ref else 0.0,
        "precision": true_kept / kept if kept else 0.0,
        "detections": kept,
        "per_image": per_image,
    }


def detection_metrics(
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    references: Sequence[Sequence[Sequence[float]]],
    *,
    threshold: float,
) -> dict[str, Any]:
    """AP@0.5 / AP@0.75 / mAP over all ranked predictions, plus the operating point at `threshold`."""
    _check_pairs(predictions, references)
    aps = {t: average_precision(predictions, references, t) for t in IOU_THRESHOLDS}
    n_ref = sum(len(r) for r in references)
    points = {t_: _operating_point(predictions, references, t_) for t_ in sorted({0.5, float(threshold)})}
    primary = points[float(threshold)]
    best_ious = []
    per_image = []
    for preds, refs, image_point in zip(predictions, references, primary["per_image"], strict=True):
        image_best = [max((box_iou(p["box"], ref) for p in preds), default=0.0) for ref in refs]
        best_ious.extend(image_best)
        per_image.append({**image_point, "best_iou": [round(v, 4) for v in image_best]})
    return {
        "n_images": len(references),
        "n_reference_boxes": n_ref,
        "ap50": aps[0.5],
        "ap75": aps[0.75],
        "map": float(np.mean(list(aps.values()))),
        "ap_by_iou": {str(t): v for t, v in aps.items()},
        "threshold": float(threshold),
        "recall_at_threshold": primary["recall"],
        "precision_at_threshold": primary["precision"],
        "detections_at_threshold": primary["detections"],
        "operating_points": {
            str(t_): {k: v for k, v in point.items() if k != "per_image"} for t_, point in points.items()
        },
        "mean_best_iou": float(np.mean(best_ious)) if best_ious else 0.0,
        "per_image": per_image,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def fixed_box_prior(train_records: Sequence[Mapping[str, Any]]) -> list[float]:
    """The mean normalised reference box of the training split (x_min, y_min, x_max, y_max in 0..1)."""
    rows = []
    for record in train_records:
        w, h = record["image"].size
        for box in record["boxes"]:
            rows.append([box[0] / w, box[1] / h, box[2] / w, box[3] / h])
    if not rows:
        raise ValueError("the training split has no reference boxes")
    return [float(v) for v in np.mean(np.asarray(rows, dtype=np.float64), axis=0)]


def prior_baseline(
    train_records: Sequence[Mapping[str, Any]], test_records: Sequence[Mapping[str, Any]], *, threshold: float
) -> dict[str, Any]:
    """One box per test image at the training split's mean normalised box, scored 1.0."""
    prior = fixed_box_prior(train_records)
    predictions = []
    for record in test_records:
        w, h = record["image"].size
        predictions.append([{"box": [prior[0] * w, prior[1] * h, prior[2] * w, prior[3] * h], "score": 1.0}])
    out = detection_metrics(predictions, [r["boxes"] for r in test_records], threshold=threshold)
    out["baseline"] = (
        "one box per image at the training split's mean normalised reference box (fixed-box prior)"
    )
    out["prior_box_normalised"] = prior
    return out
