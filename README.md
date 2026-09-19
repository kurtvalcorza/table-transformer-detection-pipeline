# Table Transformer (DETR-R18) table detection pipeline

DIMER inference wrapper for **Table Transformer — detection** (`microsoft/table-transformer-detection`), the DETR model with a ResNet-18 backbone that Microsoft fine-tuned on PubTables-1M to find tables on document page images, pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline returns pixel-space boxes labelled `table` or `table rotated` with the model's softmax score; it does not read the table's structure or text.

## Upstream alignment

- Model: `microsoft/table-transformer-detection`
- Revision: `2357cbe2b5a5d1c03e54f32764f06058933b65ab`
- Upstream weight license: MIT
- Upstream task: object detection (two classes: `table`, `table rotated`) on document pages
- Repository adaptation: bounded supervised adaptation on `{id, image, boxes}` records (`adapt`) — new single-class detection heads on the frozen DETR decoder features (frozen policy) and an optional unfreeze of the last decoder layers (unfrozen policy) selected against the heads by validation set loss — measured by held-out AP@0.5 / AP@0.75 / mAP (`evaluate`, `evaluate_zero_shot`, `prior_baseline`); `detect` and the PubTables heads are unchanged; the base weights are never redistributed and the adapter is a tutorial output

## Quick start

```python
from PIL import Image
from table_transformer_detection_pipeline import TableTransformerDetectionPipeline, box_iou

pipe = TableTransformerDetectionPipeline.from_pretrained()   # stages + verifies weights/table-transformer-detection first
result = pipe.detect(Image.open("page.png"))
for det in result["detections"]:                             # sorted by score, boxes are [x0, y0, x1, y1] pixels
    print(det["label"], det["box"], round(det["score"], 3))

# the threshold is the Transformers documentation example value; override per call
result = pipe.detect(Image.open("page.png"), threshold=0.5)

# adaptation: the pinned Open Food Facts nutrition-table photographs (fetched and digest-checked at run time), a prior, the zero-shot row, two policies
from table_transformer_detection_pipeline import fetch_sample_dataset, prior_baseline

splits = fetch_sample_dataset()                                            # 72 / 24 / 25 {id, image, boxes} photographs, split by product
print(prior_baseline(splits["train"], splits["test"], threshold=0.9)["ap50"])
print(pipe.evaluate_zero_shot(splits["test"])["ap50"])                       # the PubTables heads, table + table rotated mass as the score
pipe.adapt(splits["train"], splits["validation"], trainable_layers=0)         # frozen policy: new heads on cached decoder features
print(pipe.evaluate(splits["test"])["ap50"])
pipe.adapt(splits["train"], splits["validation"])                           # unfrozen policy: last two decoder layers, selected on validation
print(pipe.adapter["policy"], pipe.evaluate(splits["test"])["ap50"])
print(pipe.detect_adapted(splits["test"][0]["image"], threshold=0.5)["detections"][:1])
pipe.save_artifact("outputs/ttd_adapter")                                   # adapter.safetensors + manifest.json
reloaded = TableTransformerDetectionPipeline.from_artifact("outputs/ttd_adapter")
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed; 2 model-backed tests run when the snapshot is staged). On a fresh clone the manifest is committed but the weights are not: `TableTransformerDetectionPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/table-transformer-detection/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (4 files)
  config.json
  preprocessor_config.json
  model.safetensors          # git-ignored, 115,317,516 bytes
  README.md
weights/off-nutrition/       # git-ignored run-time cache of the 121 pinned Open Food Facts photographs (129 MB)
```

The upstream repository also hosts `pytorch_model.bin`; it is not staged, listed or loaded (DIMER does not accept `.bin` uploads).

## Input ceilings and threshold

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_DETECTIONS = 15` (the checkpoint's `num_queries`), `LABELS = ("table", "table rotated")`; `DETECTION_THRESHOLD = 0.9`. One page image per call; the processor resizes it to 800 px on the shortest edge. See `MODEL_CARD.md` for who owns tuning the threshold and the measured CPU timings.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/table-transformer-detection-pipeline/blob/main/tutorials/table_transformer_detection_colab.ipynb)

`tutorials/table_transformer_detection_colab.ipynb` is declared `E2E` (mode `GUIDED`) under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules (`pipeline.py`, `metrics.py`, `samples.py` — including the 121-row pinned photo table), the model identity, manifest digests and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`; see `tutorials/README.md`). Its default path stages the missing `model.safetensors` with `stage_missing_files(..., allow_download=True)` and digest-verifies the snapshot, fetches the 121 digest-pinned Open Food Facts photographs (129 MB) and draws 72 / 24 / 25 by a seeded split of whole products with `validate_dataset`, `check_split_disjoint` and `split_summary`, runs one test photograph through the inference contract with a `sample-sanity` `evaluation_report` against its reference boxes, scores a fixed-box prior and the untouched checkpoint on the test split (AP@0.5 2.7 % and 9.7 % in the recorded run), trains a new class head and a copy of the box head on the frozen decoder features (37.6 %) and then the last two decoder layers with them, selected by validation set loss (selected `unfrozen last 2 decoder layers + new heads`, 39.8 % / mAP 16.3 %), renders detections before and after, and exports the trained tensors as a safetensors adapter that reloads to identical query scores. Six `outputs/` artifacts are written. BYOD (a `.zip` with `boxes.csv` beside the image files) is optional and gated off by default.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`) — do not constitute clean-runtime notebook evidence. One local fresh-kernel execution is recorded in `docs/release-verification.md` as pre-flight; the supported-runtime run is pending. Complete that record against the exact release revision before calling the notebook release-grade.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output and adaptation contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance, hosting notes and the pinned adaptation corpus.
- `docs/release-verification.md` — the release gate and recorded notebook executions.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are MIT; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
