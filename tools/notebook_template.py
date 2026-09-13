"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "table_transformer_detection_pipeline",
    "repo_name": "table-transformer-detection-pipeline",
    "stem": "table_transformer_detection",
    "notebook_name": "table_transformer_detection_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "TableTransformerDetectionPipeline",
    "weights_key": "table-transformer-detection",
    "runtime_imports": ["torch", "transformers", "timm"],
    "title": "Table Transformer (DETR-R18, PubTables-1M) — DIMER table detection tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/table-transformer-detection-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/table-transformer-detection-pipeline/blob/main/tutorials/table_transformer_detection_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-microsoft%2Ftable--transformer--detection-ffcc4d?style=flat",
            "https://huggingface.co/microsoft/table-transformer-detection",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-microsoft%2Ftable--transformer-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/microsoft/table-transformer",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2110.00061-b31b1b.svg", "https://arxiv.org/abs/2110.00061"),
    ],
    "capability": "table detection on document page images (boxes labelled `table` or `table rotated`) using the pinned `microsoft/table-transformer-detection` weights",
    "intro": (
        "At inference the DETR-style model reads one page image resized to 800 px on its shortest edge, runs a ResNet-18 "
        "backbone and a 6-layer encoder–decoder, and emits exactly 15 query proposals, each a box and a softmax over "
        "`table`, `table rotated` and *no object*; the processor keeps the queries whose class score reaches the "
        "threshold and maps their boxes back to input pixels. **No adaptation occurs:** no training, fine-tuning, "
        "in-context conditioning, or preprocessing fitting happens in this notebook — the upstream checkpoint supplies "
        "the weights and image-processor configuration, and the carried module adds snapshot verification, the input "
        "contract, a fixed output contract and the `box_iou`, `validate_inputs` and `evaluation_report` helpers. The "
        "default sample is a page rendered in code with two ruled tables whose drawn boxes serve as references; its "
        "`box_iou` values are demonstration (plumbing) evidence for one page, not a detection benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, render a synthetic page with reference table boxes (or upload your own page) "
        "and validate it into an input manifest, run the supported task, read the class scores and the caller-owned "
        "threshold correctly, exercise an optional BYOD path, produce an evaluation report that is `sample-sanity` with "
        "`box_iou` only when reference boxes exist and `not-measurable` otherwise, and export machine-readable "
        "detections plus an annotated image and provenance."
    ),
    "exclusions": (
        "table *structure* recognition (rows, columns, cells — the sibling `table-transformer-structure-pipeline` covers "
        "it), OCR or cell text extraction, figure/chart/text-block detection, page rotation correction, mAP or "
        "precision/recall evaluation (which needs a labelled page set), or any training. The model was trained on "
        "PubTables-1M PDF renders; scans, photographs of pages and non-Latin layouts are outside what this notebook measures."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 4.8 s to load and 0.17 s per `detect` on the 850×1100 synthetic page in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 115 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union measures.",
        "- **Data:** the default sample is a deterministic 850×1100 page rendered in code with Pillow's bundled font — a heading, two paragraphs, a wide 8×5 table and a small 5×3 table — so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one page image decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, sides between 16 and 4096 px. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Render the synthetic page or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own reference boxes: a deterministic 850×1100 page is "
                "rendered in code with Pillow's bundled font — a heading, two paragraphs of words, a wide 8×5 ruled table at "
                "`[70, 330, 780, 660]` and a small 5×3 ruled table at `[70, 760, 430, 990]`, every cell holding a short token, "
                "the way a PDF render looks. This is the same page the repository's smoke run used. The drawn boxes are the "
                "reference for the `box_iou` sanity check later; they are not a labelled dataset, so nothing here is a "
                "precision/recall measurement. The image digest is printed for the record. BYOD is optional and disabled by "
                "default; when enabled, upload one page image — no reference boxes exist for it, so the evaluation report will "
                "be `not-measurable`.\n\n"
                "The detection threshold is a **caller-owned request parameter**, not a pipeline constant: a query survives when "
                "its softmax score for `table` or `table rotated` reaches it. The package default (`DETECTION_THRESHOLD = 0.9`) "
                "follows the Transformers documentation example, not a calibration; it is exposed here as a form parameter and "
                "passed explicitly on every call. Nothing is validated in this cell — the next section hands the image and the "
                "threshold to the pipeline's own validation stage, which is the only checker. Look for a dictionary naming the "
                "sample kind, the image size and digest, the threshold, and the drawn reference boxes."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "threshold = 0.9  # @param {{type:\"number\"}}\n\n\n"
                "def synthetic_page(width=850, height=1100):\n"
                "    \"\"\"A letter-size page rendered with Pillow's bundled font: heading, two paragraphs, two ruled tables.\"\"\"\n"
                "    page = Image.new('RGB', (width, height), 'white')\n"
                "    d = ImageDraw.Draw(page)\n"
                "    body, head = ImageFont.load_default(size=15), ImageFont.load_default(size=22)\n"
                "    words = 'quarterly revenue by region and product line for the fiscal year with notes on methodology'.split()\n"
                "    d.text((70, 50), 'Annual Report: Regional Results', fill='black', font=head)\n"
                "    y = 95\n"
                "    for _para in range(2):\n"
                "        for line in range(6):\n"
                "            text = ' '.join(words[(line * 3 + k) % len(words)] for k in range(11 - (line % 3)))\n"
                "            d.text((70, y), text, fill=(40, 40, 40), font=body)\n"
                "            y += 20\n"
                "        y += 16\n"
                "    boxes = []\n"
                "    for (x0, y0, x1, y1, rows, cols, first) in ((70, 330, 780, 660, 8, 5, 'Region'), (70, 760, 430, 990, 5, 3, 'Item')):\n"
                "        d.rectangle([x0, y0, x1, y1], outline='black', width=2)\n"
                "        rh, cw = (y1 - y0) / rows, (x1 - x0) / cols\n"
                "        d.line([(x0, y0 + rh), (x1, y0 + rh)], fill='black', width=2)\n"
                "        for r in range(2, rows):\n"
                "            d.line([(x0, y0 + rh * r), (x1, y0 + rh * r)], fill=(120, 120, 120), width=1)\n"
                "        for c in range(1, cols):\n"
                "            d.line([(x0 + cw * c, y0), (x0 + cw * c, y1)], fill=(120, 120, 120), width=1)\n"
                "        for r in range(rows):\n"
                "            for c in range(cols):\n"
                "                token = (first if c == 0 else f'Q{{c}}') if r == 0 else (f'North {{r}}' if c == 0 else f'{{(r * 7 + c * 13) % 97 + 1}},{{(r * 31 + c) % 900 + 100:03d}}')\n"
                "                d.text((x0 + cw * c + 8, y0 + rh * r + rh / 2 - 8), token, fill='black', font=body)\n"
                "        boxes.append([float(x0), float(y0), float(x1), float(y1)])\n"
                "    d.text((70, 1010), 'Table 2 summarises the line items; see the appendix for the full breakdown.', fill=(40, 40, 40), font=body)\n"
                "    return page, boxes\n\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    drawn_boxes = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic synthetic page: no randomness, so no seed is needed and the digest is stable per Pillow build.\n"
                "    image, drawn_boxes = synthetic_page()\n"
                "    image_name = 'synthetic_page_850x1100.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'threshold': threshold, 'drawn_boxes': drawn_boxes}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `detect` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px and a threshold in `[0, 1]` — and returns an "
                "**input manifest** naming the schema (including the two labels and the 15-query ceiling on detections), the "
                "input's observed mode and size, the threshold, and the verdict. The manifest is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a threshold "
                "outside `[0, 1]` and records the pipeline's own error message as a finding. Inside the pipeline the image is "
                "converted to RGB and resized to 800 px by the processor; boxes are mapped back to input pixels, and nothing else "
                "is dropped or altered."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_DETECTIONS': MAX_DETECTIONS, 'LABELS': list(LABELS), 'DETECTION_THRESHOLD': DETECTION_THRESHOLD}}}})\n"
                "input_manifest = validate_inputs(image, threshold=threshold, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, threshold=1.5)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'out-of-range-threshold-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Detect and read the scores correctly\n\n"
                "`detect` returns a dict with `detections` — a list of `{{box, label, score}}` **ordered by descending score**, "
                "`box` in xyxy pixel coordinates of the input, `label` one of `table` / `table rotated` — plus the threshold "
                "used, `width`, `height` and the model identity. At most 15 boxes can ever be returned (the DETR decoder has 15 "
                "queries). Each `score` is the query's **softmax class probability under the model's own head, not a calibrated "
                "estimate for your pages**: it was never fitted to the frequency with which a box is a real table on your "
                "documents. The threshold you passed is the only decision rule; the pipeline ships 0.9 as a default, not as a "
                "calibration, and the caller owns it per deployment — lower it when a missed table costs more than a spurious box, "
                "raise it when a false table triggers downstream extraction. Inference is deterministic on a fixed device and dtype "
                "(no sampling, `torch.inference_mode`); CUDA kernel selection can move scores in the third or fourth decimal place. "
                "As recorded in the model card, the repository's CPU smoke on this same page at threshold 0.9 returned two `table` "
                "boxes at scores 0.999 and 0.997 with `box_iou` 0.80 and 0.66 against the drawn rectangles (the model's boxes hug "
                "the ruled area more tightly than the drawn outline); that is one observation, not a calibration point."
            ),
            "code": (
                "result = pipe.detect(image, threshold=threshold)\n"
                "print({{'n_detections': len(result['detections']), 'threshold': result['threshold'], 'device': pipe.device}})\n"
                "for rank, det in enumerate(result['detections'], start=1):\n"
                "    print(f\"{{rank:>2}}. score {{det['score']:.4f}}  label {{det['label']!r}}  box {{[round(v, 1) for v in det['box']]}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No detection metric "
                "is reported by default: mean average precision needs a labelled page set, and this repository ships none. The "
                "repository's only metric helper is `box_iou(a, b)` (intersection-over-union of two xyxy boxes), the building "
                "block a caller would use to compute mAP on their own labelled pages; when reference boxes are supplied the report "
                "carries one `box_iou` entry per reference — its value and which detection matched it best — with the verdict "
                "`sample-sanity`. On the synthetic path those references are tables **you rendered yourself**, so a high IoU "
                "proves only that the input contract, forward pass and coordinate mapping round-trip. On BYOD no reference exists, "
                "the verdict is `not-measurable`, and the report states what would make the task measurable: labelled table boxes "
                "on your own pages. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, drawn_boxes, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference boxes exist for this input, so box_iou is not computed; inspect the annotated PNG instead.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the full result (score-ordered detections with boxes and labels, the threshold), "
                "the evaluation report, the input manifest, the sample identity, digest and drawn boxes, the notebook's source "
                "(repository, revision, embedded module digest, generator), the model identifier, the immutable model revision, "
                "the model licence, and the runtime identity (Python, `torch`, `transformers`, `timm`, device). The detections "
                "are also written as CSV with explicit `image`, `rank`, `label`, `score`, `x0`, `y0`, `x1`, `y1` columns so score "
                "ordering survives downstream use, and an annotated PNG draws every returned box for visual inspection (a "
                "supplement to, not a replacement for, the machine-readable files). No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "annotated = image.convert('RGB').copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for det in result['detections']:\n"
                "    draw.rectangle(det['box'], outline=(0, 160, 0), width=3)\n"
                "    draw.text((det['box'][0] + 4, det['box'][1] + 4), f\"{{det['label']}} {{det['score']:.3f}}\", fill=(0, 160, 0))\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'prediction': result,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'drawn_boxes': drawn_boxes}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'timm': timm.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_detections.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1'])\n"
                "    for rank, det in enumerate(result['detections'], start=1):\n"
                "        writer.writerow([image_name, rank, det['label'], f\"{{det['score']:.6f}}\", *[f\"{{v:.2f}}\" for v in det['box']]])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The boxes locate regions the model classifies as a table on a page image; the label is one of two classes and the "
        "softmax score is not calibrated for your documents. The threshold is a request parameter you own; the default is the "
        "documentation example, not a tuned operating point. On the synthetic page the `box_iou` values in the evaluation report "
        "compare detections to tables you rendered yourself and the verdict is `sample-sanity`, which proves only that the input "
        "contract, forward pass and coordinate mapping work; they say nothing about scans, photographed pages, borderless "
        "tables, multi-column layouts, or non-Latin documents, and a BYOD result is a single-page observation with the verdict "
        "`not-measurable`. Everything is resized to 800 px on the shortest edge, so tables that are tiny at that scale may be "
        "missed. The pipeline provides no structure recognition, OCR, rotation correction, mAP evaluation, or training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** lower `threshold` to 0.5 and count how many extra boxes appear on the same page (the smoke run "
        "found none, but a scan behaves differently); remove the ruling lines from one table in `synthetic_page` and see whether "
        "the score survives on text alignment alone; enable `USE_BYOD` with a scanned page, hand-label its table boxes and pass "
        "them to `evaluation_report` to see the verdict switch to `sample-sanity` — the first step towards a real precision/recall "
        "number; then feed a detected crop to the sibling structure-recognition pipeline.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/table-transformer-detection-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/table-transformer-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/table-transformer-detection-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/microsoft/table-transformer\n"
        "- PubTables-1M: Towards Comprehensive Table Extraction From Unstructured Documents (Smock, Pesala, Abraham, 2021): https://arxiv.org/abs/2110.00061"
    ),
}
