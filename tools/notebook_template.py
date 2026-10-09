"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E supervised-adaptation workflow: the pinned Table Transformer detection snapshot
is digest-verified and loaded, a digest-pinned real box-labelled photo corpus (Open Food Facts nutrition-table
photographs, a domain the PubTables detector was not trained on) is fetched, validated and split by product,
the inference contract is exercised on a real photograph with reference boxes (the first `sample-sanity`
report of this repository), the untouched checkpoint and a fixed-box prior are scored on the test split, new
heads are trained on the frozen DETR decoder features (the frozen policy) and then the last decoder layers
with them (the unfrozen policy), the policy is selected on validation, the held-out split is scored by AP, and
the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "table_transformer_detection_pipeline",
    "repo_name": "table-transformer-detection-pipeline",
    "stem": "table_transformer_detection",
    "notebook_name": "table_transformer_detection_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline): managed CPython, a size- and
    # SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    # TTD-m2: the default path also reaches static.openfoodfacts.org for the photographs, so the generated access bullet
    # names both hosts instead of "the Hugging Face Hub only".
    "external_access": (
        "the Hugging Face Hub, to fetch the pinned `{MODEL_ID}` snapshot (~{total_mb:.0f} MB in total) at revision "
        "`{MODEL_REVISION:.12}…`, and `static.openfoodfacts.org`, for the 121 pinned photographs named in the data bullet "
        "above. No credentials are required; nothing is installed from this repository."
    ),
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned Table Transformer detection snapshot (safetensors, 115 MB), fetches 121 digest-pinned Open Food Facts product "
        "photographs with their nutrition-table boxes (129 MB, no credential), validates them and draws 72 / 24 / 25 training, "
        "validation and test photographs by a seeded split of whole products, runs one test photograph through the inference "
        "contract with an input manifest, a rejection probe and a `sample-sanity` evaluation report against its reference "
        "boxes, scores a fixed-box prior and the untouched checkpoint on the test split (the **zero-shot row**), trains a new "
        "class head and a copy of the box head on the frozen DETR decoder features (the **frozen policy**) and then the last "
        "two decoder layers with them (the **unfrozen policy**), selects between the two by validation DETR loss, scores the "
        "held-out split by AP@0.5 / AP@0.75 / mAP with the selected model, renders detections before and after, exports the "
        "trained tensors as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify parity. "
        "The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no "
        "configuration edit (NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about 4 minutes of model time "
        "after the downloads; a CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to a zip or folder "
        "in the runtime (Colab, Kaggle or Jupyter) or leave it empty to upload one zip in Colab, then choose **Run after** from "
        "that cell (it first puts the pipeline back to the pinned base) to supply your own box-labelled images: at least **12 "
        "photographs in 12 groups** (or fewer groups with several photographs each — the training split needs 8 photographs "
        "after 20 % + 20 % of the groups are held out), photographs stored upright (an EXIF-rotated photograph is refused), as a "
        "`.zip` or folder holding `boxes.csv` (columns `id`, `file`, `group`, `x_min`, `y_min`, `x_max`, `y_max`; "
        "one row per box, pixel coordinates; `group` — the product, session or device — must be non-empty on every row, and "
        "rows of one `id` must agree on `file` and `group`, or the loader refuses the set) beside the image files — images are "
        "decoded from the archive, never extracted to disk. They pass through the same validation, seeded group-disjoint split, prior, zero-shot scoring, "
        "frozen-policy heads, unfrozen-policy training and selection, held-out evaluation, detection rendering, artifact "
        "export and reload-parity cells as the Open Food Facts sample. The expected schema and the ceilings are stated in the "
        "Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and never part of the "
        "default path."
    ),
    "pipeline_class": "TableTransformerDetectionPipeline",
    "weights_key": "table-transformer-detection",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "transformers", "timm"],
    "title": "Table Transformer (DETR-R18, PubTables-1M) — DIMER E2E supervised adaptation tutorial: nutrition-table detection on product photographs, new heads vs bounded decoder unfreeze (standalone)",
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
    "capability": "table detection on document page images (boxes labelled `table` or `table rotated`) and bounded supervised adaptation to a new box class on a new image domain — new detection heads on the frozen DETR decoder features with an optional unfreeze of the last decoder layers — measured by held-out AP@0.5 / AP@0.75 / mAP, using the pinned `microsoft/table-transformer-detection` weights",
    "intro": (
        "At inference the DETR-style model reads one image resized to 800 px on its shortest edge, runs a ResNet-18 "
        "backbone and a 6-layer encoder–decoder, and emits exactly 15 query proposals, each a box and a softmax over "
        "`table`, `table rotated` and *no object*; the processor keeps the queries whose class score reaches the "
        "threshold and maps their boxes back to input pixels. The carried pipeline module adds snapshot verification, the "
        "input contract, a fixed output contract and the `box_iou`, `validate_inputs` and `evaluation_report` helpers; "
        "`evaluation_report` becomes `sample-sanity` only when a caller supplies reference boxes — which this notebook, "
        "unlike its inference-only predecessor, does, on a real photograph.\n\n"
        "What this notebook adds to inference is **supervised adaptation to a new box class on a new image domain, under "
        "an explicit frozen-vs-unfrozen policy**. The dataset is real and far from PubTables-1M's PDF renders: 121 product "
        "photographs from the Open Food Facts nutrition-table detection set (version 1.1, manually reviewed; images CC BY-SA "
        "3.0), each with one to three boxes around the printed nutrition table, pinned per file by byte size and SHA-256 "
        "of the served original, fetched at run time, refused on any mismatch and downscaled to 1,280 px. Two products "
        "have two photographs, so the sample is split by **product**, never by photograph. The carried `metrics.py` scores "
        "a prediction set by **AP@0.5**, **AP@0.75** and **mAP** (the COCO convention, one class, over every query of every "
        "image) plus the recall and precision at the pipeline's operating threshold and at 0.5; a **fixed-box prior** (the "
        "training split's mean box) and the **untouched checkpoint** (its `table` + `table rotated` mass as the score) "
        "frame the numbers. The **frozen policy** trains a new two-way class head and a copy of the box head on the frozen "
        "decoder features under the DETR set loss (Hungarian matching, cross-entropy with a 0.1 no-object weight, L1 and "
        "GIoU — implemented in the carried module, no external matcher); the **unfrozen policy** continues by training the "
        "last decoder layers with them end to end, and the epoch with the lowest validation loss — which may be the heads "
        "alone — is kept. The PubTables heads and `detect` are never trained or exported. The adaptation question is "
        "whether unfreezing the decoder buys anything over new heads on 72 photographs. Nothing here is a quality claim "
        "about your images: it is one seeded split of one small corpus."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter, and wants to see how a pretrained detector is adapted to a new class and domain with a small box-labelled set — new heads on frozen features versus a bounded unfreeze — and how to tell from held-out numbers which one helped, without fooling themselves. No prior experience with DETR or fine-tuning is assumed; each term is explained where it first matters and again in the **Glossary** at the end. CPU is adequate (about four minutes of model time); a GPU is faster.\n\n**Input → Model → Output.**\n\n| | Detection (PubTables heads) | Adapted detection | Bounded adaptation |\n|---|---|---|---|\n| Input | one image, sides 16..4,096 px, and a score threshold | one product photograph | box-labelled photographs (72 training and 24 validation in the sample), split by product |\n| Model | Table Transformer: ResNet-18 backbone, DETR encoder-decoder with 15 queries, PubTables heads (`table`, `table rotated`) | the same decoder with a new two-way class head and a copied box head | frozen policy: the new heads on frozen decoder features; unfrozen policy: also the last two decoder layers; validation loss chooses |\n| Output | boxes with a label and a softmax score — not calibrated | `nutrition-table` boxes with scores | a safetensors adapter, and held-out AP@0.5 / AP@0.75 / mAP beside a prior and the zero-shot row |\n\n**How to use this notebook.** Choose a runtime (CPU works; a GPU is faster), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package (including the pinned photograph table) and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer that names the run it quotes — the Kaggle T4 release run of 19 September 2026 and the CPU build record, which disagree in an instructive way. Section 10 is a **change-one-thing experiment**, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the corpus, validation and a product-level split *(evaluation practice)* → 5 the inference contract with reference boxes *(core concept: what a detector returns)* → 6 the prior, the zero-shot checkpoint and the frozen policy *(evaluation practice)* → 7 the unfrozen policy and validation selection *(core concept)* → 8 held-out evaluation → 9 detections before and after, export and reload *(engineering)* → 10 change one thing (optional) → conclude."
            )
        ]
    },
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned real box-labelled corpus and validate and "
        "split it by product without leakage; run a real photograph through the public detection API and read a "
        "`sample-sanity` report against reference boxes; read AP@0.5 / AP@0.75 / mAP beside a fixed-box prior and the "
        "zero-shot checkpoint and understand why a detector's score threshold is an operating point, not part of AP; train "
        "new detection heads on frozen features and a bounded decoder unfreeze with explicit hyperparameters and "
        "validation-based selection between the two policies; evaluate on an independent product-disjoint test split; "
        "compare detections before and after; and export a safetensors adapter (heads plus any trained decoder layers) "
        "that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "table *structure* recognition (rows, columns, cells — the sibling `table-transformer-structure-pipeline` covers "
        "it), OCR or nutrient text extraction, multi-class detection, backbone or encoder training, data augmentation, "
        "any training of the PubTables heads, any PubTables-1M accuracy claim, and any claim that 121 product photographs "
        "from one site stand in for your images. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with DETR or fine-tuning. The notebook explains DETR queries, the set loss with Hungarian matching and GIoU, AP@0.5 / AP@0.75 / mAP, operating points, the frozen and unfrozen policies and the adapter where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter and nothing is installed into it. The default path runs on CPU and uses CUDA automatically when available; float32 on both. The build record measured about 0.3 s per photograph to run the decoder on CPU (3 s for the 25-photograph zero-shot pass), about 20 s for the new heads including feature extraction, and about 30 s per unfreeze epoch over 72 photographs plus a 24-photograph validation pass. The pinned `torch==2.14.0` install and the 115 MB checkpoint are the large downloads of the run, then the 129 MB of photographs.",
        "- **Knowledge:** basic Python and PIL; what a bounding box in xyxy pixel coordinates is; what intersection-over-union and average precision measure and why AP does not depend on a score threshold; what a Hungarian (one-to-one) matching between predictions and references is; what validation-based selection between two policies means.",
        "- **Data contract:** records are `{id, image, boxes}` — a PIL image (or a path to one) with sides 16..4,096 px, a list of 1..15 `[x_min, y_min, x_max, y_max]` pixel boxes inside the image with sides of at least 4 px, ids matching `[A-Za-z0-9_.:-]{1,64}` and unique; a training set needs 8..2,000 records, so with the default 20 % + 20 % group hold-out the effective BYOD minimum is **12 photographs in 12 groups** (`min_byod_records()` computes it); a photograph with an EXIF orientation tag is refused, since its boxes would not line up with the pixels; images are de-duplicated by decoded-pixel digest and split by `group` / `barcode` so one product never straddles splits. BYOD accepts a `.zip` (or a directory) holding `boxes.csv` and the image files, and requires a non-empty `group` on every row — the notebook's automatic split is group-disjoint only because the loader refuses ungrouped rows (`load_byod_dataset(..., require_group=False)` is the explicit opt-out, without that guarantee).",
        "- **Validation is structural, not semantic:** nothing checks that a box is around a table — a mislabelled set is trained on without complaint; all boxes are one class, and the four Open Food Facts nutrition-table categories are merged into it (kept under `categories` for provenance).",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — product photographs can show hands, homes and receipts. The default path uploads nothing.",
        "- **External access (data):** besides the Hub, the default path fetches 121 pinned objects (`<barcode path>/<image id>.jpg`, 129,318,620 bytes in total, one SHA-256 each in the carried `SAMPLE_RECORDS` table) from `static.openfoodfacts.org` over HTTPS, each refused on any byte-size or SHA-256 mismatch before it is decoded; the images are CC BY-SA 3.0 (Open Food Facts contributors) and the boxes come from the Open Food Facts nutrition-table detection dataset (ODbL), credited in the References.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and product-level split\n\n"
                "`fetch_corpus` downloads the 121 pinned photographs (or reads them from the cache), refuses a byte-size "
                "or SHA-256 mismatch per file before it is decoded, and `read_corpus` turns each into a `{{id, image, "
                "boxes}}` record — the served original downscaled to a longest side of 1,280 px, its normalised boxes "
                "rendered to pixels, one sliver annotation (1.5 × 4 px) dropped by the contract's 4 px rule — with its "
                "barcode, categories and image URL. `build_sample_dataset` draws 71 / 24 / 24 whole **products** by a "
                "seeded shuffle (72 / 24 / 25 photographs); `validate_dataset` then checks every record against the "
                "contract, `check_split_disjoint` asserts no photograph (by decoded-pixel digest) and no product appears "
                "in two splits, `split_summary` reports photographs, boxes and products per split, and the training boxes "
                "table is written to `outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 121 photographs, splits 72 / 24 / 25 with 76 / 25 / 26 boxes, three digests, and four refusal "
                "probes — a duplicate id, a box outside its image, an oversized image and a record with no boxes — each "
                "rejected before `torch` does anything. About a minute on the first run for the downloads.\n\n"
                "*Evaluation practice.* **Bring your own data (optional):** set `USE_BYOD = True` and either `BYOD_PATH` (a zip or "
                "a folder holding `boxes.csv` and the images, as a path in this runtime — this works on Colab, Kaggle and "
                "Jupyter) or leave `BYOD_PATH` empty to upload exactly one zip through the Colab dialog; then choose **Run after** "
                "from this cell. This cell first puts the pipeline back to the pinned base, so the zero-shot row and both policies "
                "start from the untouched checkpoint. The effective minimum is 12 photographs in 12 groups.\n\n"
                "**Predict before running:** the split is made by *product* (barcode), not by photograph. Why does that matter "
                "when one product can have several photographs?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import time\n\n"
                "from PIL import ImageDraw\n\n"
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                'SPLIT_SEED = 42  # @param {{type:"integer"}}\n\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                "# A re-run after Sections 6-7 (BYOD, or a new split): the zero-shot row and both policies must start from the pinned base.\n"
                "had_adapter = pipe.adapter is not None\n"
                "restored_layers = pipe.restore_base()\n"
                "if had_adapter or restored_layers:\n"
                "    print({{'restored_pinned_base': len(restored_layers), 'note': 'the adapted heads and decoder layers were removed; Sections 5-7 start from the checkpoint again'}})\n"
                "t0 = time.perf_counter()\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_path = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_path.exists():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding boxes.csv and the images.')\n"
                "        file_name = byod_path.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                "        uploaded = files.upload() or {{}}\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        if not file_name.lower().endswith('.zip'):\n"
                "            raise ValueError(f'{{file_name}}: upload one .zip holding boxes.csv and the images.')\n"
                "        byod_path = Path('work') / 'byod.zip'\n"
                "        byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_count = {{'byod': len(records), 'duplicate_images_dropped': len(records) - sum(len(part) for part in splits.values()), 'effective_minimum': min_byod_records()['total']}}\n"
                "    if len(splits['test']) < 10:\n"
                "        print({{'caution': f\"only {{len(splits['test'])}} held-out test photographs: AP moves in large steps and carries no dispersion estimate; add photographs before reading it\"}})\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/off-nutrition'))\n"
                "    raw_count = {{'photographs': len(corpus), 'products': len({{r['barcode'] for r in corpus}}), 'boxes': sum(len(r['boxes']) for r in corpus), 'dropped_slivers': sum(r['dropped_boxes'] for r in corpus)}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "fetch_seconds = round(time.perf_counter() - t0, 1)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "# The training split must hold MIN_RECORDS; validation and test only need one photograph each (split_dataset checks that).\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "summary = split_summary(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw': raw_count, 'splits': disjoint, 'split_summary': summary, 'fetch_seconds': fetch_seconds, 'corpus_bytes': CORPUS_BYTES}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'boxes': manifest['n_boxes'], 'boxes_per_image': manifest['boxes_per_image'], 'image_side': manifest['image_side'], 'box_area_fraction': manifest['box_area_fraction'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{k: example[k] for k in ('id', 'barcode', 'categories', 'image_url', 'source_size') if k in example}}, 'size': example['image'].size, 'boxes': [[round(v, 1) for v in b] for b in example['boxes']]}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'box outside image': [{{**train_records[0], 'boxes': [[0, 0, train_records[0]['image'].width + 5, 50]]}}, *train_records[1:8]],\n"
                "    'oversized image': [{{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8))}}, *train_records[1:8]],\n"
                "    'no boxes': [{{**train_records[0], 'boxes': []}}, *train_records[1:8]],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                '**What to notice:** 121 photographs, 72 / 24 / 25 photographs in 71 / 24 / 24 products, the three digests and the four refusals.\n\n<details><summary>Check your reasoning</summary>Photographs of one product share a package, a label layout and usually a photographer and a kitchen table. If they were split by photograph, the test set would contain near-copies of training images and the score would measure recognition of the same packages, not detection of nutrition tables. Splitting by barcode keeps every product on one side; a review probe confirmed no product is shared between the splits.</details>'
            ),
        },
        {
            "md": (
                "## 5. Detect through the inference contract, with reference boxes\n\n"
                "Before any adaptation, the inference contract is exercised as it always was, on one test photograph. "
                "`validate_inputs` applies exactly the checks `detect` applies — image type, sides `MIN_IMAGE_SIDE`.."
                "`MAX_IMAGE_SIDE` px, a threshold in `[0, 1]` — and returns an input manifest; a deliberately invalid "
                "threshold is validated too and its rejection recorded as a finding. `detect` returns the PubTables "
                "queries at or above `DETECTION_THRESHOLD` with their `table` / `table rotated` labels — on a product "
                "photograph expect few or none: the checkpoint was trained on PDF page renders. Because this photograph "
                "carries reference boxes, `evaluation_report` can for the first time return **`sample-sanity`**: one "
                "`box_iou` entry per reference, the best-overlapping detection at a low threshold (0.05, so the report has "
                "something to score; the build record's probe photograph scored IoU 0.47 with one `table` detection at 0.05). A rendered preview "
                "(reference boxes in green, detections in red) is displayed. That domain gap, not a quality defect, is what "
                "the rest of the notebook adapts around.\n\n"
                "*Core concept.* A DETR detector answers with a fixed set of query slots (15 here), each a box with a class "
                "distribution; the threshold decides which slots you see. The score is a softmax over the checkpoint's own "
                "classes, not a calibrated confidence.\n\n"
                "**Predict before running:** the checkpoint was trained on PDF page renders. How many `table` detections will it "
                "make on a product photograph at the default threshold of 0.9?"
            ),
            "code": (
                "def draw_boxes(image, reference, detections, width=4):\n"
                "    canvas = image.convert('RGB').copy()\n"
                "    pen = ImageDraw.Draw(canvas)\n"
                "    for box in reference:\n"
                "        pen.rectangle([round(v) for v in box], outline=(0, 200, 0), width=width)\n"
                "    for det in detections:\n"
                "        pen.rectangle([round(v) for v in det['box']], outline=(230, 30, 30), width=width)\n"
                "        pen.text((det['box'][0] + 4, det['box'][1] + 4), f\"{{det['label']}} {{det['score']:.2f}}\", fill=(230, 30, 30))\n"
                "    return canvas\n\n"
                "probe_record = test_records[0]\n"
                "image = probe_record['image']\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_DETECTIONS': MAX_DETECTIONS, 'LABELS': list(LABELS), 'DETECTION_THRESHOLD': DETECTION_THRESHOLD}}, 'contract': {{'NUM_QUERIES': NUM_QUERIES, 'D_MODEL': D_MODEL, 'DECODER_LAYERS': DECODER_LAYERS, 'PARAMETER_COUNT': PARAMETER_COUNT}}}})\n"
                "input_manifest = validate_inputs(image, threshold=DETECTION_THRESHOLD, names=[probe_record['id']])\n"
                "try:\n"
                "    validate_inputs(image, threshold=1.5)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'threshold-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "started = time.perf_counter()\n"
                "result = pipe.detect(image, threshold=DETECTION_THRESHOLD)\n"
                "detect_seconds = round(time.perf_counter() - started, 3)\n"
                "low = pipe.detect(image, threshold=0.05)\n"
                "checks = {{\n"
                "    'at_most_num_queries': len(low['detections']) <= MAX_DETECTIONS,\n"
                "    'labels_in_vocabulary': all(d['label'] in LABELS for d in low['detections']),\n"
                "    'scores_descending': all(a['score'] >= b['score'] for a, b in zip(low['detections'], low['detections'][1:])),\n"
                "    'boxes_inside_image': all(0 <= d['box'][0] <= d['box'][2] <= image.width + 1 and 0 <= d['box'][1] <= d['box'][3] <= image.height + 1 for d in low['detections']),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'detect output failed a sanity check: {{checks}}')\n"
                "report = evaluation_report(low, probe_record['boxes'], sample_kind='one Open Food Facts test photograph' if not USE_BYOD else 'one BYOD test image')\n"
                "print({{'probe_id': probe_record['id'], 'reference_boxes': len(probe_record['boxes']), 'detections_at_threshold': len(result['detections']), 'detections_at_0.05': len(low['detections']), 'seconds': detect_seconds, 'device': pipe.device, 'checks': checks, 'findings': len(input_manifest['findings'])}})\n"
                "print({{'verdict': report['verdict'], 'metrics': report['metrics'], 'reason': report.get('reason')}})\n"
                "if report['verdict'] != 'sample-sanity':  # a contract check on the report itself, not a quality claim\n"
                "    raise RuntimeError(f\"evaluation_report returned verdict {{report['verdict']!r}} for a record with reference boxes\")\n"
                "try:  # `display` is provided by the isolated worker (and by any IPython kernel); the worker has no IPython\n"
                "    display(draw_boxes(image, probe_record['boxes'], low['detections']).reduce(2))\n"
                "except NameError:\n"
                "    print({{'preview': 'display unavailable; the preview PNG is written in Section 9'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** `detections_at_threshold` against `detections_at_0.05`, the checks, and the report's `box_iou`.\n\n<details><summary>Check your reasoning</summary>None, or almost none. The checkpoint's `table` class was learned on rendered PDF pages, and on a product photograph its scores stay far below 0.9. In the CPU build record the probe photograph had one `table` detection at the low threshold of 0.05, overlapping the reference box at IoU 0.47 — the queries look in roughly the right place but do not believe it is a table. That domain gap, not a defect, is what Sections 6–7 adapt around.</details>"
            ),
        },
        {
            "md": (
                "## 6. The fixed-box prior, the zero-shot checkpoint and the frozen policy\n\n"
                "Three rows frame the adaptation, all on the 25 test photographs and all threshold-free: AP@0.5, AP@0.75 "
                "and mAP rank every query of every image by score, so a detector is judged on its ordering, and the "
                "recall / precision at the operating threshold (0.9, the pipeline's default) and at 0.5 are printed beside "
                "them. The **fixed-box prior** puts one box per photograph at the training split's mean normalised box — "
                "what \"the table is usually here\" alone buys. The **zero-shot checkpoint** (`evaluate_zero_shot`) scores "
                "each PubTables query by its `table` + `table rotated` softmax mass, the closest thing the untouched model "
                "has to the new class. The **frozen policy** is `adapt` with `trainable_layers=0`: a new two-way class "
                "head and a copy of the box head trained on the cached decoder features of the 72 training photographs "
                "under the DETR set loss for `HEAD_STEPS` full-batch steps, then scored on the test split by `evaluate`. "
                "**What to look for:** the frozen policy far above the prior and the zero-shot row, and the zero-shot row's mean best "
                "IoU against its AP — the untrained queries can sit near the tables while scoring them as something else. About half "
                "a minute on CPU.\n\n"
                "*Evaluation practice.* The cell reports a **verdict** — whether the frozen policy beats the prior on AP@0.5 — and "
                "records it; when the prior is competitive (your boxes are always in one place) that is a finding, and the notebook "
                "continues. The policy identity stays a hard check: `trainable_layers=0` must give the frozen policy.\n\n"
                "**Predict before running:** order the three rows by AP@0.5 — the fixed-box prior, the untouched checkpoint "
                "(zero-shot), and new heads trained on its frozen decoder features."
            ),
            "code": (
                'HEAD_STEPS = 300  # @param {{type:"integer"}}\n'
                'HEAD_LR = 1e-3  # @param {{type:"number"}}\n\n'
                "def brief(m):\n"
                "    return {{'ap50': round(m['ap50'], 4), 'ap75': round(m['ap75'], 4), 'map': round(m['map'], 4), 'recall_at_0.9': round(m['operating_points'][str(DETECTION_THRESHOLD)]['recall'], 4), 'recall_at_0.5': round(m['operating_points']['0.5']['recall'], 4), 'precision_at_0.5': round(m['operating_points']['0.5']['precision'], 4), 'mean_best_iou': round(m['mean_best_iou'], 4), 'n': m['n_images']}}\n\n"
                "prior = prior_baseline(train_records, test_records, threshold=DETECTION_THRESHOLD)\n"
                "print({{'fixed_box_prior': brief(prior), 'baseline': prior['baseline'], 'prior_box_normalised': [round(v, 3) for v in prior['prior_box_normalised']]}})\n"
                "t0 = time.perf_counter()\n"
                "zero_shot_test = pipe.evaluate_zero_shot(test_records)\n"
                "print({{'zero_shot': brief(zero_shot_test), 'policy': zero_shot_test['policy'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "t0 = time.perf_counter()\n"
                "probe_result = pipe.adapt(train_records, val_records, head_steps=HEAD_STEPS, head_lr=HEAD_LR, trainable_layers=0)\n"
                "frozen_test = pipe.evaluate(test_records)\n"
                "print({{'frozen_policy': probe_result['policy'], 'head_final_loss': round(probe_result['head_final_loss'], 4), 'validation': probe_result['history'][0]['val'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'frozen_policy_test': brief(frozen_test), 'loss': round(frozen_test['loss'], 4), 'verdict': frozen_test['verdict']}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "if probe_result['policy'] != POLICY_FROZEN:  # a contract check: trainable_layers=0 is the frozen policy by definition\n"
                "    raise RuntimeError(f\"adapt(trainable_layers=0) returned policy {{probe_result['policy']!r}}\")\n"
                "# A reported verdict, not an assertion: on your data the prior may be competitive, and that is a finding.\n"
                "frozen_verdict = 'frozen policy above the fixed-box prior on AP@0.5' if frozen_test['ap50'] > prior['ap50'] else 'frozen policy NOT above the fixed-box prior on AP@0.5: your boxes may sit in one place; read the prior before the detector'\n"
                "print({{'frozen_vs_prior': frozen_verdict}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the three rows' AP@0.5, AP@0.75, mAP and mean best IoU, and the recall at 0.9 against the recall at 0.5.\n\n<details><summary>Check your reasoning</summary>Prior, then zero-shot, then frozen heads. In the Kaggle T4 release run (19 September 2026) the prior scored AP@0.5 2.66 %, the zero-shot checkpoint 9.67 % (mean best IoU 0.49 — the queries already sit near the tables) and the frozen policy 31.79 % (AP@0.75 6.81 %, mAP 14.38 %); the CPU build record measured 37.6 % / 7.4 % / 15.5 % for the frozen policy. The frozen features already locate nutrition tables once a head knows what to look for. The recall at 0.9 was 0 for the new head: its scores rarely reach a threshold set for the PubTables head.</details>"
            ),
        },
        {
            "md": (
                "## 7. The unfrozen policy: a bounded decoder unfreeze selected against the heads\n\n"
                "`adapt` with `TRAINABLE_LAYERS` > 0 first retrains the new heads on the frozen features (epoch 0 of the "
                "history, the frozen policy), then unfreezes the last `TRAINABLE_LAYERS` decoder layers — two by default, "
                "3,157,504 of 28,799,431 parameters; the backbone, the input projection, the encoder, the query embeddings "
                "and the earlier decoder layers stay frozen — and trains them with both heads end to end, one photograph "
                "per step, for `EPOCHS` epochs (AdamW at `LEARNING_RATE`, weight decay 0.01, gradient clipping 0.1, seeded "
                "order, no augmentation) under the same DETR set loss: Hungarian matching of the 15 queries to the "
                "reference boxes (the exact rectangular assignment, the O(n²m) shortest-augmenting-path algorithm, for every supported box count), cross-entropy with a 0.1 no-object weight, L1 "
                "and GIoU box terms — all in the carried module, no external matcher. Every epoch is scored on validation, "
                "and the epoch with the **lowest validation loss** is kept — epoch 0, the heads alone, competes on equal "
                "terms, so the selected policy can be either. AP@0.5 and mAP are printed beside the loss at every "
                "epoch.\n\n"
                "Watch the validation loss: in the build record it fell every epoch at 1e-4 (3.418 for the heads → 3.350 → 3.305 → 3.292, so epoch 3 was selected); at 3e-4 it fell only to 3.401 and epoch 3 was selected by a hair; with all six layers unfrozen it reached 3.357 at epoch 1 and rose afterwards (3.373, 3.485), so epoch 1 was kept. Note that DETR's train-mode dropout makes the per-photograph "
                "training loss sit above the full-batch head loss of epoch 0.\n\n"
                "*Core concept.* Every call to `pipe.adapt` starts from the **pinned base**: decoder layers an earlier call (or "
                "an artifact) changed are restored before the features are cached, so both policies start from the checkpoint "
                "and re-running Sections 6–8 with a changed field repeats the comparison validly. To compare a change side by "
                "side without replacing the default exports, use Section 10.\n\n"
                "**Predict before running:** if validation loss chooses the unfrozen policy, will the held-out AP@0.75 and mAP "
                "necessarily be higher too?"
            ),
            "code": (
                'EPOCHS = 3  # @param {{type:"integer"}}\n'
                'LEARNING_RATE = 1e-4  # @param {{type:"number"}}\n'
                'TRAINABLE_LAYERS = 2  # @param {{type:"integer"}}\n\n'
                "def report_epoch(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'stage': entry['stage'], 'train_loss': round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row['val_loss'] = round(entry['val']['loss'], 4)\n"
                "        row['val_ap50'] = round(entry['val']['ap50'], 4)\n"
                "        row['val_map'] = round(entry['val']['map'], 4)\n"
                "    print(row)\n\n"
                "settings = {{'epochs': EPOCHS, 'lr': LEARNING_RATE, 'trainable_layers': TRAINABLE_LAYERS, 'head_steps': HEAD_STEPS, 'head_lr': HEAD_LR}}\n"
                "if settings != {{'epochs': 3, 'lr': 1e-4, 'trainable_layers': 2, 'head_steps': 300, 'head_lr': 1e-3}}:\n"
                "    print({{'note': 'changed settings: this run starts again from the pinned base and replaces the default results of Sections 8-9; Section 10 compares a change side by side instead', 'settings': settings}})\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, head_steps=HEAD_STEPS, head_lr=HEAD_LR, trainable_layers=TRAINABLE_LAYERS, epochs=EPOCHS, lr=LEARNING_RATE, progress=report_epoch)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "report_epoch(adapt_result['history'][0])\n"
                "print({{'selected_policy': adapt_result['policy'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'trainable_heads': adapt_result['n_trainable_head'], 'trainable_layers': adapt_result['n_trainable_layers'], 'total_parameters': adapt_result['n_total'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "**What to notice:** epoch 0 (the frozen policy re-trained as the baseline of the history), `val_loss` per epoch, and the selected policy.\n\n<details><summary>Check your reasoning</summary>In the CPU build record the validation loss fell every epoch at 1e-4 (3.418 for the heads → 3.350 → 3.305 → 3.292), so the unfrozen policy was selected at epoch 3; the Kaggle T4 release run also selected the unfrozen policy. Validation loss is the selection criterion, not the result: whether the selected model is better on held-out photographs, and on which metric, is Section 8's question.</details>"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or policy selection, and no product in it appears in the "
                "training or validation splits. The selected model is scored exactly as the frozen policy was in Section 6, "
                "and the four rows are put side by side: fixed-box prior, zero-shot checkpoint, frozen policy, selected "
                "policy. Read the policy first: if validation kept the heads, the last two rows are the same model; if it "
                "chose the unfreeze, the delta is what the unfreeze bought on 25 photographs. **What to look for:** compare AP@0.5, "
                "AP@0.75 and mAP between the two policies *separately* — they need not move together, and on 26 boxes a difference "
                "of a few points is within noise (one box is about four points of recall). The cell records verdicts — whether "
                "the selected model beats the fixed-box prior, and whether it improved on the frozen heads at AP@0.5 and at mAP — "
                "instead of asserting them, because a policy that does not help is a finding, not an error, and Section 9 still "
                "exports, reloads and writes the result. 25 photographs with 26 boxes from one seeded split of one corpus give no "
                "dispersion estimate.\n\n"
                "**Predict before running:** validation loss chose between the policies. Will the held-out test agree with it on "
                "every metric?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "rows = {{'fixed_box_prior': prior, 'zero_shot': zero_shot_test, 'frozen_policy': frozen_test, 'selected_policy': adapted_test}}\n"
                "comparison = {{metric: {{name: round(m[metric], 4) for name, m in rows.items()}} for metric in ('ap50', 'ap75', 'map', 'mean_best_iou')}}\n"
                "comparison['recall_at_0.5'] = {{name: round(m['operating_points']['0.5']['recall'], 4) for name, m in rows.items()}}\n"
                "comparison['precision_at_0.5'] = {{name: round(m['operating_points']['0.5']['precision'], 4) for name, m in rows.items()}}\n"
                "comparison['recall_at_0.9'] = {{name: round(m['operating_points'][str(DETECTION_THRESHOLD)]['recall'], 4) for name, m in rows.items()}}\n"
                "comparison['loss'] = {{'frozen_policy': round(frozen_test['loss'], 4), 'selected_policy': round(adapted_test['loss'], 4)}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 4) for metric in ('ap50', 'ap75', 'map')}}\n"
                "comparison['delta_vs_zero_shot'] = {{metric: round(adapted_test[metric] - zero_shot_test[metric], 4) for metric in ('ap50', 'ap75', 'map')}}\n"
                "comparison['selected_policy'] = adapt_result['policy']\n"
                "# Reported verdicts, not assertions: a policy that does not help is a result to record, and export and reload still run.\n"
                "comparison['verdicts'] = {{\n"
                "    'frozen_vs_prior': frozen_verdict,\n"
                "    'selected_above_prior_ap50': bool(adapted_test['ap50'] > prior['ap50']),\n"
                "    'selected_vs_frozen_ap50': 'improved' if adapted_test['ap50'] > frozen_test['ap50'] else ('no gain' if adapted_test['ap50'] == frozen_test['ap50'] else 'worse'),\n"
                "    'selected_vs_frozen_map': 'improved' if adapted_test['map'] > frozen_test['map'] else ('no gain' if adapted_test['map'] == frozen_test['map'] else 'worse'),\n"
                "}}\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'split_summary': summary,\n"
                "    'single_image_report': report,\n"
                "    'baselines': {{'fixed_box_prior': prior, 'zero_shot': zero_shot_test}},\n"
                "    'frozen_policy': {{'adaptation': {{k: v for k, v in probe_result.items() if k not in ('history', 'trainable_names')}}, 'history': probe_result['history'], 'test': frozen_test}},\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'verdicts': comparison['verdicts']}})\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the selected policy, the four rows on each metric, `delta_vs_frozen` and the `verdicts` row.\n\n<details><summary>Check your reasoning</summary>Not necessarily — and in the release run it did not. In the Kaggle T4 run (19 September 2026) validation loss chose the unfrozen policy, and on the test split it scored AP@0.5 33.15 % against the frozen heads' 31.79 %, but AP@0.75 6.59 % against 6.81 % and mAP 12.88 % against 14.38 %: better at 0.5, slightly worse at the stricter thresholds. The CPU build record had shown the opposite pattern (39.8 % / 13.4 % / 16.3 % against 37.6 % / 7.4 % / 15.5 %). On 26 boxes these differences are within noise; the lesson is that a selection criterion (validation loss) and a held-out metric can disagree, so read every metric and do not assert a direction one run cannot support.</details>"
            ),
        },
        {
            "md": (
                "## 9. Detect before and after, export the adapter and reload it\n\n"
                "Three test photographs are run through `detect_adapted` with the selected model at a 0.5 threshold and "
                "rendered beside the frozen policy's detections (from a fresh pipeline with new heads trained the same way — "
                "after an unfreeze the decoder inside `pipe` has moved, so the frozen column needs its own decoder) and the "
                "reference boxes: reference in green, detections in red with the head's score, which is a softmax under a "
                "0.1 no-object weight, not a calibrated confidence; `outputs/{stem}_preview.png` holds the sheet. `detect` "
                "— the PubTables heads — still answers in its own label space on the same photographs.\n\n"
                "`save_artifact` writes the new class head and box head and, when the unfrozen policy was selected, the "
                "trained decoder-layer tensors — about 0.5 MB for the heads alone, 13.2 MB with two decoder "
                "layers — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id "
                "and revision, the digest of the base `model.safetensors`, the class, the selected policy, the tensor names, "
                "the file size and SHA-256, the training configuration and the epoch history (OUT8). "
                "`TableTransformerDetectionPipeline.from_artifact` re-verifies the base snapshot, checks the artifact "
                "manifest and digest **before** deserialising, rebuilds the heads from the manifest, refuses any tensor "
                "that is not a decoder-layer tensor of the base, and overlays the tensors onto a freshly loaded base — a new "
                "object from files, not the in-memory model (VER2). The cell asserts identical query scores and boxes on "
                "three photographs and an identical test AP@0.5 (VER4) — a contract check, so it stays a hard check.\n\n"
                "**Predict before running:** in the preview sheet, will the selected model's boxes look tighter than the frozen "
                "heads' on these three photographs?"
            ),
            "code": (
                "import shutil\n\n"
                "show = test_records[:3]\n"
                "frozen_pipe = TableTransformerDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "frozen_pipe.adapt(train_records, val_records, head_steps=HEAD_STEPS, head_lr=HEAD_LR, trainable_layers=0)\n"
                "before_after = []\n"
                "panels = []\n"
                "for record in show:\n"
                "    after = pipe.detect_adapted(record['image'], threshold=0.5)\n"
                "    before = frozen_pipe.detect_adapted(record['image'], threshold=0.5)\n"
                "    base = pipe.detect(record['image'], threshold=0.05)\n"
                "    best_after = max((box_iou(d['box'], ref) for d in after['detections'] for ref in record['boxes']), default=0.0)\n"
                "    best_before = max((box_iou(d['box'], ref) for d in before['detections'] for ref in record['boxes']), default=0.0)\n"
                "    before_after.append({{'id': record['id'], 'reference_boxes': len(record['boxes']), 'frozen_detections': len(before['detections']), 'frozen_best_iou': round(best_before, 4), 'selected_detections': len(after['detections']), 'selected_best_iou': round(best_after, 4), 'pubtables_detections_at_0.05': len(base['detections']), 'image_url': record.get('image_url', '')}})\n"
                "    print(before_after[-1])\n"
                "    left = draw_boxes(record['image'], record['boxes'], before['detections'])\n"
                "    right = draw_boxes(record['image'], record['boxes'], after['detections'])\n"
                "    panel = Image.new('RGB', (left.width * 2 + 8, left.height), (255, 255, 255))\n"
                "    panel.paste(left, (0, 0))\n"
                "    panel.paste(right, (left.width + 8, 0))\n"
                "    panels.append(panel)\n"
                "sheet = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels) + 8 * (len(panels) - 1)), (255, 255, 255))\n"
                "y = 0\n"
                "for panel in panels:\n"
                "    sheet.paste(panel, (0, y))\n"
                "    y += panel.height + 8\n"
                "sheet.save('outputs/{stem}_preview.png')\n"
                "try:  # `display` is provided by the isolated worker (and by any IPython kernel)\n"
                "    display(sheet.reduce(4))\n"
                "except NameError:\n"
                "    pass\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'policy': artifact_manifest['adapter']['policy'], 'classes': artifact_manifest['adapter']['classes'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = TableTransformerDetectionPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_test = reloaded.evaluate(test_records)\n"
                "parity = {{'queries_identical': reloaded.predict_boxes(show) == pipe.predict_boxes(show), 'ap50_in_memory': round(adapted_test['ap50'], 6), 'ap50_reloaded': round(reloaded_test['ap50'], 6), 'classes_identical': reloaded.classes == pipe.classes}}\n"
                "print({{'reload_parity': parity, 'reloaded_policy': reloaded.adapter['policy'], 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['queries_identical'] and parity['classes_identical'] and abs(adapted_test['ap50'] - reloaded_test['ap50']) < 1e-9\n\n"
                "weight_entry = next(entry for entry in MANIFEST['files'] if entry['path'] == WEIGHTS_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': len(MANIFEST['files']), 'total_bytes': MANIFEST['totalBytes'], 'fetched_this_run': fetched, 'weight_file': WEIGHTS_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'photographs': len(SAMPLE_RECORDS), 'bytes': CORPUS_BYTES, 'license': CORPUS_LICENSE, 'class': CLASS_NAME, 'longest_side': CORPUS_LONGEST_SIDE}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'probe_id': probe_record['id'], 'single_image_report': report, 'seconds': detect_seconds}},\n"
                "    'comparison': comparison,\n"
                "    'before_after': before_after,\n"
                "    'preview_file': 'outputs/{stem}_preview.png',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors']), 'policy': artifact_manifest['adapter']['policy']}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'timm': timm.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "**What to notice:** `frozen_best_iou` against `selected_best_iou` in `before_after`, the preview sheet, and the reload parity line.\n\n<details><summary>Check your reasoning</summary>Three photographs cannot answer that: each row's best IoU is one box, and the release run's held-out numbers (Section 8) moved in different directions at different IoU thresholds. Use the sheet to see *what* the detector does — where it fires, how many boxes, how the scores sit below 0.9 — not to rank the policies. Reload parity held in the release run: identical query scores and boxes, and test AP@0.5 0.3315 in memory and reloaded.</details>"
            ),
        },
        {
            "md": (
                "## 10. Change one thing: unfreeze all six decoder layers (optional)\n\n"
                "*Evaluation practice.* A **Predict → Change one thing → Run → Observe → Explain** activity, off by default so "
                "Run all is unaffected. Set `RUN_EXPERIMENT = True`, change **one** field — by default all six decoder layers "
                "train instead of two — and run this cell after Sections 4–9. The experiment loads its **own** pipeline from the "
                "verified snapshot, so it starts from the checkpoint and never touches the default `pipe`; it writes only to "
                "`outputs/{stem}_experiment/`, prints the default and the changed run side by side, and checks that the default "
                "exports (adapter, evaluation report, result) are byte-identical afterwards. A few minutes on CPU.\n\n"
                "**Predict:** with three times the trainable decoder layers, which epoch will validation select, and will the "
                "held-out mAP rise?"
            ),
            "code": (
                "RUN_EXPERIMENT = False  # @param {{type:\"boolean\"}}\n"
                "EXPERIMENT_TRAINABLE_LAYERS = 6  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_EPOCHS = 3  # @param {{type:\"integer\"}}\n"
                "EXPERIMENT_LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n\n"
                "if not RUN_EXPERIMENT:\n"
                "    print({{'experiment': 'skipped (RUN_EXPERIMENT = False); the default path above is complete'}})\n"
                "else:\n"
                "    canonical_files = {{'adapter': artifact_dir / 'adapter.safetensors', 'evaluation_report': Path('outputs/{stem}_evaluation_report.json'), 'result': Path('outputs/{stem}_result.json')}}\n"
                "    canonical = {{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in canonical_files.items()}}\n"
                "    experiment_dir = Path('outputs/{stem}_experiment')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_dir.mkdir(parents=True)\n"
                "    # Its own pipeline from the verified snapshot: the experiment starts from the checkpoint and the default pipe is untouched.\n"
                "    experiment_pipe = TableTransformerDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "    experiment_result = experiment_pipe.adapt(train_records, val_records, head_steps=HEAD_STEPS, head_lr=HEAD_LR, trainable_layers=EXPERIMENT_TRAINABLE_LAYERS, epochs=EXPERIMENT_EPOCHS, lr=EXPERIMENT_LEARNING_RATE, progress=report_epoch)\n"
                "    experiment_test = experiment_pipe.evaluate(test_records)\n"
                "    side_by_side = {{\n"
                "        'settings': {{'default': {{'trainable_layers': adapt_result['trainable_layers'], 'epochs': adapt_result['epochs'], 'lr': adapt_result['lr']}}, 'experiment': {{'trainable_layers': EXPERIMENT_TRAINABLE_LAYERS, 'epochs': EXPERIMENT_EPOCHS, 'lr': EXPERIMENT_LEARNING_RATE}}}},\n"
                "        'selected_policy': {{'default': adapt_result['policy'], 'experiment': experiment_result['policy']}},\n"
                "        'best_epoch': {{'default': adapt_result['best_epoch'], 'experiment': experiment_result['best_epoch']}},\n"
                "        'validation_loss_by_epoch': {{'default': [round(e['val']['loss'], 4) for e in adapt_result['history'] if e.get('val')], 'experiment': [round(e['val']['loss'], 4) for e in experiment_result['history'] if e.get('val')]}},\n"
                "        'test': {{metric: {{'frozen_policy': round(frozen_test[metric], 4), 'default': round(adapted_test[metric], 4), 'experiment': round(experiment_test[metric], 4)}} for metric in ('ap50', 'ap75', 'map')}},\n"
                "    }}\n"
                "    for key, row in side_by_side.items():\n"
                "        print({{key: row}})\n"
                "    with open(experiment_dir / 'experiment_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'side_by_side': side_by_side, 'history': experiment_result['history']}}, handle, indent=2, ensure_ascii=False, default=str)\n"
                "    unchanged = {{name: hashlib.sha256(path.read_bytes()).hexdigest() == canonical[name] for name, path in canonical_files.items()}}\n"
                "    if not all(unchanged.values()):\n"
                "        raise RuntimeError(f'the experiment changed a default export: {{unchanged}}')\n"
                "    print({{'default_exports_unchanged': unchanged, 'experiment_outputs': str(experiment_dir)}})\n"
                "    del experiment_pipe"
            ),
        },
        {
            "md": (
                "**Observe → Explain.** Compare the two validation-loss curves and the `test` rows.\n\n"
                "<details><summary>Check your reasoning</summary>Epoch 0 of both runs is the frozen policy (new heads on the "
                "checkpoint's features), so the two curves start from the same kind of model. In the CPU build record all six layers "
                "reached the lowest validation loss at epoch 1 (3.357) and rose afterwards (3.373, 3.485), so epoch 1 was kept, "
                "with 41.0 % AP@0.5 / 18.3 % mAP and a 38 MB adapter — more capacity began to overfit 72 photographs within two "
                "epochs. No experiment run is recorded on the release runtime; on 26 test boxes, read a difference of a few points "
                "as noise, and explain it from the validation curve rather than from the test number.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "On 25 held-out product photographs the fixed-box prior and the untouched PubTables detector score a few percent "
        "AP@0.5 and new heads on its frozen decoder features roughly ten times that — in the Kaggle T4 release run "
        "(19 September 2026) 2.66 %, 9.67 % and 31.79 %. Whether unfreezing the last two decoder layers adds anything is "
        "not settled by this sample: validation loss chose the unfreeze in that run, which then scored higher at AP@0.5 "
        "(33.15 %) but lower at AP@0.75 and mAP (6.59 % and 12.88 % against 6.81 % and 14.38 %), while the CPU build record "
        "had seen gains on all three — differences of a few points on 26 boxes, within noise. The adapter reloads to "
        "identical query scores. That is the claim and the finding: the adaptation contract gives the PubTables detector a "
        "class and a domain it was not trained on, runs both policies end to end on a real box-labelled corpus with the "
        "set loss implemented in the open, chooses between them on validation rather than by assumption, and reports the "
        "answer against a prior and the untouched checkpoint rather than in isolation.\n\n"
        "The test split is 25 photographs with 26 boxes from one seeded split of one small corpus with no dispersion "
        "estimate — one box is about four points of recall, so a few points of AP is noise. AP ranks every query by the "
        "new head's score; the pipeline's operating threshold of 0.9 was set for the PubTables head and the new head's "
        "scores rarely reach it (the recall at 0.9 is reported beside the recall at 0.5) — a deployment must choose its "
        "own threshold on its own labelled images, exactly as the model card says for the base detector. When the "
        "unfrozen policy is selected it changes the last decoder layers, which every query shares, so `detect` — which "
        "keeps the PubTables heads — reads a moved decoder afterwards; the artifact records which policy won.\n\n"
        "Three things to carry to real data. **Baselines first:** the fixed-box prior and the zero-shot checkpoint on "
        "*your* images are the numbers to read before any trained head's — if the prior is competitive, your boxes are "
        "in one place and the detector is not the interesting part. **Leakage:** split by product, session or device "
        "(the contract splits by `group` / `barcode`, never by photograph). **Thresholds:** AP is threshold-free and an "
        "operating point is not; pick it on validation for the class you trained, not from the base checkpoint.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real box-labelled "
        "corpus, validate the demonstrated dataset contract without leakage, execute the inference contract with a "
        "`sample-sanity` report against reference boxes, train new detection heads and a bounded decoder unfreeze with "
        "validation-based policy selection, evaluate by AP against a prior and the zero-shot checkpoint on an independent "
        "split, and emit the shown machine-readable artifacts — without the repository being reachable. It does **not** "
        "establish benchmark superiority, PubTables-1M accuracy, a usable acceptance threshold, or production fitness.\n\n"
        "**Optional experiments (off by default; each names its field and what to run):** Section 10 unfreezes all six "
        "decoder layers in its own pipeline and prints it beside the default run — change `EXPERIMENT_TRAINABLE_LAYERS`, "
        "`EXPERIMENT_EPOCHS` or `EXPERIMENT_LEARNING_RATE` (the build record's 3e-4 was selected by a hair) there and run "
        "that cell again. Setting `EPOCHS = 0` or changing `HEAD_STEPS` / `TRAINABLE_LAYERS` / `LEARNING_RATE` and choosing "
        "**Run after** from Section 6 also starts from the pinned base — every `adapt` restores it first — but replaces the "
        "default results and exports. BYOD: `USE_BYOD` and `BYOD_PATH` in Section 4, then **Run after** from Section 4, "
        "and read the prior and the zero-shot row before either policy.\n\n"
        "## Troubleshooting\n\n"
'- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — you are on Windows, macOS or an ARM machine. Use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused, an incomplete one is finished. If it repeats, the network is blocking or altering `files.pythonhosted.org` or `pypi.org`.\n- **"The isolated environment\'s Python process exited"** — usually out of memory. Restart the session and choose **Run all**; leave the optional experiment off on a small runtime.\n- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the cells after it keep working. After a session restart, run from the top.\n- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete it from the snapshot folder Section 3 prints and run Section 3 again; the snapshot comes from `huggingface.co`.\n- **Section 4 cannot fetch a photograph, or one fails its digest** — `fetch_corpus` names it; the default path needs `static.openfoodfacts.org`, which can throttle (the fetch retries with back-off). Run Section 4 again (cached photographs are re-hashed); delete `weights/off-nutrition/` if a cached file is corrupt.\n- **Out of memory** — the model is small; restart the session and choose **Run all**, and leave Section 10 off on a small runtime.\n- **BYOD: "BYOD_PATH … does not exist"** — the path is relative to the working directory printed in the message.\n- **BYOD: "the upload dialog exists only in Google Colab"** — on Kaggle or Jupyter, put the zip in the runtime (or attach it as a dataset) and set `BYOD_PATH`.\n- **BYOD: "Upload exactly one .zip file"** — the dialog was cancelled or several files were chosen; run the cell again.\n- **BYOD: "boxes.csv line N (file …): names an image that is not in the dataset"** — fix the `file` column of that row.\n- **BYOD: "… carries EXIF orientation …"** — a phone photograph stored sideways; re-save it upright and draw the boxes on the upright image.\n- **BYOD: "… has no `group`"** — every row needs the product, session or device the photograph belongs to.\n- **BYOD: "split leaves … supply at least 12 photographs in as many groups"** — add photographs or groups.\n'
        "## Glossary\n\n"
        "- **DETR query** — one of 15 learned slots the decoder fills with a box and a class distribution; most answer "
        "\"no object\".\n"
        "- **Set loss / Hungarian matching** — predictions are matched one-to-one to reference boxes at the lowest total "
        "cost before the loss is computed.\n"
        "- **GIoU** — generalised IoU, a box-overlap term that still gives a gradient when boxes do not overlap.\n"
        "- **IoU** — intersection over union of two boxes.\n"
        "- **AP@0.5 / AP@0.75 / mAP** — average precision over the ranked predictions, counting a match at IoU ≥ 0.5, ≥ 0.75, "
        "and averaged over 0.5..0.95; threshold-free.\n"
        "- **Operating point** — a score threshold and the recall and precision it gives; chosen per deployment.\n"
        "- **Fixed-box prior** — one box per image at the training set's mean box.\n"
        "- **Zero-shot row** — the untouched checkpoint, its `table` classes read as the new class.\n"
        "- **Frozen policy / unfrozen policy** — new heads on frozen decoder features; the same plus the last decoder "
        "layers trained.\n"
        "- **Validation selection** — choosing the policy and epoch by validation loss, never by the test split.\n"
        "- **Group-disjoint split** — every product (or session, device) on one side of the split.\n"
        "- **Adapter / reload parity** — the trained tensors only (safetensors) overlaid on the pinned base; the reloaded "
        "pipeline gives identical scores.\n"
        "- **BYOD** — bring your own data: your box-labelled photographs through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run, not the recorded one:\n\n"
        "- The task was ___ on ___ test photographs with ___ boxes.\n"
        "- The prior scored AP@0.5 ___, the zero-shot row ___ and the frozen policy ___ (mAP ___).\n"
        "- Validation selected the ___ policy at epoch ___; on the test split it scored AP@0.5 ___ / AP@0.75 ___ / mAP ___, "
        "so against the frozen heads it ___.\n"
        "- What I would need before claiming the unfreeze helps: ___ (for example more test boxes, several seeds).\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/table-transformer-detection-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/table-transformer-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/table-transformer-detection-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/microsoft/table-transformer\n"
        "- PubTables-1M: Towards Comprehensive Table Extraction From Unstructured Documents (Smock, Pesala, Abraham, 2021): https://arxiv.org/abs/2110.00061\n"
        "- End-to-End Object Detection with Transformers (DETR; the set loss and Hungarian matching, Carion et al., 2020): https://arxiv.org/abs/2005.12872\n"
        "- Open Food Facts nutrition-table detection dataset (boxes, ODbL) and Open Food Facts images (CC BY-SA 3.0, credited to their contributors): https://huggingface.co/datasets/openfoodfacts/nutrition-table-detection — https://world.openfoodfacts.org/data\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
