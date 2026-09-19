# Release verification

`tutorials/table_transformer_detection_colab.ipynb` (`E2E`, **standalone** carrier) is a
**release candidate** until the exact notebook revision has executed top-to-bottom in a clean
supported runtime. Unit tests, JSON validation, code-cell compilation, the generator parity checks
and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under
DIMER Notebook Specification 2.0. This file is the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that
  profile, spec `2.0`, a pedagogical mode, `standalone: true` and `generated_from` (repository, revision, module
  SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on
  the primary path; three cells tagged `embedded_module` equal to
  `src/table_transformer_detection_pipeline/{pipeline,metrics,samples}.py` after the generator's documented
  rewrites (in dependency order, relative imports removed); the
  inline `MANIFEST` equal to the committed snapshot manifest and the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to `tools/build_notebook.py`
  output for its recorded revision; the pinned-install cell with its restart-on-stale-import guard;
  `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cell (and repeated in the inline
  manifest, which the notebook asserts against the module before fetching), the revision is a 40-hex
  immutable commit, and the same identity string appears in `README.md`, `MODEL_CARD.md`, and
  `docs/WEIGHTS.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `TableTransformerDetectionPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus`, `read_corpus`,
  `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_summary`, `write_dataset_csv`,
  `validate_inputs`, `detect`, `evaluation_report`, `prior_baseline`, `evaluate_zero_shot`, `adapt` with
  `trainable_layers=0` and with `TRAINABLE_LAYERS` / `LEARNING_RATE`, `evaluate`, `detect_adapted`,
  `save_artifact`, `from_artifact` with the parity assertion), the ceiling print (`MIN_IMAGE_SIDE`,
  `MAX_IMAGE_SIDE`, `MAX_DETECTIONS`, `NUM_QUERIES`, `D_MODEL`, `DECODER_LAYERS`, `PARAMETER_COUNT`),
  the six exports, the learner-facing statements (frozen and unfrozen policy, lowest validation loss,
  fixed-box prior, untouched checkpoint, AP@0.5, `sample-sanity`, no dispersion estimate, the PubTables
  heads never trained, capability exclusions, the corpus licence) and the gated-off BYOD default listed
  in the validator; forbidden patterns
  (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a
  mutable `revision='main'`, direct `from transformers import` / `TableTransformerForObjectDetection`
  / `AutoImageProcessor` / `from huggingface_hub import` / `from safetensors` / `torch.optim` /
  `.backward(` / `last_hidden_state` / `pred_boxes` / `scipy` / `pipe._model` use **outside the carried
  module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, required heading order, and
  immutable provenance.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `timm`, `safetensors`, `numpy` and
`pillow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the offline unit
suite (`tests/test_pipeline.py`, `tests/test_role_helpers.py`, `tests/test_adaptation.py`,
`tests/test_notebook_parity.py`; injected runner and fetcher, synthetic photographs, no weights;
`tests/test_model_backed.py` skips without the staged snapshot). These are source/provenance and unit checks. They are **not** execution
evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; float32 either way) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy, Pillow or torch that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/table-transformer-detection/` or the corpus cache `weights/off-nutrition/` (the standalone path
   writes the manifest itself, stages the missing file from the Hub, and fetches the 121 pinned photographs from
   `static.openfoodfacts.org`, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `HEAD_STEPS = 300`, `HEAD_LR = 1e-3`, `EPOCHS = 3`,
   `LEARNING_RATE = 1e-4`, `TRAINABLE_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `transformers==4.57.6`, `timm==1.0.29`,
   `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0` (an interpreter restart after
   the install is expected where the runtime's preinstalled torch, numpy or Pillow differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `TableTransformerDetectionPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `box_iou`, `draw_boxes`, `detection_metrics`,
     `prior_baseline`, `SAMPLE_RECORDS`, `fetch_corpus`, `read_corpus`, `build_sample_dataset`, `validate_dataset`,
     `check_split_disjoint`, `split_summary`, `split_dataset`, `load_byod_dataset`, `write_dataset_csv` and the
     ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `microsoft/table-transformer-detection` at the immutable revision, and `verify_snapshot` returning its dict
     (4 files); `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory;
   - Section 4: `fetch_corpus` fetching the 121 pinned photographs (129,318,620 bytes) from
     `static.openfoodfacts.org` into `weights/off-nutrition/`, 121 records read (downscaled to 1,280 px, one
     sliver box dropped), and the seeded draw of 71 / 24 / 24 whole products into 72 / 24 / 25 photographs
     (76 / 25 / 26 boxes) with `check_split_disjoint` reporting no shared photograph and no shared product,
     `split_summary` printed and the three dataset digests `__DIG_TRAIN__` / `__DIG_VAL__` / `__DIG_TEST__`;
     `outputs/…_train.csv` written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `MAX_DETECTIONS` 15) and the contract
     (`NUM_QUERIES` 15, `D_MODEL` 256, `DECODER_LAYERS` 6, `PARAMETER_COUNT` 28,799,431) surfaced;
     `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding
     from the out-of-range-threshold probe); `detect` on one test photograph at `DETECTION_THRESHOLD` with the
     four sanity checks; `evaluation_report` against the photograph's reference boxes at 0.05 with verdict
     **`sample-sanity`** and one `box_iou` entry per reference (≈ 0.47 on the recorded probe photograph, one
     `table` detection); the rendered preview displayed;
   - Section 6: the fixed-box prior (≈ 2.7 % AP@0.5) on the 25 test photographs, `pipe.evaluate_zero_shot` on the
     untouched checkpoint (≈ 9.7 %, mean best IoU ≈ 0.49), then `pipe.adapt(..., trainable_layers=0)` training the
     new heads for 300 steps (policy `frozen backbone, encoder and decoder + new heads`) and `pipe.evaluate`
     (≈ 37.6 % AP@0.5, mAP ≈ 15.5 %) with per-image rows and the cell's assertion that the heads beat the prior;
   - Section 7: `pipe.adapt` printing epoch 0 as the heads on validation (loss ≈ 3.418), then 3 epochs of the last
     two decoder layers (3,157,504 + 133,126 trainable of 28,799,431 parameters) with validation loss / AP@0.5 /
     mAP each epoch (3.418 → 3.350 → 3.305 → 3.292 in the recorded run) and the selected policy
     `unfrozen last 2 decoder layers + new heads` (`best_epoch` 3);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison (prior, zero-shot,
     frozen policy, selected policy), the operating points at 0.9 and 0.5, per-image rows and
     `outputs/…_evaluation_report.json` written (the cell asserts the selected model beats the prior — on the
     sample ≈ 39.8 % versus 2.7 %; the delta over the frozen heads, ≈ +2.2 points, is reported, not asserted);
   - Section 9: three test photographs rendered with `detect`, the heads-only detections (a fresh pipeline adapted
     with `trainable_layers=0`) and the selected model's `detect_adapted` at 0.5, `outputs/…_preview.png`
     written; `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors, manifest.json}` (60 tensors,
     about 13.2 MB with two trained decoder layers — 8 tensors, about 0.5 MB when the heads alone are selected;
     `policy` recorded) and `TableTransformerDetectionPipeline.from_artifact` reloading it with identical query
     scores on two test photographs and an identical test AP@0.5 (the cell asserts both); `outputs/…_result.json`
     written with `NOTEBOOK_SOURCE`, the model identity and licence, the snapshot block (`weight_format`,
     `weight_sha256`), the `corpus` block, the inference-contract items with the single-photograph report, the
     comparison, the before/after rows, the artifact digest and policy, the reload parity, the runtime versions and
     device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model identifier
   and immutable revision, whether the model cache, the weights directory and the corpus cache were clean, outcome,
   produced outputs, the observed metrics and the selected policy (as observations, not a benchmark) and any warning
   or applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `table_transformer_detection_colab.ipynb` (`E2E`) | `__LOCAL_STAMP__` / `__LOCAL_BLOB__` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `table_transformer_detection_colab.ipynb` (`TASK-INFERENCE`, superseded) | `0a86159` / `7a62e8eff1a9` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-table-transformer-detection` v1) | PASSED — 8/8 code cells, 228.7 s, 10 files, 115 MB staged; evidence for the earlier inference-only notebook, which it promoted to Release-grade — not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/table_transformer_detection_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/table_transformer_detection_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
__LOCAL_ROW__
| 2026-09-14 | `0a86159` / `7a62e8eff1a9` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-table-transformer-detection` v1) | Default sample path of the inference-only notebook: synthetic 850×1100 page, `stage_missing_files` fetching the four manifest entries from the Hub, `verify_snapshot`, `detect` → 2 `table` boxes (0.9985 / 0.9973), `evaluation_report` `sample-sanity` with `box_iou` 0.80 / 0.66, 5 exports | 228.7 s | **PASSED** — 8/8 code cells, 10 files, 115 MB staged; history only |
| 2026-09-13 | working tree of the initial build (blob `410df1003fef`, commit `1cd378a`; `TASK-INFERENCE`, superseded) | Local Windows-venv harness, Python 3.12, torch 2.14.0+cu130, transformers 4.57.6, timm 1.0.29 | Default synthetic path, all 8 code cells | 19.9 s | PASS — pre-flight only; history |

## Current status

The `E2E` notebook source is complete and passes all static checks, including the generator parity checks
(`--check` OK). A local pre-flight execution of the committed blob completed the whole default path on CPU —
the 121 photographs read from the cache, validation and product-level split, the inference contract with a
`sample-sanity` report on a real photograph, the fixed-box prior, the zero-shot checkpoint, the new heads and the
decoder unfreeze with validation selection, held-out evaluation, before/after detections, adapter export and reload
parity — which catches defects but is **not** a supported runtime under REL1/REL10, and it ran with the snapshot and
the 121 photographs pre-staged, so neither the 115 MB Hub fetch nor the 129 MB corpus download has been exercised by
this notebook end to end; the earlier `TASK-INFERENCE` Kaggle CPU run did exercise the Hub fetch and digest check
of the same snapshot. Because the notebook blob has changed, the repository steps back from **Release-grade** to
**Candidate** until a Colab or fresh-container run of the exact `E2E` release revision is recorded above. Facts a
reviewer should weigh: the CUDA path has not been executed; every metric is one seeded split of 25 photographs
(26 boxes) with no dispersion estimate; the new head's scores rarely reach the base threshold of 0.9 (recall 3.8 %
there against 65.4 % at 0.5), so the adapted class needs its own operating point.
