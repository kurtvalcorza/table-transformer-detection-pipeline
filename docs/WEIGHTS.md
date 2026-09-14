# Weight provenance and DIMER hosting

- Upstream: `microsoft/table-transformer-detection`
- Immutable revision: `2357cbe2b5a5d1c03e54f32764f06058933b65ab`
- Weight format: SafeTensors (`model.safetensors`, 115,317,516 bytes). The upstream repository also hosts an equivalent `pytorch_model.bin` (pickle); it is **not** pinned, staged or loaded — DIMER does not accept `.bin` uploads and the pipeline never deserialises pickles.
- Manifest: `weights/table-transformer-detection/dimer-base-manifest.json` (4 files: `README.md`, `config.json`, `model.safetensors`, `preprocessor_config.json`; 115,320,191 bytes total, per-file SHA-256)
- Upstream weight license: MIT (the checkpoint's `README.md` front matter)
- DIMER hosting: MIT permits use, modification, distribution, and commercial use subject to preservation of the copyright notice and licence text. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror the pinned snapshot in its model store under the upstream license.
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files absent on disk, at the pinned revision, into the snapshot directory; `verify_snapshot()` then checks every file before any load.
- Loader trust boundary: Transformers `TableTransformerForObjectDetection` / `AutoImageProcessor` with `trust_remote_code=False`, `local_files_only=True` from the verified directory, and `use_pretrained_backbone=False` so the timm ResNet-18 backbone is built from the checkpoint's own weights instead of fetching ImageNet weights from the Hub (the pinned `config.json` says `use_pretrained_backbone: true`; the smoke run loaded with `HF_HUB_OFFLINE=1` to prove no such fetch happens).
