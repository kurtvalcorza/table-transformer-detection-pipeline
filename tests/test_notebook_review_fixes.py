"""Regression tests for the 2026-10-05 notebook review findings (TTD-M1..M5, TTD-m1, TTD-m2).

Every test needs only CI's dependencies and no model: the notebook's own cell sources are executed with stand-ins
where a model would be needed, and restore_base() is exercised on a small torch module, not the checkpoint. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

from table_transformer_detection_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "table_transformer_detection_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PIPELINE = ROOT / "src" / "table_transformer_detection_pipeline" / "pipeline.py"
STEM = "table_transformer_detection"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# --- TTD-M1: no in-kernel install, no restart, idempotent Section 1 ------------------------------------------


def test_ttd_m1_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source


def test_ttd_m1_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    spec = importlib.util.spec_from_file_location("_review_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.check_lock(build._pins(ROOT), lock_text)


def test_ttd_m1_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker — with every variable later cells created — is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


# --- TTD-M2: every adaptation starts from the pinned base -------------------------------------------------------


def test_ttd_m2_adapt_and_load_artifact_restore_the_base_first():
    """The decoder is restored before the frozen features are cached (torch-backed, so checked statically here)."""
    text = PIPELINE.read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    def save_artifact(")]
    order = [adapt.index(m) for m in ("restored = self.restore_base()", "self._remember_base(names)", "cached = [self._decoder_features(", "for epoch in range(1, epochs + 1):")]
    assert order == sorted(order)
    load = text[text.index("    def load_artifact(") : text.index("    def from_artifact(")]
    assert load.index("self.restore_base()") < load.index("params[key].copy_(")
    restore = text[text.index("    def restore_base(") : text.index("    @classmethod")]
    assert "params[name].copy_(self._base_layers[name])" in restore and "self._head, self._bbox_head, self.adapter = None, None, None" in restore


def test_ttd_m2_restore_base_undoes_every_earlier_change_on_torch_tensors():
    """restore_base() on real torch parameters (a two-layer stand-in module, not the checkpoint)."""
    torch = pytest.importorskip("torch")
    from table_transformer_detection_pipeline import TableTransformerDetectionPipeline

    module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    base = {name: value.detach().clone() for name, value in module.named_parameters()}
    pipe = TableTransformerDetectionPipeline(_runner=lambda image, threshold: [], device="cpu", _model=module, _processor=object())
    assert pipe.restore_base() == []
    pipe._remember_base(["1.weight", "1.bias"])
    with torch.no_grad():
        for value in module.parameters():
            value.add_(1.0)
    pipe._remember_base(["1.weight", "0.weight"])  # a second run: 1.weight keeps its first (base) value
    pipe._head, pipe._bbox_head, pipe.adapter = object(), object(), {"policy": "x"}
    assert pipe.restore_base() == ["0.weight", "1.bias", "1.weight"]
    params = dict(module.named_parameters())
    assert torch.equal(params["1.weight"], base["1.weight"]) and torch.equal(params["1.bias"], base["1.bias"])
    assert torch.equal(params["0.weight"], base["0.weight"] + 1.0)  # remembered after it changed: its value then is kept
    assert (pipe._head, pipe._bbox_head, pipe.adapter) == (None, None, None)


def test_ttd_m2_byod_rerun_restores_the_base_and_the_experiment_has_its_own_pipeline(notebook):
    section_4 = _cell(notebook, "USE_BYOD = False")
    assert section_4.index("restored_layers = pipe.restore_base()") < section_4.index("if USE_BYOD:")
    experiment = _cell(notebook, "RUN_EXPERIMENT = False")
    assert "experiment_pipe = TableTransformerDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)" in experiment
    assert f"Path('outputs/{STEM}_experiment')" in experiment
    assert "raise RuntimeError(f'the experiment changed a default export: {unchanged}')" in experiment
    assert not re.search(r"(?<!experiment_)pipe\.adapt\(", experiment)
    assert "**Predict → Change one thing → Run → Observe → Explain**" in _markdown(notebook)
    assert "they do not affect the default path" not in _markdown(notebook)


# --- TTD-M3: quality outcomes are reported verdicts -------------------------------------------------------------


def test_ttd_m3_no_quality_assert_remains(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    asserts = re.findall(r"(?m)^\s*assert .*$", code)
    assert asserts == ["assert parity['queries_identical'] and parity['classes_identical'] and abs(adapted_test['ap50'] - reloaded_test['ap50']) < 1e-9"]


def _ap(ap50, ap75=0.05, m=0.1):
    op = {"recall": 0.5, "precision": 0.5}
    return {"ap50": ap50, "ap75": ap75, "map": m, "mean_best_iou": 0.5, "n_images": 4, "loss": 2.5, "operating_points": {"0.9": op, "0.5": op}, "baseline": "stand-in", "prior_box_normalised": [0.1, 0.1, 0.5, 0.5], "policy": "stand-in", "verdict": "measured-small-sample", "definitions": {}}


def test_ttd_m3_a_competitive_prior_is_recorded_and_does_not_stop_the_notebook(notebook, tmp_path, monkeypatch):
    """Sections 6 and 8 executed with stand-ins where the fixed-box prior beats both policies (the outcome the notebook
    itself anticipates for BYOD): both cells complete and record the verdicts (stand-in evidence, no model)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    scores = iter([_ap(0.2), _ap(0.1, m=0.08), _ap(0.12)])

    class StandIn:
        def evaluate_zero_shot(self, records):
            return _ap(0.05)

        def adapt(self, *a, **k):
            return {"policy": "frozen backbone, encoder and decoder + new heads", "history": [{"val": None}], "trainable_names": [], "head_final_loss": 1.0}

        def evaluate(self, records):
            return next(scores)

    ns = {
        "pipe": StandIn(), "train_records": [], "val_records": [], "test_records": [], "time": __import__("time"), "json": json,
        "prior_baseline": lambda *a, **k: _ap(0.3), "DETECTION_THRESHOLD": 0.9, "POLICY_FROZEN": "frozen backbone, encoder and decoder + new heads",
        "MODEL_ID": "stand-in", "MODEL_REVISION": "0" * 40, "MODEL_KEY": "stand-in", "data_source": "stand-in", "dataset_manifests": {"test": {"digest": "d"}},
        "disjoint": {}, "summary": {}, "report": {}, "adapt_result": {"policy": "unfrozen last 2 decoder layers + new heads", "history": []}, "adapt_seconds": 0.0,
    }
    exec(_cell(notebook, "prior = prior_baseline("), ns)
    assert ns["frozen_verdict"].startswith("frozen policy NOT above the fixed-box prior")
    exec(_cell(notebook, "adapted_test = pipe.evaluate("), ns)
    verdicts = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))["comparison"]["verdicts"]
    assert verdicts["selected_above_prior_ap50"] is False and verdicts["selected_vs_frozen_map"] == "worse"


def test_ttd_m3_policy_identity_stays_a_hard_check(notebook):
    source = _cell(notebook, "prior = prior_baseline(")
    assert "if probe_result['policy'] != POLICY_FROZEN:" in source and "raise RuntimeError" in source


# --- TTD-M4: the prose does not assert a direction the release run contradicts --------------------------------


def test_ttd_m4_unfreeze_prose_names_its_runs_and_asserts_no_direction(notebook):
    markdown = _markdown(notebook)
    for stale in ("a doubling at 0.75", "the unfreeze mostly tightened boxes", "frozen features already locate nutrition tables once the heads know what to look for. About"):
        assert stale not in markdown, stale
    assert "31.79 %" in markdown and "33.15 %" in markdown and "12.88 %" in markdown
    assert "Kaggle T4 release run" in markdown and "build record" in markdown
    assert "a selection criterion (validation loss) and a held-out metric can disagree" in markdown


# --- TTD-M5: guided layer and infrastructure labelling ----------------------------------------------------------


def test_ttd_m5_guided_layer_is_present(notebook):
    markdown = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "## 10. Change one thing", "**Learner:**"):
        assert heading in markdown, heading
    assert markdown.count("**Predict") >= 7
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 7
    assert markdown.count("**What to notice:**") >= 6


def test_ttd_m5_infrastructure_cells_are_labelled_and_collapsed(notebook):
    infra = [c for c in _code_cells(notebook) if c["metadata"].get("cellView") == "form"]
    assert len([c for c in infra if c["metadata"].get("dimer", {}).get("embedded_module")]) == 3
    titled = [c["source"].splitlines()[0] for c in infra if not c["metadata"].get("dimer")]
    assert len(titled) == 3 and all(t.startswith("# @title Infrastructure:") for t in titled), titled


def test_ttd_m5_no_template_placeholders_leak(notebook):
    learner = "\n".join(c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in learner, leftover
    assert "}}" not in _markdown(notebook)


# --- TTD-m1 / TTD-m2: BYOD contract and the opening cells ------------------------------------------------------


def _jpeg(i: int, orientation: int | None = None) -> bytes:
    from PIL import Image

    image = Image.fromarray(np.random.default_rng(i).integers(0, 255, (120, 160, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    if orientation is None:
        image.save(buffer, "JPEG")
    else:
        exif = image.getexif()
        exif[0x0112] = orientation
        image.save(buffer, "JPEG", exif=exif)
    return buffer.getvalue()


def _zip(path: Path, n: int, *, drop: str | None = None, orientation: dict[int, int] | None = None, extra: dict[str, bytes] | None = None) -> Path:
    rows = "id,file,group,x_min,y_min,x_max,y_max\n" + "".join(f"p{i},img{i}.jpg,g{i},10,10,80,60\n" for i in range(n))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("boxes.csv", rows)
        for i in range(n):
            if f"img{i}.jpg" != drop:
                archive.writestr(f"img{i}.jpg", _jpeg(i, (orientation or {}).get(i)))
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return path


def test_ttd_m1_stated_minimum_is_what_the_split_accepts(tmp_path):
    assert sm.min_byod_records() == {"total": 12, "train": 8, "validation": 2, "test": 2}
    split = sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "ok.zip", 12)), seed=42)
    assert {k: len(v) for k, v in split.items()} == {"test": 2, "validation": 2, "train": 8}
    with pytest.raises(ValueError, match=r"split leaves 7 training records from 11 group\(s\).*supply at least 12 photographs"):
        sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "small.zip", 11)), seed=42)


def test_ttd_m1_missing_image_exif_and_litter(tmp_path):
    with pytest.raises(ValueError, match=r"boxes.csv line 5 \(file 'img3.jpg'\): names an image that is not in the dataset"):
        sm.load_byod_dataset(_zip(tmp_path / "missing.zip", 14, drop="img3.jpg"))
    with pytest.raises(ValueError, match=r"boxes.csv line 4 \(file 'img2.jpg'\): the image carries EXIF orientation 6"):
        sm.load_byod_dataset(_zip(tmp_path / "exif.zip", 14, orientation={2: 6}))
    assert len(sm.load_byod_dataset(_zip(tmp_path / "mac.zip", 14, extra={"__MACOSX/._img0.jpg": b"\0"}))) == 14
    with zipfile.ZipFile(tmp_path / "header.zip", "w") as archive:
        archive.writestr("boxes.csv", "id,file,group,x_min,y_min,x_max,y_max\n")
    with pytest.raises(ValueError, match="no data rows"):
        sm.load_byod_dataset(tmp_path / "header.zip")


def _section_4(notebook: dict, path: str) -> str:
    source = _cell(notebook, "USE_BYOD = False")
    source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    return source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param", 1)


def _section_4_namespace(restored: list) -> dict:
    from table_transformer_detection_pipeline import metrics as mt
    from table_transformer_detection_pipeline import pipeline as pl

    ns = {k: getattr(pl, k) for k in dir(pl) if not k.startswith("__")}
    ns.update({k: getattr(mt, k) for k in dir(mt) if not k.startswith("__")})
    ns.update({k: getattr(sm, k) for k in dir(sm) if not k.startswith("__")})
    pipe = types.SimpleNamespace(adapter={"policy": "x"}, restore_base=lambda: restored.append(True) or ["a"])
    ns.update({"os": __import__("os"), "Path": Path, "pipe": pipe, "__name__": "__main__"})
    return ns


def test_ttd_m1_byod_path_runs_section_4_outside_colab_from_the_base(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _zip(tmp_path / "mine.zip", 14)
    restored: list = []
    ns = _section_4_namespace(restored)
    exec(_section_4(notebook, "mine.zip"), ns)
    out = capsys.readouterr().out
    assert restored == [True], "a BYOD re-run must put the pipeline back to the pinned base first"
    assert ns["raw_count"] == {"byod": 14, "duplicate_images_dropped": 0, "effective_minimum": 12}
    assert "only 3 held-out test photographs" in out


def test_ttd_m1_upload_outside_colab_cancelled_and_bad_path_are_explained(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(_section_4(notebook, ""), _section_4_namespace([]))
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nowhere.zip' does not exist"):
        exec(_section_4(notebook, "nowhere.zip"), _section_4_namespace([]))
    for uploaded, message in (({}, "received 0"), ({"a.zip": b"", "b.zip": b""}, "received 2")):
        google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
        files.upload = lambda uploaded=uploaded: uploaded
        colab.files, google.colab = files, colab
        for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
            monkeypatch.setitem(sys.modules, name, module)
        with pytest.raises(ValueError, match=message):
            exec(_section_4(notebook, ""), _section_4_namespace([]))


def test_ttd_m2_no_escaped_braces_and_both_hosts_are_named(notebook):
    markdown = _markdown(notebook)
    assert "{{" not in markdown and "}}" not in markdown
    assert "`[A-Za-z0-9_.:-]{1,64}`" in markdown and "`{id, image, boxes}`" in markdown
    access = next(line for line in markdown.splitlines() if line.startswith("- **External access:**"))
    assert "Hugging Face Hub" in access and "static.openfoodfacts.org" in access and "No GitHub access" not in markdown
