"""Evaluasi model survival event-based - TIGA lapis (bukan dua seperti model
statis, lihat "Lapis 1b" di bawah untuk alasannya).

    python survival_model/event_based/evaluate.py

Lapis 1 - native FULL LANDMARK: C-index dkk dihitung dari SEMUA baris
landmark (satu lifecycle bisa menyumbang beberapa baris). Ini metrik
OPERASIONAL model event-based yang sebenarnya - tapi BUKAN apples-to-apples
dengan C-index model statis (survival_model/), karena baris-baris landmark
satu lifecycle SALING BERKORELASI (repeated measures) - C-index yang dihitung
naif di atasnya BISA bias optimis (lihat README bagian "Keterbatasan").

Lapis 1b - native T0-ONLY (BARU, TIDAK ada di model statis): subset HANYA
baris landmark_source=='INSTALL' (SATU baris per lifecycle, age=0) - populasi
IDENTIK dengan yang dipakai C-index model statis. Ini perbandingan HEAD-TO-HEAD
yang adil: "kalau model event-based ini HANYA diberi info instalasi (sama
seperti model statis), seberapa baik dia?" - mengisolasi efek ARSITEKTUR
(landmark training) dari efek INFORMASI TAMBAHAN (banyak landmark per
lifecycle) pada C-index akhir.

Lapis 2 - operasional vs classification production (SAMA metodologinya
dengan model statis: `training_utils.full_metrics()` pada populasi TEST
classification yang dipinjam read-only) - dijalankan pada fitur T0-ONLY
(satu baris per lifecycle, index=installation_cycle_id) supaya
`evaluate.score_operational()` (fungsi ASLI model statis, TIDAK diubah)
bisa dipakai APA ADANYA - fungsi itu butuh index unik per lifecycle.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SURVIVAL_DIR = Path(__file__).resolve().parent.parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))
EVENT_BASED_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVENT_BASED_DIR))  # lihat catatan di build_dataset.py/train.py

import joblib

from partrisk import config

from src import evaluation, model_fit

import build_dataset
from eb_src import features


def _load_static_evaluate():
    """Muat survival_model/evaluate.py (parent) lewat file path, BUKAN
    `import evaluate` biasa - modul INI SENDIRI juga bernama evaluate.py
    (event_based/evaluate.py), `import evaluate` akan meng-import DIRINYA
    SENDIRI (self-import) alih-alih versi statis. `train.py`/`predict.py` di
    root TIDAK butuh pola ini lagi setelah restrukturisasi src/partrisk/ -
    keduanya sekarang punya nama qualified unik (`partrisk.train`,
    `partrisk.predict`), tidak lagi bertabrakan nama bare dengan
    survival_model/train.py. survival_model/evaluate.py di sini TETAP
    bertabrakan nama dengan file ini sendiri, jadi trik file-path TETAP
    perlu - cuma untuk kasus INI."""
    spec = importlib.util.spec_from_file_location("static_survival_evaluate", SURVIVAL_DIR / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["static_survival_evaluate"] = module
    spec.loader.exec_module(module)
    return module


static_evaluate = _load_static_evaluate()

ARTIFACTS_DIR = EVENT_BASED_DIR / "artifacts"
REPORTS_DIR = EVENT_BASED_DIR / "reports"


def load_artifacts() -> dict:
    models = joblib.load(ARTIFACTS_DIR / "models.joblib")
    for model in models.values():
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1  # alasan sama seperti survival_model/evaluate.py
    encoder = joblib.load(ARTIFACTS_DIR / "encoder.joblib")
    y_train = joblib.load(ARTIFACTS_DIR / "y_train.joblib")
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    return {"models": models, "encoder": encoder, "y_train": y_train, "metadata": metadata}


def _native_by_split(models, encoder, y_train, dataset, feature_frame, mask_by_split) -> dict:
    results: dict = {}
    for split_name, mask in mask_by_split.items():
        y_eval = model_fit.make_survival_target(dataset, mask)
        x_eval = features.encode(feature_frame.loc[mask], encoder)
        results[split_name] = {
            name: evaluation.native_metrics(
                model, y_train, x_eval, y_eval, risk_sign=model_fit.MODEL_REGISTRY.get(name, {}).get("risk_sign", 1)
            )
            for name, model in models.items()
        }
    return results


def main() -> int:
    print("[1/4] Memuat artifacts hasil train.py...")
    artifacts = load_artifacts()
    models, encoder, y_train = artifacts["models"], artifacts["encoder"], artifacts["y_train"]

    print("[2/4] Menyusun dataset (cache kalau SURVIVAL_BUILD_CACHE=1)...")
    built = build_dataset.build()
    dataset, feature_frame = built["dataset"], built["features"]
    masks = {name: (dataset["split"] == name).to_numpy() for name in ("VALIDATION", "TEST")}

    print("[3/4] Lapis 1 (full landmark) & Lapis 1b (t0-only, adil vs model statis)...")
    native_full = _native_by_split(models, encoder, y_train, dataset, feature_frame, masks)
    for split_name, per_model in native_full.items():
        for model_name, m in per_model.items():
            print(f"      [full]    {split_name:10s} {model_name:24s} C-index={m['c_index']:.4f}")

    t0_mask_global = (dataset["landmark_source"] == "INSTALL").to_numpy()
    t0_masks = {name: masks[name] & t0_mask_global for name in masks}
    # y_train UNTUK IPCW/Uno-C tetap dataset TRAIN PENUH (semua landmark) -
    # y_train dipakai HANYA untuk estimasi model censoring (IPCW), bukan
    # sebagai populasi yang dievaluasi - konsisten dengan cara model statis
    # memakai y_train yang sama untuk VALIDATION dan TEST.
    native_t0 = _native_by_split(models, encoder, y_train, dataset, feature_frame, t0_masks)
    for split_name, per_model in native_t0.items():
        for model_name, m in per_model.items():
            print(f"      [t0-only] {split_name:10s} {model_name:24s} C-index={m['c_index']:.4f}")

    print("[4/4] Lapis 2: perbandingan adil dengan classification model production (fitur t0-only)...")
    test_rows, window_days = static_evaluate.load_classification_test_rows()
    t0_feature_frame = feature_frame.loc[t0_mask_global].copy()
    t0_feature_frame.index = dataset.loc[t0_mask_global, "installation_cycle_id"].to_numpy()
    # numeric_columns HARUS diisi eksplisit: static_evaluate.score_operational()
    # secara default memakai daftar kolom numerik milik features.py STATIS
    # (survival_model/src/features.py, 14 kolom - TIDAK sama dengan kolom
    # event-based di sini, mis. log_days_since_installation ada di sini tapi
    # tidak di statis) - tanpa override ini akan salah kolom/KeyError.
    eb_numeric_columns = features.NUMERIC_FEATURES + features.FLEET_FEATURES
    operational = {
        name: static_evaluate.score_operational(
            model, t0_feature_frame, encoder, test_rows, window_days, numeric_columns=eb_numeric_columns
        )
        for name, model in models.items()
    }
    for name, m in operational.items():
        if m is None:
            print(f"      {name:24s} (tidak ada baris TEST classification yang cocok)")
            continue
        print(
            f"      {name:24s} PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
            f"Recall@cap={m['recall_at_capacity']:.4f}  Precision@cap={m['precision_at_capacity']:.4f}"
        )

    report = _format_report(native_full, native_t0, operational, dataset, window_days)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "evaluation_report.md").write_text(report, encoding="utf-8")
    print(f"\n[OK] Laporan tersimpan di {REPORTS_DIR / 'evaluation_report.md'}")
    return 0


def _format_report(native_full, native_t0, operational, dataset, window_days) -> str:
    lines = ["# Laporan evaluasi event-based survival (Tahap 6-9)", ""]
    lines.append(
        "Tiga lapis - lihat docstring evaluate.py untuk definisi lengkap tiap lapis dan kenapa "
        "dipisah. **Lapis 1b (t0-only) adalah angka yang SEBANDING dengan C-index model statis** "
        "(survival_model/reports/evaluation_report.md) - Lapis 1 (full landmark) TIDAK sebanding "
        "langsung (repeated measures per lifecycle)."
    )
    lines.append("")
    lines.append("## Lapis 1 - native, SEMUA baris landmark (bukan perbandingan apples-to-apples)")
    for split_name, per_model in native_full.items():
        lines.append(f"\n### {split_name}")
        for model_name, m in per_model.items():
            lines.append(
                f"- **{model_name}**: rows={m['rows']:,} events={m['events']:,} "
                f"C-index(Harrell)={m['c_index']:.4f} IBS={m['integrated_brier_score']}"
            )

    lines.append("\n## Lapis 1b - native, T0-ONLY (satu baris/lifecycle, SEBANDING dengan model statis)")
    for split_name, per_model in native_t0.items():
        lines.append(f"\n### {split_name}")
        for model_name, m in per_model.items():
            lines.append(
                f"- **{model_name}**: rows={m['rows']:,} events={m['events']:,} "
                f"C-index(Harrell)={m['c_index']:.4f} IBS={m['integrated_brier_score']}"
            )

    lines.append("\n## Lapis 2 - perbandingan adil vs classification model (fitur t0-only, populasi TEST classification)")
    lines.append(f"\nWindow {window_days:.0f} hari, kapasitas {config.FAILURE_CAPACITY_PER_MONTH}/bulan.")
    for name, m in operational.items():
        if m is None:
            lines.append(f"- {name}: tidak ada baris TEST classification yang cocok")
            continue
        lines.append(
            f"- **{name}**: PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
            f"Recall@cap={m['recall_at_capacity']:.4f} Precision@cap={m['precision_at_capacity']:.4f} "
            f"Brier={m['brier_calibrated']:.4f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
