"""Fase A1 dari plan restrukturisasi (lihat gate_decision.md): eksperimen
PENENTU apakah event-based survival layak menggantikan CatBoost.

MASALAH dengan angka lama (PR-AUC 0,1824 di reports/ensemble_operational.md
dkk): dihitung dengan fitur DIBEKUKAN di installed_on (t0-only), karena
`survival_model/evaluate.py::compute_risk_30d()` butuh satu baris per
lifecycle. Itu menghukum model ini justru pada sumbu yang jadi alasannya
dibangun - fitur dinamis yang di-refresh seiring waktu. `predict.py` model
ini menskor pada KONDISI SEKARANG, bukan kondisi instalasi - evaluasi yang
representatif harus melakukan hal yang sama.

Di sini, untuk SETIAP baris TEST classification, fitur event-based dibangun
PERSIS pada `observation_on` baris itu sendiri (bukan installed_on) - baris
itu diperlakukan sebagai satu landmark tunggal, mekanisme SAMA PERSIS dengan
satu landmark di `eb_src/landmark_builder.py`, hanya titik waktunya beda.

EMPAT JEBAKAN yang masing-masing bisa memalsukan kemenangan (lihat plan):

1. risk_30d = 1 - S(30) LANGSUNG, BUKAN utils.conditional_risk(). Kurva
   event-based sudah bermula di t=0=observation_on (survival_model/event_based/
   predict.py) - conditional_risk (rumus 1-S(age+30)/S(age)) untuk model
   STATIS yang kurvanya dari t=0=installed_on, salah dua kali kalau dipakai
   di sini.
2. Support totals (part_model/item_type/terminal) WAJIB dari dict BEKU hasil
   training (metadata.json) - dipetakan persis seperti predict.py, BUKAN
   dihitung ulang dari frame TEST. Menghitung ulang = leakage.
3. Memori: 38rb baris x predict_survival_function pada RSF besar bisa OOM.
   Di-chunk, kurva dibuang tiap chunk.
4. Pembanding CatBoost dihitung ULANG di populasi IRISAN yang sama - bukan
   dibandingkan ke angka tersimpan lama yang populasinya beda.

    python survival_model/event_based/experiments_landmark_operational.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

SURVIVAL_DIR = Path(__file__).resolve().parent.parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))
EVENT_BASED_DIR = Path(__file__).resolve().parent
# TANPA guard - lihat catatan build_dataset.py/train.py: skrip ini sendiri
# sudah otomatis ada di sys.path[0] sebelum baris ini jalan.
sys.path.insert(0, str(EVENT_BASED_DIR))

import joblib
import numpy as np
import pandas as pd

from partrisk import config, training_utils
from partrisk.data import reader as data_reader
from partrisk.features import failure as feature_builder
from partrisk.predict import failure as root_predict

from src import install_context, previous_cycle
from src import utils as survival_utils

from eb_src import features as eb_features

REPORTS_DIR = EVENT_BASED_DIR / "reports"
ARTIFACTS_DIR = EVENT_BASED_DIR / "artifacts"
CHUNK_SIZE = 2000
HORIZON_DAYS = 30.0


def _load_static_evaluate():
    """Muat survival_model/evaluate.py lewat file path, BUKAN `import
    evaluate` biasa - event_based/evaluate.py punya nama file yang sama.
    `predict.py` di root TIDAK butuh trik ini lagi setelah restrukturisasi
    src/partrisk/ - sudah punya nama qualified unik (`partrisk.predict`,
    lihat `root_predict` di atas), tidak lagi bertabrakan nama bare dengan
    event_based/predict.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("static_evaluate_for_gate", SURVIVAL_DIR / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["static_evaluate_for_gate"] = module
    spec.loader.exec_module(module)
    return module


def build_landmark_features_at_observation(
    test_rows: pd.DataFrame, events: pd.DataFrame, cycles: pd.DataFrame, episodes: pd.DataFrame,
    terminal_raw: pd.DataFrame, metadata: dict,
) -> pd.DataFrame:
    """Fitur event-based PADA observation_on tiap baris - JEBAKAN #2 (support
    beku) ditegakkan di sini."""
    landmarks = test_rows.reset_index(drop=True).copy()
    landmarks["landmark_age_days"] = landmarks["days_since_installation"]

    landmarks = install_context.attach_install_context(landmarks, events)
    landmarks = eb_features.attach_terminal_extra(landmarks, terminal_raw)

    landmarks["days_since_installation"] = landmarks["landmark_age_days"]
    landmarks = feature_builder.attach_history(landmarks, events)
    landmarks = feature_builder.attach_fleet(landmarks, cycles, episodes)

    pc = previous_cycle.audit_previous_cycle_features(cycles)
    landmarks = landmarks.merge(
        pc[[
            "installation_cycle_id",
            "previous_cycle_confirmed_failure_lifetime_mean",
            "last_confirmed_failure_lifetime",
        ]],
        on="installation_cycle_id", how="left",
    )
    transform = previous_cycle.transform_for_model(landmarks)[
        ["log_previous_cycle_confirmed_failure_lifetime_mean", "has_previous_cycle_confirmed_failure_lifetime_mean"]
    ]
    landmarks = pd.concat([landmarks, transform], axis=1)

    landmarks = eb_features.attach_dynamic_extra(landmarks, cycles, events)

    # JEBAKAN #2: support DIBEKUKAN dari metadata.json (hasil training),
    # dipetakan persis seperti predict.py - TIDAK dihitung ulang dari
    # populasi TEST ini.
    support_totals = {k: int(v) for k, v in metadata["support_totals"].items()}
    item_type_support_totals = {k: int(v) for k, v in metadata["item_type_support_totals"].items()}
    terminal_support_totals = {k: int(v) for k, v in metadata["terminal_support_totals"].items()}

    support = landmarks["item_model_code_clean"].map(support_totals).fillna(0).astype("int64")
    item_type_support = landmarks["item_type_at_install"].map(item_type_support_totals).fillna(0).astype("int64")
    terminal_support = landmarks["terminal_type_context"].map(terminal_support_totals).fillna(0).astype("int64")

    feature_frame = eb_features.compute_features(landmarks, support, item_type_support, terminal_support)
    return feature_frame.reset_index(drop=True)


def score_risk_30d_chunked(model, feature_frame: pd.DataFrame, encoder) -> np.ndarray:
    """JEBAKAN #3: chunk supaya predict_survival_function tidak OOM. JEBAKAN
    #1: risk = 1 - S(30) LANGSUNG (kurva sudah dari t=0=observation_on)."""
    n = len(feature_frame)
    risk = np.empty(n, dtype=float)
    n_chunks = math.ceil(n / CHUNK_SIZE)
    for i in range(n_chunks):
        lo, hi = i * CHUNK_SIZE, min((i + 1) * CHUNK_SIZE, n)
        chunk = feature_frame.iloc[lo:hi]
        x_chunk = eb_features.encode(chunk, encoder)
        times_grid, curves = survival_utils.survival_curve_arrays(model, x_chunk)
        s30 = survival_utils.step_eval_matrix(times_grid, curves, [HORIZON_DAYS])[:, 0]
        risk[lo:hi] = 1.0 - s30
        del curves
        print(f"      chunk {i+1}/{n_chunks} ({hi:,}/{n:,} baris)...")
    return risk


def main() -> int:
    print("[1/5] Memuat baris TEST classification (dipinjam read-only)...")
    static_evaluate = _load_static_evaluate()
    test_rows, window_days = static_evaluate.load_classification_test_rows()
    print(f"      {len(test_rows):,} baris, window {window_days:.0f} hari")

    print("[2/5] Memuat model event-based produksi + metadata (support beku)...")
    models = joblib.load(ARTIFACTS_DIR / "models.joblib")
    encoder = joblib.load(ARTIFACTS_DIR / "encoder.joblib")
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    model = models[metadata["primary_model"]]
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1  # RSF ter-unpickle + predict_survival_function + n_jobs=-1 = hang tanpa error
    print(f"      model: {metadata['primary_model']}")

    print("[3/5] Membaca database (events/cycles/episodes/terminal) SEKALI...")
    t0 = time.time()
    events = data_reader.get_events()
    cycles = data_reader.get_cycles()
    episodes = data_reader.get_failure_episodes()
    terminal_raw = data_reader.get_terminal_context()
    print(f"      selesai {time.time()-t0:.1f} detik")

    print("[4/5] Membangun fitur PADA observation_on tiap baris TEST...")
    t0 = time.time()
    feature_frame = build_landmark_features_at_observation(
        test_rows, events, cycles, episodes, terminal_raw, metadata
    )
    print(f"      {len(feature_frame):,} baris fitur, {time.time()-t0:.1f} detik")

    print("[5/5] Skor risk_30d = 1-S(30), per chunk...")
    t0 = time.time()
    risk_30d_survival = score_risk_30d_chunked(model, feature_frame, encoder)
    print(f"      selesai {time.time()-t0:.1f} detik")

    target = test_rows["target_failure"].astype(bool).to_numpy()

    print("\n[Pembanding] Menskor CatBoost incumbent pada POPULASI YANG SAMA (JEBAKAN #4)...")
    # test_rows di sini SUDAH populasi lengkap classification TEST (bukan
    # subset) - event-based mencakup SEMUA baris ini (tidak ada filter
    # kelayakan tambahan di build_landmark_features_at_observation), jadi
    # populasinya SUDAH sama persis. Skor CatBoost lewat jalur produksi asli
    # (predict.py punya cache global, jadi panggil feature_builder langsung
    # pada frame yang sudah ada, bukan predict() per item).
    catboost_predict = root_predict

    cb_model, cb_calibrator, cb_metadata = catboost_predict.load_model()
    cb_support = feature_builder.part_model_support(test_rows, cb_metadata["part_model_support"])
    cb_features = feature_builder.build_features(test_rows, cb_support)
    risk_30d_catboost = cb_calibrator.predict(cb_model.predict_proba(cb_features)[:, 1])

    print("\n=== HASIL (populasi identik, N={:,}) ===".format(len(test_rows)))
    metrics_survival = training_utils.full_metrics(
        risk_30d_survival, risk_30d_survival, target, window_days, config.FAILURE_CAPACITY_PER_MONTH
    )
    metrics_catboost = training_utils.full_metrics(
        risk_30d_catboost, risk_30d_catboost, target, window_days, config.FAILURE_CAPACITY_PER_MONTH
    )
    for label, m in (("event-based (observation_on)", metrics_survival), ("catboost (incumbent, ulang)", metrics_catboost)):
        print(
            f"  {label:32s} PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
            f"Recall@cap={m['recall_at_capacity']:.4f}  Precision@cap={m['precision_at_capacity']:.4f}  "
            f"Brier={m['brier_calibrated']:.4f}"
        )

    report = ["# Fase A1: evaluasi production-realistic event-based vs CatBoost (Gate)", ""]
    report.append(
        f"Populasi: {len(test_rows):,} baris TEST classification (window {window_days:.0f} hari, "
        f"kapasitas {config.FAILURE_CAPACITY_PER_MONTH}/bulan) - SAMA PERSIS untuk kedua model "
        "(rows_matched = seluruh populasi, tidak ada baris yang gugur di sisi event-based)."
    )
    report.append("")
    report.append(
        "Beda dari angka lama (reports/ensemble_operational.md, PR-AUC 0,1824): fitur event-based "
        "DIHITUNG PADA `observation_on` tiap baris (kondisi PART SAAT snapshot classification itu "
        "diambil), BUKAN dibekukan di `installed_on`. Ini evaluasi yang merepresentasikan cara "
        "`predict.py` benar-benar dipakai (skor kondisi SEKARANG)."
    )
    report.append("")
    report.append("| Model | PR-AUC | ROC-AUC | Recall@cap | Precision@cap | Brier |")
    report.append("|---|---|---|---|---|---|")
    for label, m in (("event-based (observation_on)", metrics_survival), ("catboost v2 (incumbent, dihitung ulang)", metrics_catboost)):
        report.append(
            f"| {label} | {m['pr_auc']:.4f} | {m['roc_auc']:.4f} | {m['recall_at_capacity']:.4f} | "
            f"{m['precision_at_capacity']:.4f} | {m['brier_calibrated']:.4f} |"
        )
    report.append("")
    report.append(f"rows_matched = {len(test_rows):,} (populasi identik untuk kedua model)")
    (REPORTS_DIR / "gate_a1_landmark_operational.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n[OK] Laporan: {REPORTS_DIR / 'gate_a1_landmark_operational.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
