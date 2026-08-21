"""Ablation lanjutan untuk event-based: degradation trend, cumulative
physical usage, jendela corrective tambahan, device/terminal context
(dari schema `analytics`, DIFLAG jelas - lihat bagian bawah), dan concept
drift (jendela tahun TRAIN). Semua keputusan dari VALIDATION t0-only
(SEBANDING dengan model statis - lihat evaluate.py), TEST hanya laporan
akhir. TIDAK mengubah train.py/artifacts produksi event-based - itu
dijalankan terpisah SETELAH konfigurasi final diketahui dari sini, sama
seperti pola survival_model/experiments.py.

    python survival_model/event_based/experiments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
SURVIVAL_DIR = Path(__file__).resolve().parent.parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))
EVENT_BASED_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVENT_BASED_DIR))  # lihat catatan build_dataset.py/train.py

import numpy as np
import pandas as pd

import data_reader

from src import categorical_support, evaluation, model_fit

import build_dataset
from eb_src import dynamic_history, features

REPORTS_DIR = EVENT_BASED_DIR / "reports"


# ---------------------------------------------------------------------------
# Device/terminal context - schema `analytics` (BUKAN data_reader.py, BUKAN
# live production - lihat config.py: "supaya production tidak bergantung
# pada tabel di schema analytics"). Dipakai HANYA untuk ablation eksperimen
# ini, dengan flag point-in-time yang TERSEDIA di sumbernya sendiri
# (`parent_link_quality_status`) - relasi INI TIDAK dibuat baru di sini,
# hanya dipakai APA ADANYA dari hasil riset sebelumnya yang SUDAH
# mem-verifikasi validitasnya. Kalau terbukti membantu VALIDATION, perlu
# keputusan terpisah: reproduksi query kanonikal ini ke data_reader.py
# (supaya tidak bergantung pada schema analytics), BUKAN dipakai permanen
# lewat schema ini.
# ---------------------------------------------------------------------------


def load_terminal_context() -> pd.DataFrame:
    """Satu baris per installation_cycle_id: terminal_type/terminal_model_code
    pada observasi PALING AWAL cycle itu (point-in-time paling ketat) +
    apakah link-nya VALID_POINT_IN_TIME_RELATION saat itu. Cycle yang
    relasinya baru "direkam setelah instalasi" (43% populasi) diberi
    UNKNOWN_LABEL - bukan dianggap tahu sesuatu yang sebenarnya baru
    diketahui belakangan."""
    with data_reader.connect() as conn:
        df = pd.read_sql(
            """
            SELECT DISTINCT ON (installation_cycle_id)
                installation_cycle_id, terminal_type, terminal_model_code, parent_link_quality_status
            FROM analytics.eda_item_observation_30d_hierarchy
            ORDER BY installation_cycle_id, observation_on ASC
            """,
            conn,
        )
    safe = df["parent_link_quality_status"].eq("VALID_POINT_IN_TIME_RELATION")
    df["terminal_type_safe"] = np.where(safe, df["terminal_type"].fillna("UNKNOWN"), "UNKNOWN")
    df["terminal_model_safe"] = np.where(safe, df["terminal_model_code"].fillna("UNKNOWN"), "UNKNOWN")
    return df[["installation_cycle_id", "terminal_type_safe", "terminal_model_safe"]]


# ---------------------------------------------------------------------------
# Fit + evaluasi ringkas: TRAIN (full landmark) -> VAL full & VAL t0-only.
# t0-only ADALAH metrik yang sah dibandingkan dengan model statis (evaluate.py).
# ---------------------------------------------------------------------------


def fit_eval(
    label: str, feature_frame: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str],
    dataset: pd.DataFrame, model_names: list[str] | None = None,
) -> dict:
    train_mask = (dataset["split"] == "TRAIN").to_numpy()
    val_mask = (dataset["split"] == "VALIDATION").to_numpy()
    t0_mask = (dataset["landmark_source"] == "INSTALL").to_numpy()
    val_t0_mask = val_mask & t0_mask

    encoder = features.fit_encoder(feature_frame.loc[train_mask], categorical_cols)
    x_train = features.encode(feature_frame.loc[train_mask], encoder, numeric_cols)
    x_val = features.encode(feature_frame.loc[val_mask], encoder, numeric_cols)
    x_val_t0 = features.encode(feature_frame.loc[val_t0_mask], encoder, numeric_cols)
    y_train = model_fit.make_survival_target(dataset, train_mask)
    y_val = model_fit.make_survival_target(dataset, val_mask)
    y_val_t0 = model_fit.make_survival_target(dataset, val_t0_mask)

    names = model_names if model_names is not None else ["random_survival_forest", "cox_ph"]
    overrides = {}
    for name in names:
        base = dict(model_fit.MODEL_REGISTRY[name]["default_params"])
        if "n_jobs" in base:
            base["n_jobs"] = 1  # lihat catatan survival_model/experiments.py soal loky hang
        overrides[name] = base
    models = model_fit.fit_models(x_train, y_train, names, overrides)

    rows = []
    for name, model in models.items():
        risk_sign = model_fit.MODEL_REGISTRY.get(name, {}).get("risk_sign", 1)
        full_m = evaluation.native_metrics(model, y_train, x_val, y_val, risk_sign=risk_sign)
        t0_m = evaluation.native_metrics(model, y_train, x_val_t0, y_val_t0, risk_sign=risk_sign)
        # AUC time-dependent 30/90 hari (t0-only) - proxy MURAH untuk
        # Recall@kapasitas operasional (tidak perlu membangun ulang populasi
        # TEST classification 1,4 juta baris seperti Lapis 2 evaluate.py) -
        # sudah dihitung native_metrics() untuk IBS/Brier, sebelumnya TIDAK
        # pernah dilaporkan. Dipakai sebagai kriteria pemilihan TAMBAHAN
        # (bukan pengganti) mulai sesi "stop kejar C-index, jaga jangan turun".
        auc30 = t0_m["time_dependent_auc_at_horizon"].get(30)
        auc90 = t0_m["time_dependent_auc_at_horizon"].get(90)
        rows.append({
            "label": label, "model": name,
            "val_full_c_index": full_m["c_index"], "val_t0_c_index": t0_m["c_index"],
            "val_t0_ibs": t0_m["integrated_brier_score"],
            "val_t0_auc30": auc30, "val_t0_auc90": auc90,
        })
        auc30_str = f"{auc30:.4f}" if auc30 is not None else "N/A"
        print(
            f"      {label:28s} {name:24s} VAL-full={full_m['c_index']:.4f}  "
            f"VAL-t0={t0_m['c_index']:.4f}  AUC30={auc30_str}"
        )
    return {"rows": rows, "models": models, "encoder": encoder}


def render_table(rows: list[dict]) -> str:
    header = (
        "| Experiment | Model | VAL C-index (full) | VAL C-index (t0-only, ADIL) | VAL t0 IBS | "
        "VAL t0 AUC-30d | VAL t0 AUC-90d |"
    )
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "N/A"
        lines.append(
            f"| {r['label']} | {r['model']} | {fmt(r['val_full_c_index'])} | "
            f"{fmt(r['val_t0_c_index'])} | {fmt(r['val_t0_ibs'])} | "
            f"{fmt(r.get('val_t0_auc30'))} | {fmt(r.get('val_t0_auc90'))} |"
        )
    return "\n".join(lines)


def build_configs(built: dict) -> dict[str, dict]:
    """Bangun SEMUA konfigurasi ablation (A..F) dari hasil `build_dataset.build()`.
    Diekstrak dari main() supaya skrip lanjutan (GBSA, concept drift, tuning
    pada F_combined_all - lihat experiments_round2.py) bisa memakai fitur
    yang SAMA PERSIS tanpa menyalin ulang logic konstruksinya."""
    dataset, base_features = built["dataset"], built["features"]
    landmarks, cycles, events = built["landmarks"], built["cycles"], built["events"]

    print("[2/5] Menghitung fitur dynamic tambahan (degradation trend, cumulative usage, jendela corrective)...")
    cum = dynamic_history.cumulative_cycle_age(cycles)
    landmarks_with_cum = landmarks.merge(cum, on="installation_cycle_id", how="left")
    physical_age_now = (
        landmarks_with_cum["cumulative_prior_cycle_days"].to_numpy()
        + landmarks_with_cum["landmark_age_days"].to_numpy()
    )
    cumulative_cols = pd.DataFrame(index=landmarks.index)
    cumulative_cols["log_cumulative_prior_cycle_days"] = np.log1p(landmarks_with_cum["cumulative_prior_cycle_days"].to_numpy())
    cumulative_cols["log_physical_age_now"] = np.log1p(np.clip(physical_age_now, 0, None))
    cumulative_cols["previous_cycle_count"] = landmarks_with_cum["previous_cycle_count"].to_numpy(dtype=float)

    trend_cols = dynamic_history.corrective_degradation_trend(landmarks, events)
    windowed_cols = dynamic_history.windowed_corrective_extra(landmarks, events)

    print("[3/5] Mengambil device/terminal context (schema analytics, point-in-time filtered)...")
    terminal_raw = load_terminal_context()
    landmarks_with_terminal = landmarks.merge(terminal_raw, on="installation_cycle_id", how="left")
    landmarks_with_terminal["terminal_type_safe"] = landmarks_with_terminal["terminal_type_safe"].fillna("UNKNOWN")
    terminal_support = categorical_support.cumulative_support(
        landmarks_with_terminal.assign(observation_on=landmarks["observation_on"]),
        "terminal_type_safe", "observation_on",
    )
    device_cols = pd.DataFrame(index=landmarks.index)
    device_cols["terminal_type_grouped"] = categorical_support.apply_threshold(
        landmarks_with_terminal["terminal_type_safe"], terminal_support, 200
    ).to_numpy()

    base_categorical = features.CATEGORICAL_FEATURES
    base_numeric = features.NUMERIC_FEATURES + features.FLEET_FEATURES

    configs: dict[str, dict] = {}
    configs["A_t0_baseline"] = dict(
        feature_frame=base_features, categorical_cols=base_categorical, numeric_cols=base_numeric,
    )
    configs["B_plus_degradation_trend"] = dict(
        feature_frame=pd.concat([base_features, trend_cols], axis=1),
        categorical_cols=base_categorical,
        numeric_cols=base_numeric + [c for c in trend_cols.columns if c != "has_failure_interval_trend"] + ["has_failure_interval_trend"],
    )
    configs["C_plus_cumulative_history"] = dict(
        feature_frame=pd.concat([base_features, cumulative_cols], axis=1),
        categorical_cols=base_categorical,
        numeric_cols=base_numeric + list(cumulative_cols.columns),
    )
    configs["D_plus_windowed_corrective"] = dict(
        feature_frame=pd.concat([base_features, windowed_cols], axis=1),
        categorical_cols=base_categorical,
        numeric_cols=base_numeric + list(windowed_cols.columns),
    )
    configs["E_plus_device_terminal"] = dict(
        feature_frame=pd.concat([base_features, device_cols], axis=1),
        categorical_cols=base_categorical + ["terminal_type_grouped"],
        numeric_cols=base_numeric,
    )
    configs["F_combined_all"] = dict(
        feature_frame=pd.concat([base_features, trend_cols, cumulative_cols, windowed_cols, device_cols], axis=1),
        categorical_cols=base_categorical + ["terminal_type_grouped"],
        numeric_cols=(
            base_numeric + [c for c in trend_cols.columns if c != "has_failure_interval_trend"]
            + ["has_failure_interval_trend"] + list(cumulative_cols.columns) + list(windowed_cols.columns)
        ),
    )
    # G: SAMA seperti F, TAPI TANPA device/terminal (skema `analytics`,
    # BUKAN live production - lihat config.py) - kalau nilainya dekat
    # dengan F, fitur produksi FINAL tidak perlu bergantung pada schema itu
    # sama sekali. Keputusan penting untuk train.py, bukan cuma ablation.
    configs["G_combined_without_device"] = dict(
        feature_frame=pd.concat([base_features, trend_cols, cumulative_cols, windowed_cols], axis=1),
        categorical_cols=base_categorical,
        numeric_cols=(
            base_numeric + [c for c in trend_cols.columns if c != "has_failure_interval_trend"]
            + ["has_failure_interval_trend"] + list(cumulative_cols.columns) + list(windowed_cols.columns)
        ),
    )
    return configs


def main() -> int:
    print("[1/5] Memuat dataset event-based (cache kalau SURVIVAL_BUILD_CACHE=1)...")
    built = build_dataset.build()
    dataset = built["dataset"]

    configs = build_configs(built)

    print("[4/5] Fit + evaluasi tiap konfigurasi (RSF + Cox, VAL full & t0-only)...")
    rows: list[dict] = []
    results: dict[str, dict] = {}
    for label, cfg in configs.items():
        result = fit_eval(label, cfg["feature_frame"], cfg["categorical_cols"], cfg["numeric_cols"], dataset)
        results[label] = result
        rows.extend(result["rows"])

    print("[5/5] Menulis laporan...")
    report = ["# Ablation lanjutan event-based: dynamic history + device/terminal (Fase 2)", ""]
    report.append(
        "Semua konfigurasi ditambahkan DI ATAS A_t0_baseline (fitur event-based final saat ini, VAL t0-only "
        "0,7849 - lihat reports/evaluation_report.md). Keputusan dari **VAL t0-only** (kolom ke-4, SEBANDING "
        "dengan C-index model statis) - VAL full (kolom ke-3) TIDAK dipakai memilih (repeated measures, lihat "
        "README). E_plus_device_terminal memakai schema `analytics` (riset lama, BUKAN live production - lihat "
        "config.py) dengan filter `parent_link_quality_status=='VALID_POINT_IN_TIME_RELATION'` di observasi "
        "PALING AWAL tiap cycle - cycle yang relasinya baru diketahui SETELAH instalasi diberi UNKNOWN, bukan "
        "diam-diam dipakai."
    )
    report.append("")
    report.append(render_table(rows))
    (REPORTS_DIR / "dynamic_ablation.md").write_text("\n".join(report), encoding="utf-8")
    print(f"      Laporan: {REPORTS_DIR / 'dynamic_ablation.md'}")

    best_label = max(
        configs, key=lambda lbl: next(r["val_t0_c_index"] for r in results[lbl]["rows"] if r["model"] == "random_survival_forest")
    )
    print(f"\n[SELESAI] Konfigurasi terbaik (VAL t0-only, RSF): {best_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
