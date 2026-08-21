"""Audit metodologis + peningkatan survival model: installation context,
threshold kategori khusus survival, ablation A/B/C, audit previous-cycle,
dan small RSF tuning - semua keputusan dari VALIDATION, TEST hanya untuk
laporan akhir.

    python survival_model/experiments.py

Database dan populasi TEST classification (untuk perbandingan operasional)
masing-masing HANYA dibaca SEKALI di awal - seluruh eksperimen berikutnya
bekerja in-memory dari situ (lihat README bagian "Efisiensi eksperimen").

Menulis:
    reports/uncertainty_baseline.md  (langkah 0, sesi peningkatan C-index)
    reports/category_threshold.md   (langkah 3 instruksi)
    reports/feature_ablation.md      (langkah 4-5)
    reports/previous_cycle_audit.md   (langkah 6)
    reports/model_comparison.md        (langkah 8-9, RSF vs Cox + tuning)
    reports/model_family.md             (langkah tambahan, sesi peningkatan C-index)

Tidak menyimpan model kandidat ke artifacts/ - itu tetap tugas train.py,
dijalankan ulang SETELAH konfigurasi final diketahui dari sini.
"""

from __future__ import annotations

import sys
from pathlib import Path

SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

import joblib
import numpy as np
import pandas as pd

from partrisk import feature_builder

import build_dataset
import evaluate
from src import categorical_support, evaluation, features, hazard_features, install_context, model_fit, previous_cycle

REPORTS_DIR = SURVIVAL_DIR / "reports"
ARTIFACTS_DIR = SURVIVAL_DIR / "artifacts"
CACHE_DIR = SURVIVAL_DIR / "artifacts" / "_experiment_cache"
THRESHOLD_CANDIDATES = [20, 50, 100, 200, 300]
LIGHT_RSF_PARAMS = dict(
    n_estimators=50, min_samples_split=40, min_samples_leaf=30,
    max_features="sqrt", n_jobs=1, random_state=42,
)


# ---------------------------------------------------------------------------
# Persiapan data - SEKALI untuk semua eksperimen
# ---------------------------------------------------------------------------


def prepare_base_data(with_operational: bool = False) -> dict:
    """`with_operational=False` (bawaan): skip membangun populasi TEST
    classification (1,4 juta baris, langkah paling berat & paling rapuh di
    lingkungan ini - terbukti macet berkali-kali). Tidak masalah untuk tahap
    threshold/ablation/previous-cycle/tuning karena keputusannya memang
    HARUS dari VALIDATION native C-index (instruksi eksplisit), bukan
    metrik operasional. Perbandingan operasional (Lapis 2) untuk KONFIGURASI
    FINAL tetap didapat - lewat `evaluate.py` yang dijalankan terpisah
    setelah train.py diupdate, bukan di sini."""
    cache_path = CACHE_DIR / f"prepare_base_data_op{with_operational}.joblib"
    if cache_path.exists():
        print(f"[persiapan] [cache] memuat hasil tersimpan dari {cache_path.name}...")
        return joblib.load(cache_path)

    print("[persiapan] Membaca database (build_dataset.build(), SEKALI)...")
    built = build_dataset.build()
    observations = built["observations"]

    # build_dataset.build() SEKARANG sudah menempelkan item_type_at_install +
    # previous-cycle confirmed-failure secara native (fitur ini sudah jadi
    # bagian production, hasil eksperimen ini sendiri) - hanya tempelkan
    # kolom yang BELUM ada, supaya prepare_base_data() tetap bisa dipakai
    # ulang untuk eksperimen LANJUTAN (mis. last_confirmed_failure,
    # previous_cycle_end_reason) tanpa dobel/gagal kalau dijalankan lagi.
    if "item_type_at_install" not in observations.columns:
        print("[persiapan] Menempelkan konteks instalasi (item_type, lokasi)...")
        observations = install_context.attach_install_context(observations, built["events"])

    print("[persiapan] Audit previous-cycle (confirmed-failure-only, last-confirmed)...")
    pc = previous_cycle.audit_previous_cycle_features(built["cycles"])
    # Kolom mana yang BELUM ada di observations - build_dataset.build()
    # SEKARANG sudah menempelkan previous_cycle_confirmed_failure_lifetime_mean
    # DAN last_confirmed_failure_lifetime secara native (lewat
    # features.attach_final_context(), fitur ini sendiri hasil eksperimen
    # sesi sebelumnya). Merge TANPA guard ini menghasilkan kolom bentrok
    # (_x/_y, karena kedua sisi sudah punya nama yang sama) yang membuat
    # transform_for_model() gagal KeyError - bukan cuma soal previous_cycle_
    # confirmed_failure_lifetime_mean seperti guard lama, last_confirmed_
    # failure_lifetime butuh guard yang SAMA.
    pc_cols = ["installation_cycle_id", "previous_cycle_end_reason"]
    for column in ("previous_cycle_confirmed_failure_lifetime_mean", "last_confirmed_failure_lifetime"):
        if column not in observations.columns:
            pc_cols.append(column)
    observations = observations.merge(pc[pc_cols], on="installation_cycle_id", how="left")
    prev_transform = previous_cycle.transform_for_model(observations)
    new_transform_cols = [c for c in prev_transform.columns if c not in observations.columns]
    observations = pd.concat([observations, prev_transform[new_transform_cols]], axis=1)

    # "current_features" di sini berarti baseline LAMA (19 fitur classification
    # warisan) dipakai ablation sebagai titik pembanding "A_current" - BUKAN
    # features.FEATURE_COLUMNS (yang sekarang menunjuk ke konfigurasi FINAL
    # hasil eksperimen ini sendiri).
    current_features = feature_builder.build_features(observations, observations["_support"])[
        features.LEGACY_FEATURE_COLUMNS
    ].reset_index(drop=True)
    observations = observations.reset_index(drop=True)

    dataset = built["dataset"]
    assert len(dataset) == len(observations) == len(current_features)

    test_rows, window_days = None, None
    if with_operational:
        print("[persiapan] Membangun populasi TEST classification (dipinjam read-only, SEKALI)...")
        test_rows, window_days = evaluate.load_classification_test_rows()

    result = {
        **built,
        "observations": observations,
        "current_features": current_features,
        "test_rows": test_rows,
        "window_days": window_days,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, cache_path)
    return result


def split_masks(dataset: pd.DataFrame) -> dict:
    return {name: (dataset["split"] == name).to_numpy() for name in ("TRAIN", "VALIDATION", "TEST")}


# ---------------------------------------------------------------------------
# Helper inti: latih + evaluasi SATU kombinasi fitur (dipakai semua tahap)
# ---------------------------------------------------------------------------


def run_config(
    label: str,
    feature_frame: pd.DataFrame,
    categorical_columns: list[str],
    numeric_columns: list[str],
    dataset: pd.DataFrame,
    masks: dict,
    ctx: dict,
    *,
    rsf_params: dict | None = None,
    cox_params: dict | None = None,
    model_names: list[str] | None = None,
    model_params: dict[str, dict] | None = None,
    with_operational: bool = False,
    cache_key: str | None = None,
) -> dict:
    """Fit satu atau lebih model (dari src.model_fit.MODEL_REGISTRY) pada satu
    kombinasi fitur, evaluasi VAL/TEST native + (opsional) operasional
    30-hari vs classification - satu fungsi dipakai threshold experiment,
    ablation, previous-cycle audit, RSF tuning, DAN model-family experiment,
    supaya semuanya benar-benar memakai dataset/split/censoring yang sama.

    Dua cara memilih model, TIDAK bisa dicampur dalam satu panggilan:
    - `rsf_params`/`cox_params` (API LAMA, dipertahankan apa adanya supaya
      seluruh pemanggilan yang SUDAH ADA - threshold/ablation/previous-cycle/
      RSF tuning - tetap berjalan identik tanpa perubahan): selalu RSF+Cox.
    - `model_names`/`model_params` (BARU, dipakai run_model_family()):
      daftar model bebas dari MODEL_REGISTRY, dengan override hyperparameter
      opsional per model.

    `cache_key`: kalau diisi, hasil METRIK (bukan model - tidak dibutuhkan
    lagi setelah metriknya diambil, lihat result_row()) disimpan ke
    artifacts/_experiment_cache/ dan dipakai ulang kalau eksperimen ini
    dijalankan lagi - lingkungan terbukti bisa lambat/terputus di tengah
    puluhan fit berurutan, jadi setiap fit yang SUDAH selesai tidak perlu
    diulang saat script dijalankan kembali."""
    if cache_key is not None:
        cache_path = CACHE_DIR / f"{cache_key}.joblib"
        if cache_path.exists():
            cached = joblib.load(cache_path)
            print(f"      [cache] {label}")
            return cached

    # n_jobs DIPAKSA 1 untuk SETIAP model yang punya param itu (RSF,
    # ExtraSurvivalTrees): train.py sendiri (satu kali fit per proses) aman
    # dengan n_jobs=-1, tapi experiments.py melakukan puluhan fit BERURUTAN
    # dalam SATU proses panjang - loky (worker pool joblib) terbukti bisa
    # macet total tanpa error setelah beberapa siklus buat/bongkar pool di
    # sandbox ini (gejala yang sama dengan hang predict_survival_function
    # pada model ter-unpickle, lihat evaluate.py). Lebih lambat per fit,
    # tapi selesai - dan dataset ~15rb baris cukup kecil sehingga dampaknya
    # kecil. Model boosting (GBSA/Componentwise) tidak punya n_jobs sama
    # sekali (sekuensial by design) - tidak terpengaruh masalah ini.
    if model_names is None:
        names = ["random_survival_forest", "cox_ph"]
        overrides = {
            "random_survival_forest": {**(rsf_params or model_fit.DEFAULT_RSF_PARAMS), "n_jobs": 1},
        }
        if cox_params is not None:
            overrides["cox_ph"] = cox_params
    else:
        names = model_names
        overrides = {}
        for name in names:
            base = (model_params or {}).get(name, model_fit.MODEL_REGISTRY[name]["default_params"])
            overrides[name] = {**base, "n_jobs": 1} if "n_jobs" in base else base

    train_mask, val_mask, test_mask = masks["TRAIN"], masks["VALIDATION"], masks["TEST"]

    encoder = features.fit_encoder(feature_frame.loc[train_mask], categorical_columns)
    x_train = features.encode(feature_frame.loc[train_mask], encoder, numeric_columns)
    x_val = features.encode(feature_frame.loc[val_mask], encoder, numeric_columns)
    x_test = features.encode(feature_frame.loc[test_mask], encoder, numeric_columns)
    y_train = model_fit.make_survival_target(dataset, train_mask)
    y_val = model_fit.make_survival_target(dataset, val_mask)
    y_test = model_fit.make_survival_target(dataset, test_mask)

    models = model_fit.fit_models(x_train, y_train, names, overrides)
    native = model_fit.evaluate_models(models, y_train, x_val, y_val, x_test, y_test)

    operational = {}
    if with_operational:
        feature_frame_by_cycle = feature_frame.copy()
        feature_frame_by_cycle.index = dataset["installation_cycle_id"].to_numpy()
        for name, model in models.items():
            operational[name] = evaluate.score_operational(
                model, feature_frame_by_cycle, encoder, ctx["test_rows"], ctx["window_days"],
                numeric_columns=numeric_columns,
            )

    result = {"label": label, "models": models, "encoder": encoder, "native": native, "operational": operational}
    if cache_key is not None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Model TIDAK di-cache (result_row()/downstream stages hanya baca
        # "native"/"operational"/"config" - lihat docstring) supaya file cache
        # kecil dan cepat ditulis/dibaca.
        joblib.dump({**result, "models": {}, "encoder": None}, CACHE_DIR / f"{cache_key}.joblib")
    return result


def result_row(label: str, model_name: str, result: dict) -> dict:
    val = result["native"][model_name]["validation"]
    test = result["native"][model_name]["test"]
    op = result["operational"].get(model_name) if result["operational"] else None
    return {
        "experiment": f"{label} ({model_name})",
        "val_c_index": val["c_index"],
        "test_c_index": test["c_index"],
        "uno_c_index": test.get("uno_c_index"),
        "auc_30d": test["time_dependent_auc_at_horizon"].get(30),
        "auc_90d": test["time_dependent_auc_at_horizon"].get(90),
        "ibs": test["integrated_brier_score"],
        "pr_auc_30d_operational": op["pr_auc"] if op else None,
        "roc_auc_operational": op["roc_auc"] if op else None,
        "recall_at_capacity": op["recall_at_capacity"] if op else None,
        "precision_at_capacity": op["precision_at_capacity"] if op else None,
    }


def render_table(rows: list[dict]) -> str:
    header = (
        "| Experiment | VAL C-index | TEST C-index | Uno C | AUC30 | AUC90 | IBS | "
        "PR-AUC30(op) | ROC-AUC(op) | Recall@cap | Precision@cap |"
    )
    sep = "|---" * 10 + "|"

    def fmt(value):
        return f"{value:.4f}" if isinstance(value, (int, float)) else "N/A"

    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {fmt(row['val_c_index'])} | {fmt(row['test_c_index'])} | "
            f"{fmt(row['uno_c_index'])} | {fmt(row['auc_30d'])} | {fmt(row['auc_90d'])} | "
            f"{fmt(row['ibs'])} | {fmt(row['pr_auc_30d_operational'])} | "
            f"{fmt(row['roc_auc_operational'])} | {fmt(row['recall_at_capacity'])} | "
            f"{fmt(row['precision_at_capacity'])} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Langkah 3: threshold kategori khusus survival
# ---------------------------------------------------------------------------


def run_threshold_experiment(ctx: dict) -> dict:
    print("\n[1/4] Threshold kategori khusus survival (item_model, item_type, place)...")
    observations, dataset, masks = ctx["observations"], ctx["dataset"], ctx["masks"]
    current_features = ctx["current_features"]

    support_lookup = {
        "item_model_code_clean": observations["_support"],
        "item_type_at_install": categorical_support.cumulative_support(
            observations, "item_type_at_install", "observation_on"
        ),
        "place_at_install": categorical_support.cumulative_support(
            observations, "place_at_install", "observation_on"
        ),
    }

    stats_rows: list[dict] = []
    val_c_index_by_column_threshold: dict[tuple[str, int], float] = {}

    for column, support in support_lookup.items():
        raw_values = observations[column]
        for threshold in THRESHOLD_CANDIDATES:
            grouped = categorical_support.apply_threshold(raw_values, support, threshold)
            # PENTING: threshold_report_stats() butuh nilai MENTAH (sebelum
            # grouping), bukan `grouped` - ia sendiri yang menghitung
            # grouping-nya dari situ. Memasukkan `grouped` di sini membuat
            # "kategori asli" ikut berubah-ubah menurut threshold (seharusnya
            # tetap/threshold-invariant) - dicek manual & diperbaiki sebelum
            # eksperimen penuh dijalankan.
            stats = categorical_support.threshold_report_stats(
                raw_values.loc[masks["TRAIN"]], raw_values.loc[masks["VALIDATION"]], raw_values.loc[masks["TEST"]],
                threshold,
            )
            stats["column"] = column

            feat = current_features.copy()
            if column == "item_model_code_clean":
                feat["part_model_category"] = grouped.to_numpy()
                cat_cols, num_cols = features.LEGACY_CATEGORICAL_FEATURES, features.LEGACY_NUMERIC_FEATURES + features.FLEET_FEATURES
            else:
                feat[f"{column}_grouped"] = grouped.to_numpy()
                cat_cols = features.LEGACY_CATEGORICAL_FEATURES + [f"{column}_grouped"]
                num_cols = features.LEGACY_NUMERIC_FEATURES + features.FLEET_FEATURES

            result = run_config(
                f"{column}@{threshold}", feat, cat_cols, num_cols, dataset, masks, ctx,
                rsf_params=LIGHT_RSF_PARAMS, with_operational=False,
                cache_key=f"threshold_{column}_{threshold}",
            )
            stats["val_c_index_rsf"] = result["native"]["random_survival_forest"]["validation"]["c_index"]
            val_c_index_by_column_threshold[(column, threshold)] = stats["val_c_index_rsf"]
            stats_rows.append(stats)
            print(
                f"      {column:24s} threshold={threshold:>4d}  kategori asli={stats['original_categories']:>3d}  "
                f"digabung={stats['merged_into_low_support']:>3d}  unseen VAL={stats['val_rows_unseen_category']:>4d}  "
                f"unseen TEST={stats['test_rows_unseen_category']:>4d}  VAL C-index={stats['val_c_index_rsf']:.4f}"
            )

    chosen: dict[str, int] = {}
    for column in support_lookup:
        best_threshold = max(
            THRESHOLD_CANDIDATES, key=lambda t: val_c_index_by_column_threshold[(column, t)]
        )
        chosen[column] = best_threshold

    report_lines = ["# Threshold kategori khusus survival", ""]
    report_lines.append(
        "Threshold KHUSUS survival (bukan `config.MIN_PART_MODEL_SUPPORT=300` classification, "
        "yang dikalibrasi untuk skala 251.568 baris TRAIN classification, bukan ~15rb lifecycle "
        "TRAIN survival). Dipilih dari VAL C-index (RSF ringan, 50 pohon) - TEST TIDAK dipakai "
        "memilih, hanya dilaporkan pada tahap ablation/final."
    )
    report_lines.append("")
    report_lines.append(
        "| Kolom | Threshold | Kategori asli | Digabung LOW_SUPPORT | Unseen VAL | Unseen TEST | VAL C-index (RSF ringan) |"
    )
    report_lines.append("|---|---|---|---|---|---|---|")
    for row in stats_rows:
        marker = " **<-dipilih**" if chosen[row["column"]] == row["threshold"] else ""
        report_lines.append(
            f"| {row['column']} | {row['threshold']} | {row['original_categories']} | "
            f"{row['merged_into_low_support']} | {row['val_rows_unseen_category']}/{row['val_rows_total']} | "
            f"{row['test_rows_unseen_category']}/{row['test_rows_total']} | {row['val_c_index_rsf']:.4f}{marker} |"
        )
    report_lines.append("")
    report_lines.append(f"Threshold terpilih: `{chosen}`")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "category_threshold.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"      Threshold terpilih: {chosen}")
    print(f"      Laporan: {REPORTS_DIR / 'category_threshold.md'}")
    return chosen


# ---------------------------------------------------------------------------
# Langkah 4-5: ablation A(current)/B(context-only)/C(combined) + incremental
# ---------------------------------------------------------------------------


def build_context_columns(observations: pd.DataFrame, chosen_thresholds: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=observations.index)
    out["part_model_category_survival"] = categorical_support.apply_threshold(
        observations["item_model_code_clean"], observations["_support"], chosen_thresholds["item_model_code_clean"],
    ).to_numpy()
    out["item_type_at_install_grouped"] = categorical_support.apply_threshold(
        observations["item_type_at_install"],
        categorical_support.cumulative_support(observations, "item_type_at_install", "observation_on"),
        chosen_thresholds["item_type_at_install"],
    ).to_numpy()
    out["place_at_install_grouped"] = categorical_support.apply_threshold(
        observations["place_at_install"],
        categorical_support.cumulative_support(observations, "place_at_install", "observation_on"),
        chosen_thresholds["place_at_install"],
    ).to_numpy()
    return out


def run_ablation(ctx: dict, chosen_thresholds: dict) -> dict:
    print("\n[2/4] Ablation A(current)/B(context-only)/C(combined) + incremental...")
    observations, dataset, masks = ctx["observations"], ctx["dataset"], ctx["masks"]
    current_features = ctx["current_features"]
    context_cols = build_context_columns(observations, chosen_thresholds)

    configs: dict[str, dict] = {}

    configs["A_current"] = dict(
        feature_frame=current_features,
        categorical_columns=features.LEGACY_CATEGORICAL_FEATURES,
        numeric_columns=features.LEGACY_NUMERIC_FEATURES + features.FLEET_FEATURES,
    )

    context_only = pd.concat(
        [context_cols[["part_model_category_survival", "item_type_at_install_grouped", "place_at_install_grouped"]],
         current_features[["client_category"]]],
        axis=1,
    )
    configs["B_context_only"] = dict(
        feature_frame=context_only,
        categorical_columns=[
            "part_model_category_survival", "client_category",
            "item_type_at_install_grouped", "place_at_install_grouped",
        ],
        numeric_columns=[],
    )

    a_plus_type = pd.concat([current_features, context_cols[["item_type_at_install_grouped"]]], axis=1)
    configs["A_plus_item_type"] = dict(
        feature_frame=a_plus_type,
        categorical_columns=features.LEGACY_CATEGORICAL_FEATURES + ["item_type_at_install_grouped"],
        numeric_columns=features.LEGACY_NUMERIC_FEATURES + features.FLEET_FEATURES,
    )

    a_plus_place = pd.concat([current_features, context_cols[["place_at_install_grouped"]]], axis=1)
    configs["A_plus_place"] = dict(
        feature_frame=a_plus_place,
        categorical_columns=features.LEGACY_CATEGORICAL_FEATURES + ["place_at_install_grouped"],
        numeric_columns=features.LEGACY_NUMERIC_FEATURES + features.FLEET_FEATURES,
    )

    combined = pd.concat(
        [current_features, context_cols[["item_type_at_install_grouped", "place_at_install_grouped"]]], axis=1
    )
    configs["C_combined"] = dict(
        feature_frame=combined,
        categorical_columns=features.LEGACY_CATEGORICAL_FEATURES + ["item_type_at_install_grouped", "place_at_install_grouped"],
        numeric_columns=features.LEGACY_NUMERIC_FEATURES + features.FLEET_FEATURES,
    )

    results: dict[str, dict] = {}
    rows: list[dict] = []
    for label, cfg in configs.items():
        print(f"      Melatih {label}...")
        result = run_config(label, cfg["feature_frame"], cfg["categorical_columns"], cfg["numeric_columns"],
                             dataset, masks, ctx, cache_key=f"ablation_{label}")
        results[label] = {**result, "config": cfg}
        for model_name in ("random_survival_forest", "cox_ph"):
            row = result_row(label, model_name, result)
            rows.append(row)
            print(f"        {row['experiment']:38s} VAL C-index={row['val_c_index']:.4f}  TEST C-index={row['test_c_index']:.4f}")

    report = ["# Feature ablation: current vs context-only vs combined", ""]
    report.append(
        "A = fitur classification warisan (19 kolom, tidak berubah). "
        "B = HANYA konteks instalasi (part model/client/item type/lokasi, threshold khusus "
        "survival dari category_threshold.md, TANPA riwayat/armada/lifecycle). "
        "C = A + item_type_at_install + place_at_install (part_model/client sudah ada di A). "
        "A_plus_* mengisolasi kontribusi 1 fitur baru saja."
    )
    report.append("")
    report.append(render_table(rows))
    (REPORTS_DIR / "feature_ablation.md").write_text("\n".join(report), encoding="utf-8")
    print(f"      Laporan: {REPORTS_DIR / 'feature_ablation.md'}")

    best_label = max(
        configs, key=lambda label: results[label]["native"]["random_survival_forest"]["validation"]["c_index"]
    )
    print(f"      Konfigurasi terbaik (VAL C-index, RSF): {best_label}")
    return {"results": results, "best_label": best_label, "rows": rows}


# ---------------------------------------------------------------------------
# Langkah 6: audit previous-cycle
# ---------------------------------------------------------------------------


def run_previous_cycle_audit(ctx: dict, ablation: dict) -> dict:
    print("\n[3/4] Audit previous-cycle (existing vs confirmed-failure vs last-confirmed)...")
    dataset, masks = ctx["dataset"], ctx["masks"]
    best = ablation["results"][ablation["best_label"]]
    # "existing" previous_cycle_lifetime_mean SUDAH ada sebagai kolom asli di
    # feature_frame konfigurasi terbaik (bagian dari 19 fitur current) - kalau
    # tidak di-drop dari DataFrame-nya (bukan cuma dari daftar nama kolom),
    # varian "existing" di bawah akan menempelkan kolom bernama SAMA lagi dan
    # menghasilkan DataFrame dengan kolom duplikat (RandomSurvivalForest
    # menolaknya). Ditemukan lewat error nyata saat run pertama, bukan
    # diasumsikan benar dari awal.
    _prev_cycle_cols = ["log_previous_cycle_lifetime_mean", "has_previous_cycle"]
    base_feature_frame = best["config"]["feature_frame"].drop(
        columns=[c for c in _prev_cycle_cols if c in best["config"]["feature_frame"].columns]
    )
    base_cat = best["config"]["categorical_columns"]
    base_num = [c for c in best["config"]["numeric_columns"] if c not in _prev_cycle_cols]

    observations = ctx["observations"]
    variant_columns = {
        "existing": ["log_previous_cycle_lifetime_mean", "has_previous_cycle"],
        "confirmed_failure_only": [
            "log_previous_cycle_confirmed_failure_lifetime_mean",
            "has_previous_cycle_confirmed_failure_lifetime_mean",
        ],
        "last_confirmed_failure": ["log_last_confirmed_failure_lifetime", "has_last_confirmed_failure_lifetime"],
    }

    rows: list[dict] = []
    variant_results: dict[str, dict] = {}
    for variant_name, cols in variant_columns.items():
        source = ctx["current_features"] if variant_name == "existing" else observations
        extra = source[cols].reset_index(drop=True)
        feat = pd.concat([base_feature_frame.reset_index(drop=True), extra], axis=1)
        label = f"prev_cycle={variant_name}"
        result = run_config(
            label, feat, base_cat, base_num + cols, dataset, masks, ctx, with_operational=False,
            cache_key=f"prevcycle_{ablation['best_label']}_{variant_name}",
        )
        variant_results[variant_name] = result
        row = result_row(label, "random_survival_forest", result)
        rows.append(row)
        print(f"      {label:38s} VAL C-index={row['val_c_index']:.4f}  TEST C-index={row['test_c_index']:.4f}")

    best_variant = max(
        variant_columns, key=lambda v: variant_results[v]["native"]["random_survival_forest"]["validation"]["c_index"]
    )
    baseline_val = variant_results["existing"]["native"]["random_survival_forest"]["validation"]["c_index"]
    best_val = variant_results[best_variant]["native"]["random_survival_forest"]["validation"]["c_index"]

    # previous_cycle_end_reason: ditambahkan DI ATAS varian numerik terbaik,
    # hanya dipertahankan kalau membantu VALIDATION secara konsisten.
    extra_reason = observations[["previous_cycle_end_reason"]].reset_index(drop=True)
    feat_with_reason = pd.concat(
        [base_feature_frame.reset_index(drop=True),
         observations[variant_columns[best_variant]].reset_index(drop=True), extra_reason],
        axis=1,
    )
    label = f"prev_cycle={best_variant}+end_reason"
    result_reason = run_config(
        label, feat_with_reason, base_cat + ["previous_cycle_end_reason"],
        base_num + variant_columns[best_variant], dataset, masks, ctx, with_operational=False,
        cache_key=f"prevcycle_{ablation['best_label']}_{best_variant}_end_reason",
    )
    row_reason = result_row(label, "random_survival_forest", result_reason)
    rows.append(row_reason)
    reason_val = result_reason["native"]["random_survival_forest"]["validation"]["c_index"]
    print(f"      {label:38s} VAL C-index={reason_val:.4f}")

    keep_reason = reason_val > best_val
    final_columns = variant_columns[best_variant] + (["previous_cycle_end_reason"] if keep_reason else [])

    report = ["# Audit previous_cycle_lifetime_mean", ""]
    report.append(
        "`previous_cycle_lifetime_mean` (dari `data_reader.get_cycles()` SQL) TERBUKTI mencampur "
        "rata-rata durasi siklus sebelumnya APAPUN cara berakhirnya (FAILURE, "
        "RIGHT_CENSORED_AT_DATA_END, REINSTALL_WITHOUT_RECORDED_FAILURE) - bukan murni "
        "\"lifetime sampai gagal\" seperti namanya. Diuji di atas konfigurasi terbaik "
        f"({ablation['best_label']}) dari tahap ablation."
    )
    report.append("")
    report.append(render_table(rows))
    report.append("")
    report.append(
        f"Varian terpilih: **{best_variant}** (VAL C-index {best_val:.4f} vs existing {baseline_val:.4f}). "
        f"previous_cycle_end_reason {'DIPERTAHANKAN' if keep_reason else 'TIDAK dipertahankan'} "
        f"(VAL C-index {reason_val:.4f} {'>' if keep_reason else '<='} {best_val:.4f})."
    )
    (REPORTS_DIR / "previous_cycle_audit.md").write_text("\n".join(report), encoding="utf-8")
    print(f"      Laporan: {REPORTS_DIR / 'previous_cycle_audit.md'}")

    final_feature_frame = pd.concat(
        [base_feature_frame.reset_index(drop=True), observations[final_columns].reset_index(drop=True)], axis=1
    )
    final_categorical = base_cat + (["previous_cycle_end_reason"] if keep_reason else [])
    final_numeric = base_num + variant_columns[best_variant]
    return {
        "feature_frame": final_feature_frame,
        "categorical_columns": final_categorical,
        "numeric_columns": final_numeric,
        "best_variant": best_variant,
        "keep_end_reason": keep_reason,
    }


# ---------------------------------------------------------------------------
# Langkah 8: RSF small tuning (coordinate-wise, bukan grid penuh)
# ---------------------------------------------------------------------------


def run_rsf_tuning(ctx: dict, final_features: dict) -> dict:
    print("\n[4/4] RSF small tuning (coordinate-wise dari titik current)...")
    dataset, masks = ctx["dataset"], ctx["masks"]
    feature_frame = final_features["feature_frame"]
    cat_cols, num_cols = final_features["categorical_columns"], final_features["numeric_columns"]

    base_params = dict(model_fit.DEFAULT_RSF_PARAMS)
    axes = {
        "n_estimators": [200, 400],
        "min_samples_leaf": [10, 20, 30, 50],
        "max_features": ["sqrt", 0.5, 1.0],
        "max_depth": [None, 8, 12],
    }

    tuning_prefix = f"tuning_{final_features['best_variant']}_{final_features['keep_end_reason']}"

    rows: list[dict] = []
    current_params = dict(base_params)
    baseline_result = run_config(
        "rsf_tuning=current", feature_frame, cat_cols, num_cols, dataset, masks, ctx,
        rsf_params=current_params, with_operational=False, cache_key=f"{tuning_prefix}_baseline",
    )
    best_val = baseline_result["native"]["random_survival_forest"]["validation"]["c_index"]
    rows.append(result_row("tuning=current(baseline)", "random_survival_forest", baseline_result))
    print(f"      baseline {current_params} VAL C-index={best_val:.4f}")

    for axis, candidates in axes.items():
        for value in candidates:
            if current_params.get(axis) == value:
                continue
            trial_params = {**current_params, axis: value}
            label = f"tuning={axis}={value}"
            result = run_config(
                label, feature_frame, cat_cols, num_cols, dataset, masks, ctx,
                rsf_params=trial_params, with_operational=False,
                cache_key=f"{tuning_prefix}_{axis}_{value}",
            )
            val_c = result["native"]["random_survival_forest"]["validation"]["c_index"]
            rows.append(result_row(label, "random_survival_forest", result))
            print(f"      {axis}={value} VAL C-index={val_c:.4f}")
            if val_c > best_val:
                best_val, current_params = val_c, trial_params

    print(f"      Hyperparameter terpilih: {current_params} (VAL C-index={best_val:.4f})")

    params_key = "_".join(f"{k}={v}" for k, v in sorted(current_params.items()))
    final_result = run_config(
        "FINAL", feature_frame, cat_cols, num_cols, dataset, masks, ctx,
        rsf_params=current_params, with_operational=False,
        cache_key=f"{tuning_prefix}_FINAL_{params_key}",
    )
    final_rows = [
        result_row("FINAL (random_survival_forest)", "random_survival_forest", final_result),
        result_row("FINAL (cox_ph)", "cox_ph", final_result),
    ]

    report = ["# RSF tuning & perbandingan model final", ""]
    report.append(
        "Pencarian KECIL coordinate-wise (bukan grid penuh 2x4x3x3=72) di sekitar titik "
        "hyperparameter current - satu sumbu diubah per langkah, dipertahankan hanya kalau "
        "menaikkan VAL C-index. TEST hanya untuk pelaporan akhir."
    )
    report.append("")
    report.append(
        "Kolom operasional (PR-AUC30/ROC-AUC/Recall/Precision@kapasitas) kosong di sini dengan "
        "sengaja - membangun ulang populasi TEST classification (1,4 juta baris) terbukti langkah "
        "paling berat & paling rentan macet di lingkungan eksperimen ini. Angka operasional untuk "
        "konfigurasi final didapat dengan menjalankan `python evaluate.py` SETELAH `train.py` "
        "diupdate ke konfigurasi ini - lihat `reports/evaluation_report.md`."
    )
    report.append("")
    report.append("## Pencarian tuning")
    report.append(render_table(rows))
    report.append("")
    report.append("## Model final (RSF vs Cox PH, fitur+threshold+hyperparameter terpilih)")
    report.append(render_table(final_rows))
    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(report), encoding="utf-8")
    print(f"      Laporan: {REPORTS_DIR / 'model_comparison.md'}")

    return {
        "rsf_params": current_params,
        "feature_frame": feature_frame,
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "final_result": final_result,
    }


# ---------------------------------------------------------------------------
# Langkah 0 (sesi peningkatan C-index): ketidakpastian baseline. VALIDATION
# hanya 385 event (metadata.json) - C-index sebagai satu angka tunggal bisa
# menyesatkan soal seberapa jauh dua kandidat BENAR-BENAR berbeda. Dijalankan
# SEBELUM eksperimen model-family/fitur baru berikutnya supaya ada rentang
# pembanding eksplisit: kandidat baru hanya dianggap menang kalau naik DI
# LUAR interval ini, bukan menang tipis 0,001 yang bisa jadi murni noise
# resampling. Dipisah dari langkah 3-8 (threshold/ablation/previous-cycle/
# tuning di atas) karena bekerja pada artifact PRODUKSI yang SUDAH dilatih
# (train.py), bukan kandidat baru - tidak butuh chosen_thresholds/ablation/
# tuning apa pun sebagai input.
# ---------------------------------------------------------------------------


def run_uncertainty_baseline(ctx: dict, n_seeds: int = 5) -> dict:
    print("\n[ketidakpastian] Bootstrap CI C-index (model produksi saat ini) + variasi seed RSF...")
    dataset, masks = ctx["dataset"], ctx["masks"]
    feature_frame = ctx["features"]  # fitur FINAL produksi (src/features.py)

    models = joblib.load(ARTIFACTS_DIR / "models.joblib")
    for model in models.values():
        # Alasan sama seperti evaluate.py load_artifacts(): predict_survival_
        # function() pada RSF ter-unpickle dengan n_jobs=-1 terbukti hang.
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
    encoder = joblib.load(ARTIFACTS_DIR / "encoder.joblib")
    y_train = joblib.load(ARTIFACTS_DIR / "y_train.joblib")

    val_mask = masks["VALIDATION"]
    x_val = features.encode(feature_frame.loc[val_mask], encoder)
    y_val = model_fit.make_survival_target(dataset, val_mask)

    bootstrap_rows: list[dict] = []
    for name, model in models.items():
        risk_sign = model_fit.MODEL_REGISTRY.get(name, {}).get("risk_sign", 1)
        ci = evaluation.bootstrap_c_index(model, y_train, x_val, y_val, risk_sign=risk_sign, n_boot=200, seed=42)
        bootstrap_rows.append({"model": name, **ci})
        print(
            f"      {name:24s} VAL C-index={ci['point_estimate']:.4f}  "
            f"95% CI=[{ci['ci_lower_2_5']:.4f}, {ci['ci_upper_97_5']:.4f}]  std={ci['std']:.4f}"
        )

    # Variasi antar random_state RSF (fitur & hyperparameter TETAP sama,
    # hanya seed acak yang beda) - sumber ketidakpastian KEDUA, terpisah dari
    # bootstrap baris di atas (yang model-nya tetap, baris eval yang berubah).
    train_mask = masks["TRAIN"]
    x_train = features.encode(feature_frame.loc[train_mask], encoder)
    seed_rows: list[dict] = []
    for seed in range(n_seeds):
        rsf_params = {**model_fit.DEFAULT_RSF_PARAMS, "n_jobs": 1, "random_state": seed}
        model = model_fit.MODEL_REGISTRY["random_survival_forest"]["cls"](**rsf_params).fit(x_train, y_train)
        native = evaluation.native_metrics(model, y_train, x_val, y_val)
        seed_rows.append({"seed": seed, "val_c_index": native["c_index"]})
        print(f"      RSF seed={seed} VAL C-index={native['c_index']:.4f}")

    seed_values = np.array([r["val_c_index"] for r in seed_rows])

    report = ["# Ketidakpastian baseline C-index (sebelum eksperimen model-family/fitur baru)", ""]
    report.append(
        f"VALIDATION: {int(val_mask.sum())} baris, "
        f"{int(dataset.loc[val_mask, 'event_observed'].sum())} event - kecil, jadi C-index titik "
        "tunggal bisa menyesatkan. Dua sumber ketidakpastian diukur terpisah: (1) bootstrap resampling "
        "baris VAL pada model produksi SAAT INI (model tidak berubah, hanya baris mana yang masuk "
        "perhitungan C-index yang berubah), (2) variasi antar random_state RSF (model berubah, baris "
        "VAL tetap). Kandidat model/fitur baru pada langkah berikutnya (model_family.md dst.) HANYA "
        "dianggap menang kalau VAL C-index-nya di LUAR rentang berikut, bukan menang tipis di dalam "
        "noise ini."
    )
    report.append("")
    report.append("## Bootstrap CI (200 resample, model produksi saat ini)")
    report.append("| Model | Point estimate | 95% CI lower | 95% CI upper | Std |")
    report.append("|---|---|---|---|---|")
    for row in bootstrap_rows:
        report.append(
            f"| {row['model']} | {row['point_estimate']:.4f} | {row['ci_lower_2_5']:.4f} | "
            f"{row['ci_upper_97_5']:.4f} | {row['std']:.4f} |"
        )
    report.append("")
    report.append(f"## Variasi antar seed RSF ({n_seeds} seed, hyperparameter & fitur sama)")
    report.append("| Seed | VAL C-index |")
    report.append("|---|---|")
    for row in seed_rows:
        report.append(f"| {row['seed']} | {row['val_c_index']:.4f} |")
    report.append("")
    report.append(
        f"Rentang antar-seed: [{seed_values.min():.4f}, {seed_values.max():.4f}] "
        f"(std={float(seed_values.std()):.4f})."
    )
    (REPORTS_DIR / "uncertainty_baseline.md").write_text("\n".join(report), encoding="utf-8")
    print(f"      Laporan: {REPORTS_DIR / 'uncertainty_baseline.md'}")

    return {"bootstrap": bootstrap_rows, "seed_variation": seed_rows}


# ---------------------------------------------------------------------------
# Langkah tambahan (sesi peningkatan C-index): keluarga model. Ablation lama
# (feature_ablation.md) sudah membuktikan performa TIDAK stabil tanpa fitur
# riwayat, dan RSF tuning lama (model_comparison.md) sudah membuktikan
# tuning DALAM keluarga RSF tidak membantu - tapi keluarga model DI LUAR
# RSF/Cox belum pernah dicoba sama sekali. Sengaja TIDAK bergantung pada
# run_threshold_experiment/run_ablation/run_previous_cycle_audit/
# run_rsf_tuning di atas: dijalankan pada fitur FINAL PRODUKSI
# (features.FEATURE_COLUMNS, threshold item_model=200 yang benar - lihat
# README poin 8 soal diskrepansi threshold=300 di tabel tuning LAMA), supaya
# isolasinya bersih - HANYA keluarga model yang berubah, fitur identik
# dengan train.py saat ini.
# ---------------------------------------------------------------------------


def run_model_family(ctx: dict) -> dict:
    print(
        "\n[keluarga model] RSF vs Cox vs ExtraSurvivalTrees vs "
        "GBSA(coxph) vs ComponentwiseGBSA(coxph)..."
    )
    dataset, masks = ctx["dataset"], ctx["masks"]
    feature_frame = ctx["features"]  # fitur FINAL produksi (src/features.py), threshold=200 sudah benar

    model_names = list(model_fit.MODEL_REGISTRY.keys())
    result = run_config(
        "model_family", feature_frame, features.CATEGORICAL_FEATURES,
        features.NUMERIC_FEATURES + features.FLEET_FEATURES, dataset, masks, ctx,
        model_names=model_names, with_operational=False, cache_key="model_family_all",
    )

    rows = [result_row("model_family", name, result) for name in model_names]
    for row in rows:
        print(
            f"      {row['experiment']:38s} VAL C-index={row['val_c_index']:.4f}  "
            f"TEST C-index={row['test_c_index']:.4f}"
        )

    report = [
        "# Keluarga model: RSF vs Cox PH vs ExtraSurvivalTrees vs GBSA vs ComponentwiseGBSA", "",
    ]
    report.append(
        "Semua model dilatih pada fitur FINAL PRODUKSI yang SAMA PERSIS (src/features.py, threshold "
        "item_model=200/item_type=300 - lihat README poin 4 & 8), hyperparameter default per keluarga "
        "(BUKAN hasil tuning - tuning per-keluarga adalah langkah terpisah). Tujuannya mengisolasi "
        "kontribusi KELUARGA MODEL saja, terpisah dari kontribusi fitur (yang sudah diaudit habis di "
        "feature_ablation.md/previous_cycle_audit.md - lihat README poin 5-6 dan 11)."
    )
    report.append("")
    report.append(
        "`GradientBoostingSurvivalAnalysis(loss='ipcwls'/'squared')` DIUJI lewat smoke test dan DIBUANG "
        "dari registry (bukan dilewati tanpa dicoba): loss selain 'coxph' tidak punya baseline hazard "
        "model, `predict_survival_function()`-nya melempar ValueError - tidak kompatibel dengan seluruh "
        "pipeline di sini (evaluate.py IBS/Brier/AUC, predict.py) yang butuh kurva S(t) di SETIAP model, "
        "bukan cuma skor risiko. Lihat catatan di src/model_fit.py."
    )
    report.append("")
    report.append(
        "Bandingkan angka di sini dengan reports/uncertainty_baseline.md - kandidat hanya dianggap "
        "menang kalau naiknya di luar rentang ketidakpastian baseline di sana, bukan menang tipis."
    )
    report.append("")
    report.append(render_table(rows))
    (REPORTS_DIR / "model_family.md").write_text("\n".join(report), encoding="utf-8")
    print(f"      Laporan: {REPORTS_DIR / 'model_family.md'}")

    best_name = max(model_names, key=lambda name: result["native"][name]["validation"]["c_index"])
    print(
        f"      Keluarga model terbaik (VAL C-index): {best_name} "
        f"({result['native'][best_name]['validation']['c_index']:.4f})"
    )
    return {"result": result, "rows": rows, "best_name": best_name}


# ---------------------------------------------------------------------------
# Langkah tambahan (sesi peningkatan C-index, Fase 2): fitur hazard baru -
# prior survival empiris per grup (part model/item type/client), lihat
# src/hazard_features.py untuk definisi lengkap dan alasan point-in-time
# safety-nya. Forward-selection bertahap DARI FITUR FINAL PRODUKSI (bukan
# dari nol, beda dengan run_ablation() yang membandingkan A_current warisan
# classification vs konteks) - satu grup ditambah per langkah dulu supaya
# kontribusinya terbaca terpisah, baru kombinasi ketiganya di akhir. Sengaja
# TIDAK bergantung pada run_threshold_experiment/run_ablation/
# run_previous_cycle_audit/run_rsf_tuning/run_model_family di atas - fitur
# baru diuji di atas apa yang SUDAH final (features.FEATURE_COLUMNS), sama
# seperti run_model_family() mengisolasi kontribusi keluarga model.
# ---------------------------------------------------------------------------


def run_hazard_ablation(ctx: dict) -> dict:
    print("\n[fitur hazard] Prior survival empiris (part_model/item_type/client)...")
    observations, dataset, masks = ctx["observations"], ctx["dataset"], ctx["masks"]
    cycles, events = ctx["cycles"], ctx["events"]
    final_features = ctx["features"]  # fitur FINAL produksi (src/features.py)

    # item_type_at_install TIDAK ada di cycles mentah (hanya ditempel ke
    # observations lewat install_context.attach_install_context() di
    # build_dataset.build()) - ditempel ULANG di sini ke cycles PENUH (bukan
    # hanya lifecycle eligible survival), supaya populasi "prior" untuk grup
    # ini konsisten dengan part_model/client (populasi is_initial_model_cohort
    # penuh, sama seperti attach_fleet). Tidak ada query DB baru - events
    # SUDAH dibaca build_dataset.build().
    cycles_with_type = install_context.attach_install_context(cycles, events)

    prior_part_model = hazard_features.empirical_prior_survival(
        observations, cycles, "item_model_code_clean", "part_model"
    )
    prior_item_type = hazard_features.empirical_prior_survival(
        observations, cycles_with_type, "item_type_at_install", "item_type"
    )
    prior_client = hazard_features.empirical_prior_survival(
        observations, cycles, "installed_client_clean", "client"
    )

    base_numeric = features.NUMERIC_FEATURES + features.FLEET_FEATURES

    configs: dict[str, dict] = {}
    configs["A_final"] = dict(feature_frame=final_features, numeric_columns=base_numeric)
    configs["A_plus_partmodel_prior"] = dict(
        feature_frame=pd.concat([final_features, prior_part_model], axis=1),
        numeric_columns=base_numeric + list(prior_part_model.columns),
    )
    configs["A_plus_itemtype_prior"] = dict(
        feature_frame=pd.concat([final_features, prior_item_type], axis=1),
        numeric_columns=base_numeric + list(prior_item_type.columns),
    )
    configs["A_plus_client_prior"] = dict(
        feature_frame=pd.concat([final_features, prior_client], axis=1),
        numeric_columns=base_numeric + list(prior_client.columns),
    )
    configs["A_plus_all_priors"] = dict(
        feature_frame=pd.concat([final_features, prior_part_model, prior_item_type, prior_client], axis=1),
        numeric_columns=(
            base_numeric + list(prior_part_model.columns) + list(prior_item_type.columns) + list(prior_client.columns)
        ),
    )

    results: dict[str, dict] = {}
    rows: list[dict] = []
    for label, cfg in configs.items():
        print(f"      Melatih {label}...")
        result = run_config(
            label, cfg["feature_frame"], features.CATEGORICAL_FEATURES, cfg["numeric_columns"],
            dataset, masks, ctx, cache_key=f"hazard_{label}",
        )
        results[label] = {**result, "config": cfg}
        for model_name in ("random_survival_forest", "cox_ph"):
            row = result_row(label, model_name, result)
            rows.append(row)
            print(
                f"        {row['experiment']:38s} VAL C-index={row['val_c_index']:.4f}  "
                f"TEST C-index={row['test_c_index']:.4f}"
            )

    report = ["# Fitur hazard baru: prior survival empiris per grup (Fase 2)", ""]
    report.append(
        "F1 dari plan peningkatan C-index - untuk tiap lifecycle: di antara lifecycle LAIN pada grup "
        "yang sama (part model/item type/client) yang SUDAH BERAKHIR sebelum installed_on baris ini "
        "(point-in-time, mekanisme sama dengan feature_builder.attach_fleet - dihitung dari populasi "
        "is_initial_model_cohort PENUH, bukan dibatasi lifecycle eligible survival), berapa yang "
        "berakhir FAILURE dan berapa median durasinya. A_final = fitur produksi saat ini (tidak "
        "berubah, baseline). A_plus_* mengisolasi kontribusi 1 grup saja. Bandingkan dengan "
        "reports/uncertainty_baseline.md - hanya dianggap menang kalau naiknya di luar rentang "
        "ketidakpastian baseline di sana. Lihat src/hazard_features.py untuk definisi lengkap."
    )
    report.append("")
    report.append(render_table(rows))
    (REPORTS_DIR / "hazard_ablation.md").write_text("\n".join(report), encoding="utf-8")
    print(f"      Laporan: {REPORTS_DIR / 'hazard_ablation.md'}")

    best_label = max(
        configs, key=lambda label: results[label]["native"]["random_survival_forest"]["validation"]["c_index"]
    )
    print(f"      Konfigurasi terbaik (VAL C-index, RSF): {best_label}")
    return {"results": results, "best_label": best_label, "rows": rows, "configs": configs}


# ---------------------------------------------------------------------------


def main() -> int:
    built = prepare_base_data()
    dataset = built["dataset"]
    masks = split_masks(dataset)
    ctx = {**built, "masks": masks}

    uncertainty = run_uncertainty_baseline(ctx)
    model_family = run_model_family(ctx)
    hazard = run_hazard_ablation(ctx)

    chosen_thresholds = run_threshold_experiment(ctx)
    ablation = run_ablation(ctx, chosen_thresholds)
    final_features = run_previous_cycle_audit(ctx, ablation)
    tuning = run_rsf_tuning(ctx, final_features)

    print("\n[SELESAI] Konfigurasi final:")
    print(f"  threshold           : {chosen_thresholds}")
    print(f"  fitur ablation      : {ablation['best_label']}")
    print(f"  previous-cycle      : {final_features['best_variant']} "
          f"(end_reason={'ya' if final_features['keep_end_reason'] else 'tidak'})")
    print(f"  hyperparameter RSF  : {tuning['rsf_params']}")
    print(f"  kolom kategorikal   : {tuning['categorical_columns']}")
    print(f"  kolom numerik       : {tuning['numeric_columns']}")
    print(f"  keluarga model terbaik (VAL): {model_family['best_name']}")
    print(f"  fitur hazard terbaik (VAL)  : {hazard['best_label']}")
    print(
        f"  RSF baseline 95% CI (VAL)  : "
        f"[{uncertainty['bootstrap'][0]['ci_lower_2_5']:.4f}, {uncertainty['bootstrap'][0]['ci_upper_97_5']:.4f}]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
