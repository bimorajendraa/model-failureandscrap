"""Fit model survival + evaluasi native, lewat registry keluarga model.

Diekstrak dari train.py supaya `experiments.py` (puluhan model kandidat:
ablation, threshold sweep, tuning) memakai logic fitting/evaluasi yang SAMA
PERSIS dengan model production - tidak ada logic yang di-duplikasi/berbeda
antara eksperimen dan hasil akhir.

`MODEL_REGISTRY` diperluas (sesi peningkatan C-index) dari RSF+Cox saja
menjadi 6 keluarga model - tapi `fit_models()`/`evaluate_models()` TETAP
kompatibel dengan pemanggilan lama (`fit_models(x_train, y_train)` tanpa
argumen lain) karena `DEFAULT_MODEL_NAMES` tetap RSF+Cox, sama seperti
sebelumnya. train.py TIDAK berubah perilakunya sampai konfigurasi baru
benar-benar terpilih dari VALIDATION (lihat reports/model_family.md).
"""

from __future__ import annotations

from sksurv.ensemble import (
    ComponentwiseGradientBoostingSurvivalAnalysis,
    ExtraSurvivalTrees,
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from . import evaluation

# Titik awal (dipakai train.py apa adanya kalau tidak ada override) - hasil
# sesi sebelumnya: pembulatan duration_days ke hari bulat + min_samples_leaf
# ini menjaga artifact model tetap <1 GiB tanpa kehilangan C-index (lihat
# README bagian "Catatan teknis"). RSF small tuning di experiments.py mencari
# di sekitar titik ini, bukan grid penuh.
DEFAULT_RSF_PARAMS = dict(
    n_estimators=100,
    min_samples_split=40,
    min_samples_leaf=30,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)
DEFAULT_COX_PARAMS = dict(alpha=0.1, ties="efron")

# Keluarga model TAMBAHAN (audit "keluarga model belum pernah dicoba" -
# README/plan peningkatan C-index). Hyperparameter awal dipilih supaya
# sebanding dengan DEFAULT_RSF_PARAMS (kedalaman/regularisasi serupa), BUKAN
# hasil tuning - tuning-nya sendiri langkah terpisah (lihat run_rsf_tuning
# dan penerusnya di experiments.py), sama seperti RSF dulu.
DEFAULT_EXTRA_TREES_PARAMS = dict(
    n_estimators=100,
    min_samples_split=40,
    min_samples_leaf=30,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)
DEFAULT_GBSA_COXPH_PARAMS = dict(
    loss="coxph", n_estimators=100, learning_rate=0.1, max_depth=3,
    min_samples_leaf=30, subsample=1.0, random_state=42,
)
DEFAULT_COMPONENTWISE_GBSA_PARAMS = dict(loss="coxph", n_estimators=100, random_state=42)
# GradientBoostingSurvivalAnalysis(loss='ipcwls'/'squared') DIUJI dan
# DIBUANG (bukan dilewati tanpa dicoba): predict_survival_function() pada
# loss selain 'coxph' melempar ValueError langsung dari scikit-survival
# ("`fit` must be called with the loss option set to 'coxph'.") - loss itu
# hanya menghasilkan skor/waktu titik, TIDAK punya baseline hazard model
# untuk kurva S(t). Seluruh pipeline di sini (evaluate.py IBS/Brier/AUC,
# predict.py, utils.survival_curve_arrays) BUTUH predict_survival_function()
# di SETIAP model - bukan cuma soal arah risk_sign, tapi ketidakcocokan
# struktural. Didokumentasikan di sini (bukan didiamkan), tidak masuk
# registry sama sekali.

# risk_sign dikalikan ke model.predict() sebelum masuk ke
# concordance_index_censored/ipcw & cumulative_dynamic_auc
# (src/evaluation.native_metrics) - SEMUA model harus dibandingkan dengan
# konvensi yang sama "skor lebih tinggi = lebih berisiko/lebih cepat gagal".
# RSF/ExtraSurvivalTrees ("total kejadian" dari cumulative hazard) dan Cox PH
# (log hazard ratio) serta GradientBoostingSurvivalAnalysis/Componentwise
# dengan loss='coxph' SUDAH mengikuti konvensi itu (risk_sign=1 untuk semua
# model di registry ini SAAT INI - kolom ini dipertahankan, bukan disingkat
# jadi konstanta, karena loss='ipcwls'/'squared' yang arahnya terbalik
# SEMPAT diuji sebelum dibuang karena alasan lain, lihat catatan di bawah
# DEFAULT_COMPONENTWISE_GBSA_PARAMS - kalau suatu saat model AFT-style
# ditambahkan lagi, risk_sign adalah tempatnya, bukan pembalikan ad-hoc di
# evaluation.py).
MODEL_REGISTRY = {
    "random_survival_forest": {
        "cls": RandomSurvivalForest, "default_params": DEFAULT_RSF_PARAMS, "risk_sign": 1,
    },
    "cox_ph": {
        "cls": CoxPHSurvivalAnalysis, "default_params": DEFAULT_COX_PARAMS, "risk_sign": 1,
    },
    "extra_survival_trees": {
        "cls": ExtraSurvivalTrees, "default_params": DEFAULT_EXTRA_TREES_PARAMS, "risk_sign": 1,
    },
    "gbsa_coxph": {
        "cls": GradientBoostingSurvivalAnalysis, "default_params": DEFAULT_GBSA_COXPH_PARAMS, "risk_sign": 1,
    },
    "componentwise_gbsa": {
        "cls": ComponentwiseGradientBoostingSurvivalAnalysis,
        "default_params": DEFAULT_COMPONENTWISE_GBSA_PARAMS, "risk_sign": 1,
    },
}
# Bawaan lama (RSF + Cox) - dipertahankan supaya train.py dan seluruh
# pemanggilan run_config()/fit_models() yang SUDAH ADA di experiments.py
# (threshold sweep, ablation, previous-cycle audit, RSF tuning) tetap
# berjalan identik tanpa perubahan apa pun di sisi pemanggil.
DEFAULT_MODEL_NAMES = ["random_survival_forest", "cox_ph"]


def make_survival_target(dataset, mask):
    return Surv.from_arrays(
        event=dataset.loc[mask, "event_observed"].astype(bool).to_numpy(),
        time=dataset.loc[mask, "duration_days"].to_numpy(),
    )


def fit_models(
    x_train, y_train, model_names: list[str] | None = None, params: dict[str, dict] | None = None
) -> dict:
    """Latih satu atau lebih model dari MODEL_REGISTRY.

    Bawaan (`model_names=None`) TETAP RSF + Cox PH berdampingan - Cox TIDAK
    PERNAH dibuang begitu saja di eksperimen mana pun, kalau ia menyamai/
    mengalahkan model lain pada suatu kombinasi fitur, itu temuan penting
    (bottleneck ada di fitur, bukan kompleksitas model), bukan sekadar
    baseline formalitas. `params[name]` override hyperparameter default satu
    model tertentu tanpa mempengaruhi model lain di panggilan yang sama.
    """
    names = model_names if model_names is not None else DEFAULT_MODEL_NAMES
    overrides = params or {}
    models: dict = {}
    for name in names:
        spec = MODEL_REGISTRY[name]
        model_params = overrides.get(name, spec["default_params"])
        models[name] = spec["cls"](**model_params).fit(x_train, y_train)
    return models


def evaluate_models(models: dict, y_train, x_val, y_val, x_test=None, y_test=None) -> dict:
    """Metrik native (C-index Harrell & Uno, IBS, Brier/AUC per horizon)
    lewat src.evaluation.native_metrics() - SATU fungsi dipakai train.py,
    evaluate.py, dan experiments.py, supaya angka antar tahap selalu
    dihitung dengan cara yang identik. risk_sign per model diambil dari
    MODEL_REGISTRY (lihat catatan di atas MODEL_REGISTRY) - model yang tidak
    terdaftar (seharusnya tidak terjadi lewat fit_models()) dianggap
    risk_sign=1."""
    metrics: dict = {}
    for name, model in models.items():
        risk_sign = MODEL_REGISTRY.get(name, {}).get("risk_sign", 1)
        metrics[name] = {"validation": evaluation.native_metrics(model, y_train, x_val, y_val, risk_sign=risk_sign)}
        if x_test is not None:
            metrics[name]["test"] = evaluation.native_metrics(model, y_train, x_test, y_test, risk_sign=risk_sign)
    return metrics
