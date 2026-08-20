"""Fit Random Survival Forest + Cox PH dan evaluasi native.

Diekstrak dari train.py supaya `experiments.py` (puluhan model kandidat:
ablation, threshold sweep, tuning) memakai logic fitting/evaluasi yang SAMA
PERSIS dengan model production - tidak ada logic yang di-duplikasi/berbeda
antara eksperimen dan hasil akhir.
"""

from __future__ import annotations

from sksurv.ensemble import RandomSurvivalForest
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


def make_survival_target(dataset, mask):
    return Surv.from_arrays(
        event=dataset.loc[mask, "event_observed"].astype(bool).to_numpy(),
        time=dataset.loc[mask, "duration_days"].to_numpy(),
    )


def fit_models(x_train, y_train, rsf_params: dict | None = None, cox_params: dict | None = None) -> dict:
    """Latih RSF + Cox PH berdampingan. Cox TIDAK PERNAH dibuang begitu saja
    di eksperimen mana pun - kalau ia menyamai/mengalahkan RSF pada suatu
    kombinasi fitur, itu temuan penting (bottleneck ada di fitur, bukan
    kompleksitas model), bukan sekadar baseline formalitas."""
    rsf = RandomSurvivalForest(**(rsf_params or DEFAULT_RSF_PARAMS)).fit(x_train, y_train)
    cox = CoxPHSurvivalAnalysis(**(cox_params or DEFAULT_COX_PARAMS)).fit(x_train, y_train)
    return {"random_survival_forest": rsf, "cox_ph": cox}


def evaluate_models(models: dict, y_train, x_val, y_val, x_test=None, y_test=None) -> dict:
    """Metrik native (C-index Harrell & Uno, IBS, Brier/AUC per horizon)
    lewat src.evaluation.native_metrics() - SATU fungsi dipakai train.py,
    evaluate.py, dan experiments.py, supaya angka antar tahap selalu
    dihitung dengan cara yang identik."""
    metrics: dict = {}
    for name, model in models.items():
        metrics[name] = {"validation": evaluation.native_metrics(model, y_train, x_val, y_val)}
        if x_test is not None:
            metrics[name]["test"] = evaluation.native_metrics(model, y_train, x_test, y_test)
    return metrics
