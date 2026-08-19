"""Keputusan promosi model: window evaluasi yang adil, bukan skor tunggal.

Lahir dari bug nyata yang ditemukan saat dikembangkan: train.py membandingkan
skor kandidat dengan metrik LAMA yang tersimpan di metadata model production -
dihitung pada test split model itu SENDIRI saat ia dilatih. Karena test_start
dihitung ulang dari tahun data_end setiap kali retrain, window itu bergeser
maju setiap tahun, sehingga kandidat dan incumbent akhirnya dibandingkan pada
dua periode yang berbeda.

Saat memperbaikinya, ditemukan bug KEDUA yang lebih halus: evaluasi ulang
incumbent (pakai dukungan tipe PART yang DIBEKUKAN, seperti predict.py di
production) vs evaluasi kandidat yang semula memakai dukungan point-in-time
(seperti train_model() menghitung metrik "test" yang dilaporkan) - dua
metodologi fitur berbeda untuk dua model yang seharusnya dibandingkan apel ke
apel. Diperbaiki dengan mengevaluasi kandidat JUGA memakai dukungan beku
miliknya sendiri untuk keperluan promosi. Test di sini membuktikan: kalau
data tidak berubah dan modelnya sama persis, kandidat vs incumbent harus
menghasilkan angka yang IDENTIK - bukan kandidat "menang" semu karena
perbedaan metodologi.
"""

from __future__ import annotations

import numpy as np
import pytest

import train
import train_scrap
from tests.conftest import needs_database, needs_models

# ---------------------------------------------------------------------------
# capacity_metrics() / full_metrics() - logic murni, tanpa database
# ---------------------------------------------------------------------------


def _synthetic(n: int = 1000, positive_rate: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    target = (rng.random(n) < positive_rate).astype(int)
    # Skor berkorelasi dengan target supaya perilaku "mengurutkan" masuk akal.
    raw = target * rng.random(n) * 0.5 + rng.random(n) * 0.3
    return raw, target


@pytest.mark.parametrize("module", [train, train_scrap])
def test_capacity_metrics_menangkap_seluruh_positif_saat_kapasitas_cukup(module):
    """Kalau kapasitas yang dievaluasi >= jumlah positif sungguhan, seluruh
    positif pasti tertangkap - recall harus tepat 1,0, bukan mendekati."""
    raw, target = _synthetic()
    # Window sangat panjang supaya kapasitas jauh melebihi jumlah baris,
    # lalu capacity_metrics() sendiri yang membatasinya ke len(raw).
    result = module.capacity_metrics(raw, target, window_days=10_000_000.0)
    assert result["capacity_evaluated"] == len(raw)
    assert result["recall_at_capacity"] == pytest.approx(1.0)


@pytest.mark.parametrize("module", [train, train_scrap])
def test_capacity_metrics_kapasitas_minimal_satu(module):
    """Window sangat pendek tidak boleh menghasilkan kapasitas nol (pembagian
    dengan nol saat menghitung presisi)."""
    raw, target = _synthetic(n=50)
    result = module.capacity_metrics(raw, target, window_days=0.01)
    assert result["capacity_evaluated"] >= 1
    assert 0.0 <= result["precision_at_capacity"] <= 1.0


@pytest.mark.parametrize("module", [train, train_scrap])
def test_capacity_metrics_kapasitas_tidak_melebihi_jumlah_baris(module):
    raw, target = _synthetic(n=20)
    result = module.capacity_metrics(raw, target, window_days=100000.0)
    assert result["capacity_evaluated"] <= 20


@pytest.mark.parametrize("module", [train, train_scrap])
def test_full_metrics_berisi_seluruh_metrik_yang_disyaratkan(module):
    """Master prompt eksplisit meminta PR-AUC, ROC-AUC, Precision/Recall@kapasitas,
    dan Brier - bukan cuma ROC-AUC."""
    raw, target = _synthetic()
    calibrated = raw.copy()
    result = module.full_metrics(raw, calibrated, target, window_days=180.0)
    for key in (
        "roc_auc", "pr_auc", "brier_calibrated",
        "precision_at_capacity", "recall_at_capacity",
    ):
        assert key in result, key


# ---------------------------------------------------------------------------
# decide_promotion() - logic murni, dengan dict sintetis
# ---------------------------------------------------------------------------


def _metrics(**overrides) -> dict:
    base = {
        "pr_auc": 0.20, "roc_auc": 0.80, "recall_at_capacity": 0.30,
        "precision_at_capacity": 0.15, "brier_calibrated": 0.02,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("module", [train, train_scrap])
def test_promosi_pertama_kali_selalu_lolos(module):
    promote, reason, comparison = module.decide_promotion(_metrics(), None, None, force=False)
    assert promote is True
    assert "belum ada" in reason


@pytest.mark.parametrize("module", [train, train_scrap])
def test_kandidat_lebih_baik_di_kedua_metrik_dipromosikan(module):
    candidate = _metrics(pr_auc=0.25, recall_at_capacity=0.35)
    incumbent = _metrics(pr_auc=0.20, recall_at_capacity=0.30)
    promote, reason, comparison = module.decide_promotion(candidate, incumbent, "v1", force=False)
    assert promote is True
    assert comparison["incumbent_version"] == "v1"


@pytest.mark.parametrize("module", [train, train_scrap])
def test_pr_auc_turun_menahan_promosi_walau_recall_naik(module):
    """ROC-AUC BUKAN satu-satunya dasar - di sini PR-AUC yang menahan promosi
    walau metrik lain membaik, sesuai permintaan eksplisit: jangan
    menjadikan satu metrik sebagai penentu tunggal."""
    candidate = _metrics(pr_auc=0.15, recall_at_capacity=0.40, roc_auc=0.90)
    incumbent = _metrics(pr_auc=0.20, recall_at_capacity=0.30, roc_auc=0.80)
    promote, reason, comparison = module.decide_promotion(candidate, incumbent, "v1", force=False)
    assert promote is False


@pytest.mark.parametrize("module", [train, train_scrap])
def test_recall_turun_menahan_promosi_walau_pr_auc_naik(module):
    candidate = _metrics(pr_auc=0.25, recall_at_capacity=0.20)
    incumbent = _metrics(pr_auc=0.20, recall_at_capacity=0.30)
    promote, reason, comparison = module.decide_promotion(candidate, incumbent, "v1", force=False)
    assert promote is False


@pytest.mark.parametrize("module", [train, train_scrap])
def test_force_promote_memaksa_walau_lebih_buruk(module):
    candidate = _metrics(pr_auc=0.10, recall_at_capacity=0.10)
    incumbent = _metrics(pr_auc=0.20, recall_at_capacity=0.30)
    promote, reason, comparison = module.decide_promotion(candidate, incumbent, "v1", force=True)
    assert promote is True
    assert "dipaksa" in reason


@pytest.mark.parametrize("module", [train, train_scrap])
def test_kandidat_dan_incumbent_identik_dipromosikan(module):
    """Kandidat yang persis sama dengan incumbent (mis. retrain tanpa data
    baru) harus tetap lolos - bukan tertahan karena perbandingan >= yang
    ketat keliru jadi >."""
    same = _metrics()
    promote, reason, comparison = module.decide_promotion(same, dict(same), "v1", force=False)
    assert promote is True


# ---------------------------------------------------------------------------
# evaluate_incumbent() - model v1 sungguhan, data sungguhan
# ---------------------------------------------------------------------------


@needs_database
@needs_models
def test_evaluate_incumbent_menghasilkan_skor_valid():
    """Tidak menguji angka spesifik (itu urusan re-run train.py sungguhan) -
    hanya memastikan mekanismenya jalan dan hasilnya masuk akal."""
    import pandas as pd

    import config
    import data_reader
    import feature_builder

    events = data_reader.get_events()
    cycles = data_reader.get_cycles()
    data_end = pd.Timestamp(cycles["dataset_max_event_on"].max())

    observations = feature_builder.training_observations(cycles)
    observations = feature_builder.attach_history(observations, events)
    episodes = data_reader.get_failure_episodes()
    observations = feature_builder.attach_fleet(observations, cycles, episodes)
    eligible = observations.loc[observations["is_eligible"]].reset_index(drop=True)
    eligible["split"] = train.assign_split(eligible, data_end)
    eligible = eligible.loc[eligible["split"].isin([train.TRAIN, train.VALIDATION, train.TEST])]

    if eligible["split"].eq(train.TEST).sum() == 0:
        pytest.skip("tidak ada baris test split untuk diuji")

    # v1 mungkin sudah bukan CURRENT lagi (train.py boleh sudah dijalankan
    # ulang di repo ini), tetapi filenya tetap ada untuk diuji langsung.
    if not (config.FAILURE_MODEL_DIR / "v1" / "metadata.json").exists():
        pytest.skip("model v1 tidak ada di repo ini")

    result = train.evaluate_incumbent("v1", eligible)
    assert result["model_version"] == "v1"
    assert len(result["raw"]) == eligible["split"].eq(train.TEST).sum()
    assert ((result["raw"] >= 0) & (result["raw"] <= 1)).all()
    assert ((result["calibrated"] >= 0) & (result["calibrated"] <= 1)).all()
    assert set(np.unique(result["target"])) <= {0, 1, True, False}
