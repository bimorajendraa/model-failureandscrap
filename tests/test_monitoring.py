"""Fondasi monitoring: metrik offline (training) dan live (populasi aktif)
tidak boleh tertukar - PR-AUC/ROC-AUC LIVE tidak bisa dihitung sungguhan
karena PART yang sedang aktif belum punya label ground-truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from partrisk import config
from partrisk.api.services import monitoring_service
from tests.conftest import needs_database, needs_models

# ---------------------------------------------------------------------------
# Logic murni - tanpa database
# ---------------------------------------------------------------------------


def test_score_distribution_kosong_untuk_array_kosong():
    assert monitoring_service._score_distribution(np.array([])) == {}


def test_score_distribution_urut_naik():
    scores = np.random.default_rng(0).random(500)
    result = monitoring_service._score_distribution(scores)
    assert result["min"] <= result["p05"] <= result["p25"] <= result["median"]
    assert result["median"] <= result["p75"] <= result["p95"] <= result["max"]


def test_unknown_category_share_menghitung_dukungan_rendah():
    snapshot = pd.DataFrame({
        "item_model_code_clean": ["A", "A", "B", None, "C"],
    })
    support = {"A": config.MIN_PART_MODEL_SUPPORT + 100, "B": 5}  # C tidak ada di training
    result = monitoring_service._unknown_category_share(snapshot, support)
    # A (dukungan tinggi) x2 -> dikenal. B (dukungan < ambang), None, dan C
    # (tidak ada di dict) -> tidak dikenal. Total 3 dari 5.
    assert result["unknown_or_low_support_parts"] == 3
    assert result["unknown_or_low_support_share"] == pytest.approx(0.6)
    assert result["distinct_model_codes_active"] == 3  # A, B, C (None dibuang)
    assert result["distinct_model_codes_in_training"] == 2


def test_unknown_category_share_semua_dikenal():
    snapshot = pd.DataFrame({"item_model_code_clean": ["A", "A", "B"]})
    support = {"A": 1000, "B": 1000}
    result = monitoring_service._unknown_category_share(snapshot, support)
    assert result["unknown_or_low_support_share"] == 0.0


def test_feature_summary_mengabaikan_kolom_yang_tidak_ada():
    snapshot = pd.DataFrame({"days_since_installation": [10.0, 20.0, np.nan]})
    result = monitoring_service._feature_summary(snapshot)
    assert "days_since_installation" in result
    assert result["days_since_installation"]["missing_share"] == pytest.approx(1 / 3, abs=1e-4)
    # Kolom lain di _DRIFT_COLUMNS tidak ada di snapshot -> tidak boleh
    # muncul di hasil (bukan error, bukan nilai karangan).
    assert "prior_failure_count" not in result


def test_feature_summary_kolom_seluruhnya_kosong_tidak_meledak():
    snapshot = pd.DataFrame({"days_since_installation": [np.nan, np.nan]})
    result = monitoring_service._feature_summary(snapshot)
    assert "days_since_installation" not in result


# ---------------------------------------------------------------------------
# Integrasi - database dan model sungguhan
# ---------------------------------------------------------------------------


@needs_database
@needs_models
def test_failure_monitoring_memisahkan_offline_dan_live():
    result = monitoring_service.failure_monitoring()
    assert set(result.keys()) == {"offline", "live"}

    offline = result["offline"]
    assert offline["model_version"]
    assert "roc_auc" in offline["test_metrics"]
    assert "pr_auc" in offline["test_metrics"]

    live = result["live"]
    assert live["active_parts"] > 0
    assert live["risk_level_counts"]["HIGH"] >= 0
    # Metrik LIVE TIDAK boleh mengklaim py-auc/roc-auc - tidak ada ground
    # truth untuk PART yang sedang aktif.
    assert "roc_auc" not in live
    assert "pr_auc" not in live


@needs_database
@needs_models
def test_jumlah_high_live_dan_expected_konsisten_secara_struktur():
    result = monitoring_service.failure_monitoring()
    live = result["live"]
    total = sum(live["risk_level_counts"].values())
    assert total == live["active_parts"]
    if live["expected_high_from_training"]:
        assert live["high_count_ratio_vs_training"] == pytest.approx(
            live["risk_level_counts"]["HIGH"] / live["expected_high_from_training"], abs=1e-6
        )


@needs_database
@needs_models
def test_scrap_monitoring_prediksi_dilabeli_jelas_bukan_tingkat_sungguhan():
    result = monitoring_service.scrap_monitoring()
    live = result["live"]
    # Nama field harus eksplisit "predicted" - lihat docstring modul: ini
    # BUKAN scrap rate historis sungguhan.
    assert "predicted_scrap_probability_mean" in live
    assert "predicted_scrap_probability_distribution" in live


@needs_database
@needs_models
def test_endpoint_monitoring_metrics(client):
    response = client.get("/api/v1/monitoring/metrics")
    assert response.status_code == 200
    body = response.json()
    assert "failure" in body and "scrap" in body


@needs_database
@needs_models
def test_endpoint_monitoring_terpisah_per_model(client):
    failure_only = client.get("/api/v1/monitoring/metrics/failure").json()
    scrap_only = client.get("/api/v1/monitoring/metrics/scrap").json()
    assert "offline" in failure_only and "live" in failure_only
    assert "offline" in scrap_only and "live" in scrap_only
