"""Fondasi monitoring - metrik untuk diamati, BUKAN retraining otomatis.

Sengaja berhenti di "menyediakan angka", bukan "mengambil keputusan": tidak
ada alert, tidak ada trigger retraining. Itu pekerjaan tahap berikutnya,
setelah monitoring ini terbukti stabil - persis urutan yang diminta.

Dua jenis metrik dicampur di sini, dan keduanya diberi label jelas supaya
tidak tertukar:

- Metrik OFFLINE (dari training): PR-AUC, ROC-AUC, Precision/Recall@kapasitas,
  Brier - dibaca APA ADANYA dari metadata.json model yang sedang production
  (lihat train.py: metrik ini dihitung sekali saat training/promosi, bukan
  dihitung ulang di sini). Berguna sebagai KONTEKS, dan sebagai pengaman:
  kalau CURRENT tertukar ke model yang lebih buruk secara manual, angka ini
  langsung menunjukkannya.
- Metrik LIVE (dari populasi PART aktif SEKARANG): sebaran skor, jumlah
  HIGH/MEDIUM, pangsa kategori tipe PART yang tidak dikenal model
  (part_model_category UNKNOWN/LOW_SUPPORT), ringkasan fitur numerik, dan
  probabilitas scrap - dihitung dari hasil batch_predictor yang sudah ada,
  TANPA query tambahan.

Tidak ada label ground-truth untuk PART yang sedang aktif (belum diketahui
apakah nanti benar rusak), jadi PR-AUC/ROC-AUC LIVE secara matematis tidak
bisa dihitung di sini - itulah kenapa dua kelompok metrik di atas dipisah
tegas, bukan dicampur jadi satu angka yang menyesatkan.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
import predict as failure_model
import predict_scrap as scrap_model
from inference import batch_predictor, explanation

# Kolom numerik dari snapshot yang diringkas untuk deteksi feature drift -
# subset dari explanation.SOURCE_COLUMNS yang murni numerik (bukan
# item_model_code_clean/has_previous_cycle).
_DRIFT_COLUMNS = [
    "days_since_installation",
    "total_prior_events",
    "prior_failure_count",
    "prior_failure_365d",
    "prior_corrective_count",
    "model_failure_rate_90d",
]


def _score_distribution(scores: np.ndarray) -> dict:
    if len(scores) == 0:
        return {}
    percentiles = np.percentile(scores, [5, 25, 50, 75, 95])
    return {
        "min": round(float(scores.min()), 6),
        "p05": round(float(percentiles[0]), 6),
        "p25": round(float(percentiles[1]), 6),
        "median": round(float(percentiles[2]), 6),
        "p75": round(float(percentiles[3]), 6),
        "p95": round(float(percentiles[4]), 6),
        "max": round(float(scores.max()), 6),
        "mean": round(float(scores.mean()), 6),
    }


def _unknown_category_share(snapshot: pd.DataFrame, support_by_model: dict[str, int]) -> dict:
    """Pangsa PART aktif yang kode modelnya TIDAK dikenal baik oleh model
    (masuk UNKNOWN atau LOW_SUPPORT_LABEL saat fitur dibangun).

    Naik drastis berarti armada mulai didominasi tipe PART yang jarang
    dilihat model saat training - sinyal awal model perlu dilatih ulang
    dengan data yang mencakup tipe itu, sebelum akurasinya ikut turun.
    """
    codes = snapshot["item_model_code_clean"]
    support = codes.map(support_by_model).fillna(0)
    unknown = codes.isna() | (support < config.MIN_PART_MODEL_SUPPORT)
    return {
        "unknown_or_low_support_parts": int(unknown.sum()),
        "unknown_or_low_support_share": (
            round(float(unknown.mean()), 4) if len(codes) else 0.0
        ),
        "distinct_model_codes_active": int(codes.dropna().nunique()),
        "distinct_model_codes_in_training": len(support_by_model),
    }


def _feature_summary(snapshot: pd.DataFrame) -> dict:
    """Ringkasan tendensi sentral fitur numerik utama - bahan pembanding
    manual/otomatis di kemudian hari terhadap ringkasan periode training.

    Bukan uji statistik drift (KS-test/PSI) - itu di luar cakupan fondasi
    ini. Sengaja ringkas: median dan mean cukup untuk melihat pergeseran
    kasar tanpa membangun infrastruktur pembanding penuh.
    """
    summary = {}
    for column in _DRIFT_COLUMNS:
        if column not in snapshot.columns:
            continue
        values = pd.to_numeric(snapshot[column], errors="coerce").dropna()
        if values.empty:
            continue
        summary[column] = {
            "mean": round(float(values.mean()), 4),
            "median": round(float(values.median()), 4),
            "missing_share": round(float(snapshot[column].isna().mean()), 4),
        }
    return summary


def failure_monitoring() -> dict:
    """Metrik monitoring model kerusakan: offline (dari training) + live
    (dari populasi PART aktif sekarang)."""
    metadata = failure_model._load_model()[2]
    scores = batch_predictor.score_active_parts()
    frame = scores.frame

    offline = {
        "model_version": metadata["model_version"],
        "training_date": metadata["training_date"],
        "test_metrics": metadata["evaluation_metrics"]["test"],
        "validation_base_rate": metadata["validation_base_rate"],
        # Ada hanya untuk model yang sudah lewat promosi versi baru (lihat
        # train.py:evaluate_incumbent) - None untuk model pertama.
        "last_promotion_comparison": metadata.get("promotion_comparison"),
    }

    tier_score = frame["tier_score"].to_numpy(dtype=float)
    level_counts = frame["failure_risk_level"].value_counts().to_dict()
    expected_high = metadata["cutoff_basis"]["flagged_high"]
    actual_high = int(level_counts.get("HIGH", 0))

    live = {
        "active_parts": int(len(frame)),
        "score_distribution": _score_distribution(tier_score),
        "risk_level_counts": {
            "HIGH": actual_high,
            "MEDIUM": int(level_counts.get("MEDIUM", 0)),
            "LOW": int(level_counts.get("LOW", 0)),
        },
        # Jumlah HIGH SEHARUSNYA dekat dengan kapasitas kerja yang dipakai
        # menyetel ambang (lihat train.py:choose_cutoffs) - kalau populasi
        # PART aktif belum banyak berubah sejak training. Menjauh jauh dari
        # expected berarti populasi sudah bergeser sejak model dilatih.
        "expected_high_from_training": expected_high,
        "high_count_ratio_vs_training": (
            round(actual_high / expected_high, 3) if expected_high else None
        ),
        "category_coverage": _unknown_category_share(
            scores.snapshot, metadata["part_model_support"]
        ),
        "feature_summary": _feature_summary(scores.snapshot),
        "data_through": str(scores.data_end),
    }

    return {"offline": offline, "live": live}


def scrap_monitoring() -> dict:
    """Metrik monitoring model scrap: offline (dari training) + live (dari
    populasi PART aktif yang kerusakannya baru saja tercatat, kalau ada)."""
    metadata = scrap_model._load_model()[2]
    scores = batch_predictor.score_active_parts()
    frame = scores.frame

    offline = {
        "model_version": metadata["model_version"],
        "training_date": metadata["training_date"],
        "evaluation_metrics": metadata["evaluation_metrics"],
        "cutoff_basis": metadata["cutoff_basis"],
        "last_promotion_comparison": metadata.get("promotion_comparison"),
    }

    scrap_probability = frame["scrap_probability"].dropna().to_numpy(dtype=float)
    known_types = set(metadata["known_item_types"])
    item_types = frame["item_type"].dropna()
    unknown_types = ~item_types.isin(known_types)

    live = {
        "parts_with_scrap_score": int(len(scrap_probability)),
        "predicted_scrap_probability_distribution": _score_distribution(scrap_probability),
        # BUKAN scrap rate historis sungguhan (itu perlu event kerusakan
        # nyata, lihat scrap_features.py) - ini rata-rata PREDIKSI model
        # untuk PART yang sedang aktif, dilabeli jelas supaya tidak tertukar
        # dengan tingkat scrap yang benar-benar terjadi.
        "predicted_scrap_probability_mean": (
            round(float(scrap_probability.mean()), 4) if len(scrap_probability) else None
        ),
        "risk_level_counts": frame["scrap_risk_level"].value_counts().to_dict(),
        "unknown_item_type_share": (
            round(float(unknown_types.mean()), 4) if len(item_types) else 0.0
        ),
        "data_through": str(scores.data_end),
    }

    return {"offline": offline, "live": live}


def summary() -> dict:
    """Satu jawaban gabungan untuk /api/v1/monitoring/metrics."""
    return {
        "failure": failure_monitoring(),
        "scrap": scrap_monitoring(),
    }
