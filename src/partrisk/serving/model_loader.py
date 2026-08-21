"""Akses terpusat ke model production yang sedang dipakai.

Modul ini TIDAK memuat model sendiri. predict.py dan predict_scrap.py sudah
menyimpan hasil muatnya di variabel level-modul dan sudah mengikuti mekanisme
versi models/<nama>/CURRENT yang ada; di sini keduanya hanya dipanggil sekali
saat aplikasi start supaya request pertama tidak menanggung biaya muat, lalu
metadata-nya dibagikan ke route /api/v1/model.
"""

from __future__ import annotations

from partrisk import config
from partrisk import predict as failure_model
from partrisk import predict_scrap as scrap_model
from partrisk.serving.errors import ModelUnavailable


def failure_metadata() -> dict:
    try:
        return failure_model.load_model()[2]
    except FileNotFoundError as error:
        raise ModelUnavailable(str(error)) from error


def scrap_metadata() -> dict:
    try:
        return scrap_model.load_model()[2]
    except FileNotFoundError as error:
        raise ModelUnavailable(str(error)) from error


def versions() -> dict[str, str]:
    return {
        "failure": failure_metadata()["model_version"],
        "scrap": scrap_metadata()["model_version"],
    }


def warmup() -> None:
    """Muat kedua model ke memori. Dipanggil sekali saat aplikasi start."""
    failure_metadata()
    scrap_metadata()


def describe() -> dict:
    """Ringkasan model production untuk endpoint /api/v1/model.

    Hanya menyalin dari metadata yang ditulis train.py / train_scrap.py -
    tidak ada angka yang dihitung ulang di sini.
    """
    failure = failure_metadata()
    scrap = scrap_metadata()
    return {
        "failure": {
            "model_version": failure["model_version"],
            "training_date": failure["training_date"],
            "target": failure["target"],
            "horizons_days": config.PREDICTION_HORIZON_DAYS,
            "features": failure["features"],
            "risk_cutoffs": failure["risk_cutoffs"],
            "cutoff_basis": failure["cutoff_basis"],
            "test_metrics": failure["evaluation_metrics"]["test"],
            "data_through": failure["training_period"]["dataset_max_event_on"],
        },
        "scrap": {
            "model_version": scrap["model_version"],
            "training_date": scrap["training_date"],
            "target": scrap["target"],
            "selected_model": scrap["selected_model"],
            "features": scrap["features"],
            "risk_cutoffs": scrap["risk_cutoffs"],
            "cutoff_basis": scrap["cutoff_basis"],
            "known_item_types": scrap["known_item_types"],
            "data_through": scrap["training_period"]["onset_to"],
        },
        "notes": {
            "failure_probability": (
                "Peluang PART mengalami kerusakan dalam N hari ke depan. "
                "Model tidak memperkirakan tanggal kerusakan."
            ),
            "scrap_probability": (
                "Bersyarat: peluang PART tidak bisa diperbaiki JIKA rusak - "
                "bukan peluang PART ini rusak."
            ),
        },
    }
