"""Prediksi risiko sebuah PART dibuang (scrap).

    from predict_scrap import predict_scrap, predict_death_risk

    predict_scrap("PART-A")       # kalau PART ini rusak, apakah dibuang?
    predict_death_risk("PART-A")  # risiko MATI dalam 30 hari (gabungan)

TERPISAH dari predict.py; fungsi predict() yang lama tidak berubah sama sekali.

DUA CARA PAKAI, dan hanya yang pertama yang sudah teruji kuat:

1. Saat PART baru saja rusak - inilah pemakaian utamanya. Begitu ada kerusakan
   masuk, langsung ketahuan perlu siapkan pengganti atau tidak, tanpa menunggu
   vonis bengkel yang butuh ~3 hari.
   Teruji: ROC-AUC 0,76 dan 3,9x lebih baik daripada menebak.

2. Untuk PART yang masih sehat - dibaca sebagai "seandainya rusak besok".
   Digabung dengan model 30 hari lewat predict_death_risk(). Sudah dibacktest
   dan terbukti lebih baik daripada memakai model 30 hari sendirian, TETAPI
   kejadiannya sangat jarang (sekitar 2-3 PART mati per bulan dari belasan ribu
   PART aktif). Cocok sebagai daftar pantau perencanaan stok, BUKAN sebagai
   pemicu tindakan per PART.
"""

from __future__ import annotations

import json
import sys

import joblib

import config
import data_reader
import predict as failure_model
import scrap_features

_LOADED: tuple[object, object, dict] | None = None


class ItemNotScorable(LookupError):
    """PART tidak dikenal atau riwayatnya belum cukup untuk dinilai."""


def _load_model() -> tuple[object, object, dict]:
    global _LOADED
    if _LOADED is not None:
        return _LOADED

    pointer = config.SCRAP_MODEL_DIR / "CURRENT"
    if not pointer.exists():
        raise FileNotFoundError(
            f"Belum ada model scrap di {config.SCRAP_MODEL_DIR}. "
            "Jalankan dulu: python train_scrap.py"
        )
    directory = config.SCRAP_MODEL_DIR / pointer.read_text(encoding="utf-8").strip()
    model = joblib.load(directory / "model.joblib")
    calibrator = joblib.load(directory / "calibrator.joblib")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    _LOADED = (model, calibrator, metadata)
    return _LOADED


def _risk_level(probability: float, cutoffs: dict[str, float]) -> str:
    if probability >= cutoffs["high"]:
        return "HIGH"
    if probability >= cutoffs["medium"]:
        return "MEDIUM"
    return "LOW"


def predict_scrap(item_id: str) -> dict:
    """Kalau PART ini rusak, seberapa besar kemungkinan tidak bisa diperbaiki.

    Angkanya SELALU bersyarat "kalau rusak" - bukan peluang PART ini rusak.
    Untuk peluang rusaknya, pakai predict() dari predict.py.
    """
    model, calibrator, metadata = _load_model()

    data_end = data_reader.get_dataset_max_event_on()
    events = data_reader.get_events(item_id)
    if events.empty:
        raise ItemNotScorable(f"PART '{item_id}' tidak ditemukan di database.")

    cycles = data_reader.get_cycles(item_id, data_end)
    state = scrap_features.current_state(events, cycles, data_end)
    if state.empty:
        raise ItemNotScorable(f"PART '{item_id}' belum punya riwayat yang bisa dinilai.")

    features = scrap_features.build_features(state, metadata["known_item_types"])
    raw = model.predict_proba(features)[:, 1]
    probability = float(calibrator.predict_proba(raw.reshape(-1, 1))[:, 1][0])

    known = state["item_type_clean"].iloc[0] in metadata["known_item_types"]
    return {
        "item_id": state["item_identifier_clean"].iloc[0],
        # Sudah dikalibrasi, jadi boleh dibaca sebagai persentase - tetapi
        # PERKIRAAN, bukan angka pasti. Kalibrator dipasang pada data latih
        # yang tingkat scrap-nya 2,3%, sementara belakangan naik ke 6,5%,
        # sehingga angka ini cenderung MERENDAHKAN risiko sesungguhnya.
        # Urutannya tetap yang paling bisa dipercaya.
        "scrap_probability": round(probability, 4),
        "scrap_risk_level": _risk_level(probability, metadata["risk_cutoffs"]),
        "scrap_risk_basis": (
            "dibandingkan kerusakan lain yang masuk bengkel, bukan terhadap "
            "seluruh PART aktif"
        ),
        "item_type": state["item_type_clean"].iloc[0],
        # Jenis PART yang belum dikenal masuk kelompok "jarang", dan kelompok
        # itu cenderung diberi risiko tinggi. Ditandai supaya tidak dibaca
        # seolah model tahu sesuatu tentang jenis PART tersebut.
        "item_type_known_to_model": bool(known),
        "model_version": metadata["model_version"],
        "as_of": str(data_end),
    }


def predict_death_risk(item_id: str) -> dict:
    """Risiko PART benar-benar MATI dalam 30 hari, bukan sekadar rusak.

        P(mati) = P(rusak dalam 30 hari) x P(dibuang | rusak)

    Sudah dibacktest pada 74.412 observasi: gabungan ini lebih baik daripada
    model 30 hari sendirian (PR-AUC naik, 100% dari 500 resampling memihak
    gabungan). Tetapi kejadiannya sangat jarang, jadi pakailah sebagai daftar
    pantau untuk perencanaan stok - bukan pemicu tindakan per PART.
    """
    failure = failure_model.predict(item_id)
    scrap = predict_scrap(item_id)
    horizon = config.TARGET_HORIZON_DAYS
    failure_probability = failure[f"failure_probability_{horizon}d"]
    scrap_probability = scrap["scrap_probability"]

    return {
        "item_id": scrap["item_id"],
        f"failure_probability_{horizon}d": failure_probability,
        "scrap_probability": scrap_probability,
        # Kedua faktornya kini sama-sama terkalibrasi, jadi hasil kalinya
        # bisa dibaca sebagai perkiraan peluang - dengan catatan yang sama:
        # cenderung merendahkan, dan urutannya lebih bisa dipercaya.
        f"death_probability_{horizon}d": round(failure_probability * scrap_probability, 5),
        "failure_risk_level": failure["risk_level"],
        "scrap_risk_level": scrap["scrap_risk_level"],
        "item_type_known_to_model": scrap["item_type_known_to_model"],
        "model_version": {
            "failure": failure["model_version"],
            "scrap": scrap["model_version"],
        },
        "as_of": scrap["as_of"],
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Cara pakai: python predict_scrap.py <ITEM_ID>")
    try:
        print(json.dumps(predict_death_risk(sys.argv[1]), indent=2, ensure_ascii=False))
    except (ItemNotScorable, failure_model.ItemNotScorable) as error:
        raise SystemExit(f"[TIDAK BISA DISKOR] {error}")
