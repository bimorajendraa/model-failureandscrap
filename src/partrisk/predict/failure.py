"""Prediksi risiko kerusakan untuk satu PART.

    from partrisk.predict.failure import predict
    predict("PART-A")

Pemanggil cukup memberi ID PART. Seluruh fitur dihitung sendiri dari riwayat
PART tersebut di database - backend tidak perlu tahu apa pun soal umur,
jumlah kerusakan, client, atau fitur lainnya.

Risiko untuk beberapa horizon dihitung dengan "hazard chaining": model 30
hari yang sama dipakai berulang, dengan fitur waktu dimajukan 30 hari setiap
langkah, lalu peluang bertahan dikalikan berantai:

    P(rusak dalam 30k hari) = 1 - hasil kali (1 - hazard tiap langkah)

Cara ini menjamin risiko 30 hari <= 60 hari <= 90 hari <= 120 hari secara
matematis, dan pada pengujian research terbukti lebih akurat daripada
melatih model terpisah untuk tiap horizon.
"""

from __future__ import annotations

import json
import sys

import joblib
import pandas as pd
from catboost import CatBoostClassifier

from partrisk import config
from partrisk.data import reader as data_reader
from partrisk.features import failure as feature_builder

_LOADED: tuple[CatBoostClassifier, object, dict] | None = None
_FLEET: object = None
_ITEM_TYPE_DENSITY: object = None


def _item_type_density_snapshot(data_end):
    """Laju kerusakan tiap item_type_at_install (90/180d) - pola cache sama
    dengan _fleet_snapshot() (potret dipakai ulang selama data belum
    bertambah), TANPA fast-path CSV tersimpan (dampaknya kecil - hanya
    beberapa kategori item_type, bukan ratusan model PART seperti fleet)."""
    global _ITEM_TYPE_DENSITY
    if _ITEM_TYPE_DENSITY is not None:
        return _ITEM_TYPE_DENSITY

    events = data_reader.get_events()
    cycles = data_reader.get_cycles()
    episodes = data_reader.get_failure_episodes()
    _ITEM_TYPE_DENSITY = feature_builder.item_type_density_snapshot(cycles, events, episodes, data_end)
    return _ITEM_TYPE_DENSITY


def _fleet_snapshot(data_end):
    """Kondisi armada tiap model PART.

    Perlu riwayat kerusakan SELURUH model PART, bukan hanya PART yang
    ditanyakan - membangunnya dari nol makan waktu sekitar 45 detik. Karena
    itu potret hasil training dipakai ulang SELAMA data belum bertambah;
    begitu ada kejadian baru di database, potretnya dihitung ulang supaya
    tidak pernah memakai angka yang basi.
    """
    global _FLEET
    if _FLEET is not None:
        return _FLEET

    _, _, metadata = _load_model()
    directory = config.FAILURE_MODEL_DIR / metadata["model_version"]
    stored = directory / "fleet_snapshot.csv"
    if stored.exists() and metadata.get("fleet_snapshot_at") == str(data_end):
        # dtype=str WAJIB: kode model punya nol di depan yang akan hilang
        # kalau dibaca sebagai angka, dan pencocokan gagal tanpa suara.
        snapshot = pd.read_csv(stored, dtype={"item_model_code_clean": str})
        if _covers_known_models(snapshot, metadata):
            _FLEET = snapshot
            return _FLEET

    cycles = data_reader.get_cycles()
    episodes = data_reader.get_failure_episodes()
    _FLEET = feature_builder.fleet_snapshot(cycles, episodes, data_end)
    return _FLEET


def _covers_known_models(snapshot, metadata: dict) -> bool:
    """Pastikan potret tersimpan benar-benar cocok dengan model PART yang
    dikenal.

    Penjaga ini ada karena kegagalannya SENYAP: kalau kode model tidak cocok,
    fitur armada diam-diam jadi nol dan prediksi tetap keluar - hanya saja
    salah. Lebih baik menghitung ulang daripada memakai potret yang tidak
    cocok.
    """
    known = set(metadata.get("part_model_support", {}))
    if not known:
        return True
    overlap = len(known & set(snapshot["item_model_code_clean"].astype(str)))
    return overlap >= 0.8 * len(known)


class ItemNotScorable(LookupError):
    """PART tidak dikenal, atau sedang tidak terpasang sehingga tidak ada
    risiko kerusakan yang perlu diperkirakan."""


def _load_model() -> tuple[CatBoostClassifier, object, dict]:
    """Muat model production sekali per proses."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED

    pointer = config.FAILURE_MODEL_DIR / "CURRENT"
    if not pointer.exists():
        raise FileNotFoundError(
            f"Belum ada model kerusakan di {config.FAILURE_MODEL_DIR}. "
            "Jalankan dulu: python train.py"
        )
    directory = config.FAILURE_MODEL_DIR / pointer.read_text(encoding="utf-8").strip()

    model = CatBoostClassifier()
    model.load_model(str(directory / "model.cbm"))
    calibrator = joblib.load(directory / "calibrator.joblib")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    _LOADED = (model, calibrator, metadata)
    return _LOADED


def _risk_level(probability: float, cutoffs: dict[str, float]) -> str:
    """Kelompokkan risiko memakai ambang tetap yang ditetapkan saat training.

    `probability` di sini adalah peluang kerusakan 30-hari yang sudah
    dikalibrasi - angka yang sama persis dengan yang dibaca pengguna di
    layar. Lihat FAILURE_HIGH/MEDIUM_PROBABILITY_THRESHOLD di config.py.
    """
    if probability >= cutoffs["high"]:
        return "HIGH"
    if probability >= cutoffs["medium"]:
        return "MEDIUM"
    return "LOW"


def predict(item_id: str) -> dict:
    """Perkirakan risiko kerusakan sebuah PART.

    Mengembalikan peluang kerusakan untuk tiap horizon, kelompok risiko, dan
    keterangan versi model serta tanggal data yang dipakai.

    Melempar ItemNotScorable kalau PART tidak dikenal atau sedang tidak
    terpasang.
    """
    model, calibrator, metadata = _load_model()

    data_end = data_reader.get_dataset_max_event_on()
    cycles = data_reader.get_cycles(item_id, data_end)
    if cycles.empty:
        raise ItemNotScorable(f"PART '{item_id}' tidak ditemukan di database.")

    snapshot = feature_builder.current_observations(cycles)
    if snapshot.empty:
        raise ItemNotScorable(
            f"PART '{item_id}' sedang tidak terpasang (sudah rusak atau sudah "
            "dipasang ulang), jadi tidak ada risiko yang perlu diperkirakan."
        )

    events = data_reader.get_events(item_id)
    snapshot = feature_builder.attach_history(snapshot, events)
    # Fitur degradasi butuh riwayat siklus/event, tapi HANYA milik PART ini
    # sendiri - cumulative_cycle_age dkk. group-by item_identifier_clean
    # secara internal, jadi cukup `cycles` per-item yang sudah diambil di
    # atas (bukan seluruh armada seperti attach_fleet_snapshot di bawah).
    snapshot = feature_builder.attach_degradation_history(snapshot, cycles, events)
    # Kondisi armada butuh riwayat SELURUH model PART, bukan hanya PART ini -
    # potretnya dibaca sekali per proses lalu dipakai ulang.
    snapshot = feature_builder.attach_fleet_snapshot(snapshot, _fleet_snapshot(data_end))
    # Local failure density per item_type_at_install - sama alasannya
    # dengan kondisi armada di atas (butuh SELURUH armada), potret dicache.
    snapshot = feature_builder.attach_item_type_density_snapshot(
        snapshot, events, _item_type_density_snapshot(data_end)
    )
    support = feature_builder.part_model_support(
        snapshot, metadata["part_model_support"]
    )

    # Hazard tiap 30 hari, lalu dirantai jadi risiko kumulatif.
    steps = max(config.PREDICTION_HORIZON_DAYS) // config.OBSERVATION_STEP_DAYS
    survival = 1.0
    cumulative_risk: dict[int, float] = {}
    for step in range(steps):
        features = feature_builder.project_features(snapshot, support, step)
        # build_features()/project_features() SELALU mengembalikan
        # config.FEATURE_COLUMNS (skema TERKINI, global) - model yang benar-
        # benar CURRENT bisa saja versi lama dengan daftar fitur lebih
        # sempit (metadata["features"]). Persempit dulu di sini - kalau
        # tidak, CatBoost diam-diam menerima kolom lebih banyak dari yang
        # dilihatnya saat training (kegagalan senyap, bukan error).
        features = features[metadata["features"]]
        raw = float(model.predict_proba(features)[:, 1][0])
        hazard = float(calibrator.predict([raw])[0])
        survival *= 1.0 - hazard
        cumulative_risk[(step + 1) * config.OBSERVATION_STEP_DAYS] = 1.0 - survival

    probabilities = {
        f"failure_probability_{days}d": round(cumulative_risk[days], 4)
        for days in config.PREDICTION_HORIZON_DAYS
    }

    return {
        "item_id": snapshot["item_identifier_clean"].iloc[0],
        **probabilities,
        "risk_level": _risk_level(
            probabilities["failure_probability_30d"], metadata["risk_cutoffs"]
        ),
        "model_version": metadata["model_version"],
        "as_of": str(snapshot["observation_on"].iloc[0]),
    }


def clear_fleet_cache() -> None:
    """Buang potret armada tersimpan supaya dibangun ulang di panggilan berikutnya.

    Dipanggil dari luar (inference/data_state.py) saat data terbukti
    bertambah - lihat docstring _fleet_snapshot(). Sekaligus membuang potret
    local density item_type (_item_type_density_snapshot) - sumber datanya
    sama (cycles/events/episodes), jadi basi bersamaan.
    """
    global _FLEET, _ITEM_TYPE_DENSITY
    _FLEET = None
    _ITEM_TYPE_DENSITY = None


# Nama publik untuk pemanggil di luar modul ini (inference/, batch scoring) -
# implementasinya tetap satu, tidak diduplikasi.
load_model = _load_model
fleet_snapshot = _fleet_snapshot
item_type_density_snapshot = _item_type_density_snapshot
risk_level = _risk_level


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Cara pakai: python -m partrisk.predict.failure <ITEM_ID>")
    try:
        print(json.dumps(predict(sys.argv[1]), indent=2, ensure_ascii=False))
    except ItemNotScorable as error:
        raise SystemExit(f"[TIDAK BISA DISKOR] {error}")
