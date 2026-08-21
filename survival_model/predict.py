"""Prediksi risiko survival satu PART aktif - CLI sederhana.

    python predict.py <item_id>

Model diprediksi dari fitur BASELINE INSTALASI (kondisi PART persis saat
siklus AKTIFNYA SEKARANG dimulai) - BUKAN kondisi PART hari ini. Proyeksi
risiko ke depan hanya memperhitungkan berlalunya waktu sejak instalasi, lewat
P(fail<=N hari | selamat sampai umur sekarang) = 1 - S(age+N)/S(age). Lihat
README.md bagian "Keterbatasan: baseline instalasi vs kondisi sekarang".

Tidak pernah mengklaim tanggal kerusakan pasti - hanya risiko dalam horizon
dan (kalau valid dalam rentang follow-up training) median waktu bertahan.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SURVIVAL_DIR = Path(__file__).resolve().parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))

import joblib
import pandas as pd

from partrisk import data_reader

from src import features, utils

ARTIFACTS_DIR = SURVIVAL_DIR / "artifacts"
HORIZONS_DAYS = [30, 60, 90, 120]
CURVE_STEP_DAYS = 30
CURVE_MAX_DAYS = 1080  # 3 tahun - cukup untuk dilihat, tidak membanjiri output CLI


class ItemNotScorable(Exception):
    """PART tidak ditemukan sedang aktif, atau identitas tipe PART-nya tidak
    bisa dipastikan dari inventory - sama seperti predict.py model classification."""


def _load_primary_model():
    if not (ARTIFACTS_DIR / "models.joblib").exists():
        raise SystemExit(f"Model belum dilatih. Jalankan dulu: python train.py (artifacts belum ada di {ARTIFACTS_DIR})")
    models = joblib.load(ARTIFACTS_DIR / "models.joblib")
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    primary_name = metadata["primary_model"]
    model = models[primary_name]
    # Single-thread saat prediksi - lihat catatan di evaluate.py load_artifacts():
    # model yang di-unpickle lalu diprediksi dengan n_jobs=-1 (bawaan training)
    # membuat proses hang saat exit (loky worker pool), bukan saat komputasi.
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    encoder = joblib.load(ARTIFACTS_DIR / "encoder.joblib")
    support_totals = {k: int(v) for k, v in metadata["support_totals"].items()}
    item_type_support_totals = {k: int(v) for k, v in metadata["item_type_support_totals"].items()}
    return model, primary_name, encoder, support_totals, item_type_support_totals


def predict(item_id: str) -> dict:
    model, model_name, encoder, support_totals, item_type_support_totals = _load_primary_model()

    dataset_max_event_on = data_reader.get_dataset_max_event_on()
    cycles_for_item = data_reader.get_cycles(item_id=item_id, dataset_max_event_on=dataset_max_event_on)
    active = cycles_for_item.loc[cycles_for_item["cycle_end_reason"].eq("RIGHT_CENSORED_AT_DATA_END")]
    if active.empty:
        raise ItemNotScorable(f"{item_id}: tidak ada siklus yang sedang aktif (PART tidak terpasang sekarang).")
    if not bool(active.iloc[0]["is_initial_model_cohort"]):
        raise ItemNotScorable(f"{item_id}: identitas tipe PART tidak bisa dipastikan dari inventory - tidak diskor.")

    active_cycle = active.iloc[[0]].reset_index(drop=True)
    installed_on = pd.Timestamp(active_cycle.loc[0, "installed_on"])
    age_days = float((dataset_max_event_on - installed_on).total_seconds() / 86400.0)

    # Riwayat cukup dibaca untuk item ini (attach_history mengelompokkan per
    # item); kondisi armada butuh populasi penuh, sama seperti predict.py
    # model classification.
    events = data_reader.get_events(item_id=item_id)
    all_cycles = data_reader.get_cycles()
    episodes = data_reader.get_failure_episodes()

    observations = features.build_baseline_observations(active_cycle)
    observations = features.attach_survival_features(observations, events, all_cycles, episodes)
    observations = features.attach_final_context(observations, events, all_cycles)
    # Dukungan historis (part_model DAN item_type_at_install) pakai angka
    # yang DIBEKUKAN saat train.py - bukan dihitung ulang dari 1 baris ini
    # sendiri (yang akan selalu memberi dukungan=1) - konsisten dengan alasan
    # yang sama di feature_builder.part_model_support() (model classification).
    support = observations["item_model_code_clean"].map(support_totals).fillna(0).astype("int64")
    item_type_support = observations["item_type_at_install"].map(item_type_support_totals).fillna(0).astype("int64")
    feature_frame = features.compute_features(observations, support, item_type_support)
    x = features.encode(feature_frame, encoder)

    times_grid, curves = utils.survival_curve_arrays(model, x)
    curve = curves[0]

    # Kalau umur PART SEKARANG sudah melebihi follow-up terpanjang yang
    # pernah dilihat training (~model.event_times_.max()), S(age) hanya
    # ekstrapolasi RATA dari nilai terakhir yang diketahui - risiko bersyarat
    # S(age+N)/S(age) jadi bernilai 0 secara matematis (bukan karena PART ini
    # benar-benar aman, tapi karena tidak ada informasi sama sekali di umur
    # setua ini). Melaporkan 0,0 begitu saja menyesatkan (terbaca "pasti
    # aman"), jadi risknya dikosongkan (None) dengan penjelasan eksplisit -
    # sama seperti evaluate.py tidak memaksakan horizon di luar follow-up.
    beyond_training_followup = bool(age_days > times_grid.max())
    if beyond_training_followup:
        risk = {f"risk_{h}d": None for h in HORIZONS_DAYS}
    else:
        risk = {
            f"risk_{h}d": round(utils.conditional_risk(times_grid, curve, age_days, float(h)), 4)
            for h in HORIZONS_DAYS
        }
    median_days_from_install = utils.median_survival_time(times_grid, curve)

    curve_days = list(range(0, int(min(times_grid.max(), CURVE_MAX_DAYS)) + 1, CURVE_STEP_DAYS))
    curve_points = [
        {
            "days_since_installation": d,
            "survival_probability": round(utils.eval_survival_at(times_grid, curve, d), 4),
        }
        for d in curve_days
    ]

    return {
        "item_id": data_reader.normalize(item_id),
        "installed_on": str(installed_on),
        "as_of": str(dataset_max_event_on),
        "age_days": round(age_days, 1),
        **risk,
        "risk_beyond_training_followup": beyond_training_followup,
        "median_survival_days_from_install": (
            round(median_days_from_install, 1) if median_days_from_install is not None else None
        ),
        "estimated_survival_curve": curve_points,
        "model_name": model_name,
        "note": (
            "Fitur dihitung pada KONDISI SAAT INSTALASI (installed_on), bukan kondisi PART hari "
            "ini. Risiko di atas hanya memperhitungkan berlalunya waktu sejak instalasi. C-index "
            "model ini BUKAN akurasi tanggal kerusakan - lihat README.md."
            + (
                " PERINGATAN: umur PART ini melebihi follow-up terpanjang yang pernah dilihat "
                "training - risk_Nd dikosongkan (None) karena tidak ada dasar yang andal untuk "
                "mengekstrapolasi, bukan berarti PART ini aman."
                if beyond_training_followup
                else ""
            )
        ),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Pemakaian: python predict.py <item_id>")
        return 1
    try:
        result = predict(sys.argv[1])
    except ItemNotScorable as exc:
        print(f"[TIDAK BISA DISKOR] {exc}")
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
