"""Prediksi risiko survival satu PART aktif - CLI sederhana, model event-based.

    python -m partrisk.predict.survival <item_id>

Beda MENDASAR dari model classification (predict/failure.py): fitur dihitung
pada `observation_on = SEKARANG` (bukan `installed_on`), lewat mekanisme
landmark yang SAMA dengan training (`feature_builder.attach_history`/
`attach_fleet` dengan observation_on=sekarang) - PART yang baru saja kena
corrective bulan lalu akan terlihat berbeda dari PART sejenis yang tidak
pernah bermasalah, walau install-nya sama-sama lama. Ini pertanyaan yang
TIDAK BISA dijawab model classification 30-hari.

Konsekuensi bagus di sisi matematika: kurva `S(t)` model ini SUDAH dari
`t=0=observation_on=sekarang`, jadi `risk_Nd = 1 - S(N)` LANGSUNG - tidak
perlu rumus `1-S(age+N)/S(age)` seperti model baseline-instalasi (yang
perlu itu KARENA kurvanya dari t=0=install).

BELUM dipakai jalur serving (mode aditif - lihat gate_decision.md, model ini
TIDAK menggantikan CatBoost sebagai mesin keputusan, cuma sumber field
advisory median_days_to_failure/survival_curve kalau/ketika itu diaktifkan).
ARTIFACTS_DIR SEMENTARA masih menunjuk artifact riset (survival_model/event_based/artifacts/,
5,26 GB, BUKAN kandidat compact A2) - pindah ke models/failure/v3/ resmi
begitu training.failure_survival + artifacts.families selesai dibangun.
"""

from __future__ import annotations

import json
import sys

import joblib
import pandas as pd

from partrisk import config
from partrisk.data import reader as data_reader
from partrisk.features import failure as feature_builder
from partrisk.features.survival import builder as features
from partrisk.features.survival import install_context, previous_cycle
from partrisk.survival import curves

ARTIFACTS_DIR = config.PACKAGE_DIR / "survival_model" / "event_based" / "artifacts"
HORIZONS_DAYS = [30, 60, 90, 120]
CURVE_STEP_DAYS = 30
CURVE_MAX_DAYS = 1080


class ItemNotScorable(Exception):
    pass


def _load_primary_model():
    if not (ARTIFACTS_DIR / "models.joblib").exists():
        raise SystemExit(
            f"Model event-based belum dilatih. Jalankan dulu: "
            f"python -m partrisk.training.datasets.survival (artifacts belum ada di {ARTIFACTS_DIR})"
        )
    models = joblib.load(ARTIFACTS_DIR / "models.joblib")
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    primary_name = metadata["primary_model"]
    model = models[primary_name]
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1  # lihat catatan scripts/evaluate_survival.py load_artifacts()
    encoder = joblib.load(ARTIFACTS_DIR / "encoder.joblib")
    support_totals = {k: int(v) for k, v in metadata["support_totals"].items()}
    item_type_support_totals = {k: int(v) for k, v in metadata["item_type_support_totals"].items()}
    terminal_support_totals = {k: int(v) for k, v in metadata["terminal_support_totals"].items()}
    return model, primary_name, encoder, support_totals, item_type_support_totals, terminal_support_totals


def predict(item_id: str) -> dict:
    model, model_name, encoder, support_totals, item_type_support_totals, terminal_support_totals = (
        _load_primary_model()
    )

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

    events = data_reader.get_events(item_id=item_id)
    all_cycles = data_reader.get_cycles()
    episodes = data_reader.get_failure_episodes()

    # --- Landmark TUNGGAL = SEKARANG (observation_on=dataset_max_event_on) -
    # SAMA persis mekanismenya dengan satu baris landmark saat training
    # (features/survival/landmarks.py), hanya satu titik waktu ("sekarang")
    # bukan banyak.
    observations = active_cycle.copy()
    observations["observation_on"] = dataset_max_event_on
    observations["days_since_installation"] = age_days
    observations["landmark_age_days"] = age_days  # attach_dynamic_extra() butuh nama kolom ini (lihat landmarks.py)
    observations = install_context.attach_install_context(observations, events)
    terminal_raw = data_reader.get_terminal_context(item_id=item_id)
    observations = features.attach_terminal_extra(observations, terminal_raw)
    observations = feature_builder.attach_history(observations, events)
    observations = feature_builder.attach_fleet(observations, all_cycles, episodes)
    observations = features.attach_dynamic_extra(observations, all_cycles, events)

    pc = previous_cycle.audit_previous_cycle_features(all_cycles)
    observations = observations.merge(
        pc[[
            "installation_cycle_id", "previous_cycle_confirmed_failure_lifetime_mean", "last_confirmed_failure_lifetime",
        ]],
        on="installation_cycle_id", how="left",
    )
    transform = previous_cycle.transform_for_model(observations)[
        ["log_previous_cycle_confirmed_failure_lifetime_mean", "has_previous_cycle_confirmed_failure_lifetime_mean"]
    ]
    observations = pd.concat([observations, transform], axis=1)

    # Dukungan historis DIBEKUKAN saat training - bukan dihitung ulang dari 1
    # baris ini sendiri (alasan sama dengan predict/failure.py).
    support = observations["item_model_code_clean"].map(support_totals).fillna(0).astype("int64")
    item_type_support = observations["item_type_at_install"].map(item_type_support_totals).fillna(0).astype("int64")
    terminal_support = observations["terminal_type_context"].map(terminal_support_totals).fillna(0).astype("int64")
    feature_frame = features.compute_features(observations, support, item_type_support, terminal_support)
    x = features.encode(feature_frame, encoder)

    times_grid, curve_values = curves.survival_curve_arrays(model, x)
    curve = curve_values[0]

    # t=0 kurva ini SUDAH "sekarang" (lihat docstring modul) - risk_Nd =
    # 1-S(N) LANGSUNG, tidak perlu dibagi S(age) seperti model baseline-instalasi.
    beyond_training_followup = bool(times_grid.max() <= 0)
    if beyond_training_followup:
        risk = {f"risk_{h}d": None for h in HORIZONS_DAYS}
    else:
        risk = {
            f"risk_{h}d": round(1.0 - curves.eval_survival_at(times_grid, curve, float(h)), 4)
            for h in HORIZONS_DAYS
        }
    median_days_remaining = curves.median_survival_time(times_grid, curve)

    curve_days = list(range(0, int(min(times_grid.max(), CURVE_MAX_DAYS)) + 1, CURVE_STEP_DAYS))
    curve_points = [
        {
            "days_from_now": d,
            "survival_probability": round(curves.eval_survival_at(times_grid, curve, d), 4),
        }
        for d in curve_days
    ]

    return {
        "item_id": data_reader.normalize(item_id),
        "installed_on": str(installed_on),
        "as_of": str(dataset_max_event_on),
        "age_days": round(age_days, 1),
        **risk,
        "median_days_remaining_from_now": (
            round(median_days_remaining, 1) if median_days_remaining is not None else None
        ),
        "estimated_survival_curve_from_now": curve_points,
        "model_name": model_name,
        "note": (
            "Fitur dihitung pada KONDISI SEKARANG (observation_on=as_of), TERMASUK riwayat/armada "
            "terbaru sampai hari ini - beda dari predict/failure.py (baseline instalasi) yang fiturnya "
            "beku di installed_on. risk_Nd = P(gagal dalam N hari ke depan | kondisi sekarang)."
        ),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("Pemakaian: python -m partrisk.predict.survival <item_id>")
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
