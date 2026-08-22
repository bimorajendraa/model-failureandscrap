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

Mode ADITIF (lihat gate_decision.md) - model ini TIDAK menggantikan CatBoost
sebagai mesin keputusan, cuma sumber field advisory
(`median_days_to_failure` di batch, plus kurva penuh di endpoint satu PART -
lihat `score_batch()`/`predict()`). ARTIFACTS_DIR menunjuk artifact hasil
`training.failure_survival` (kandidat compact A2, ~66 MB) di
survival_model/event_based/artifacts/ - BUKAN models/failure/v3/, karena
tidak ada mekanisme cutover/rollback yang perlu dibangun untuk field
advisory saja.
"""

from __future__ import annotations

import json
import math
import sys

import joblib
import numpy as np
import pandas as pd

from partrisk import config
from partrisk.data import reader as data_reader
from partrisk.features import failure as feature_builder
from partrisk.features.survival import builder as features
from partrisk.features.survival import install_context, previous_cycle
from partrisk.predict import failure as predict_failure
from partrisk.survival import curves
from partrisk.training import landmark_eval

ARTIFACTS_DIR = config.PACKAGE_DIR / "survival_model" / "event_based" / "artifacts"
HORIZONS_DAYS = [30, 60, 90, 120]
CURVE_STEP_DAYS = 30
CURVE_MAX_DAYS = 1080
BATCH_CHUNK_SIZE = 2000


class ItemNotScorable(Exception):
    pass


def load_model() -> tuple:
    """Model utama (RSF) + encoder + metadata mentah + kalibrator per horizon
    - dipakai `score_batch()` (serving/batch_predictor.py) dan
    `_load_primary_model()` di bawah (CLI satu PART). Melempar
    FileNotFoundError (BUKAN SystemExit) supaya pemanggil library
    (batch_predictor) bisa menangkapnya dan melewati field advisory dengan
    aman kalau model belum pernah dilatih - CLI satu PART (`main()`) yang
    menampilkan pesan SystemExit ke pengguna.

    `calibrators` (dict[int, IsotonicRegression] per horizon 30/60/90/120,
    dilatih training.failure_survival.fit_calibrators() di populasi
    VALIDATION) SELALU dimuat kalau filenya ada - artifact yang sudah
    dilatih sejak Fase A3 selalu punya file ini (lihat
    gate_a3_calibration_study.md), jadi None di sini menandakan artifact
    versi sangat lama, bukan kondisi normal."""
    if not (ARTIFACTS_DIR / "models.joblib").exists():
        raise FileNotFoundError(f"Artifacts belum ada di {ARTIFACTS_DIR}")
    models = joblib.load(ARTIFACTS_DIR / "models.joblib")
    metadata = json.loads((ARTIFACTS_DIR / "metadata.json").read_text(encoding="utf-8"))
    model = models[metadata["primary_model"]]
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1  # lihat catatan scripts/evaluate_survival.py load_artifacts()
    encoder = joblib.load(ARTIFACTS_DIR / "encoder.joblib")
    calibrators_path = ARTIFACTS_DIR / "calibrators.joblib"
    calibrators = joblib.load(calibrators_path) if calibrators_path.exists() else None
    return model, encoder, metadata, calibrators


def _load_primary_model():
    try:
        model, encoder, metadata, calibrators = load_model()
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Model event-based belum dilatih. Jalankan dulu: "
            f"python -m partrisk.training.failure_survival ({exc})"
        ) from exc
    support_totals = {k: int(v) for k, v in metadata["support_totals"].items()}
    item_type_support_totals = {k: int(v) for k, v in metadata["item_type_support_totals"].items()}
    terminal_support_totals = {k: int(v) for k, v in metadata["terminal_support_totals"].items()}
    return (
        model, metadata["primary_model"], encoder, support_totals,
        item_type_support_totals, terminal_support_totals, calibrators,
    )


def _calibrate_risk(raw_risk: dict[int, float], calibrators) -> dict[int, float | None]:
    """Kalibrasi tiap horizon SENDIRI-SENDIRI (isotonic terpisah per horizon,
    lihat fit_calibrators()), lalu cummax lintas horizon 30->120 -
    tanggung jawab PEMANGGIL kalibrator (didokumentasikan di
    fit_calibrators()) karena isotonic per horizon BISA saling silang walau
    S(t) mentahnya monoton turun (gate_a3_calibration_study.md). Kalau
    calibrators None (artifact lama) atau nilai raw None (beyond follow-up),
    kembalikan None apa adanya - tidak mengarang angka."""
    if calibrators is None:
        return {h: None for h in raw_risk}
    ordered_horizons = sorted(raw_risk)
    calibrated: dict[int, float | None] = {}
    running_max = 0.0
    for h in ordered_horizons:
        raw = raw_risk[h]
        if raw is None or h not in calibrators:
            calibrated[h] = None
            continue
        value = float(calibrators[h].predict([raw])[0])
        running_max = max(running_max, value)
        calibrated[h] = round(running_max, 4)
    return calibrated


def predict(item_id: str) -> dict:
    (
        model, model_name, encoder, support_totals,
        item_type_support_totals, terminal_support_totals, calibrators,
    ) = _load_primary_model()

    dataset_max_event_on = data_reader.get_dataset_max_event_on()
    # Positional (bukan keyword) SENGAJA - query_cache.py mencocokkan cache
    # key persis dari (args, kwargs), predict/failure.py dan predict/scrap.py
    # sudah memanggil get_cycles/get_events positional untuk PART yang sama;
    # kalau di sini pakai keyword, kuncinya beda dan cache tidak nyambung -
    # satu assessment diam-diam kembali baca database berulang kali.
    cycles_for_item = data_reader.get_cycles(item_id, dataset_max_event_on)
    active = cycles_for_item.loc[cycles_for_item["cycle_end_reason"].eq("RIGHT_CENSORED_AT_DATA_END")]
    if active.empty:
        raise ItemNotScorable(f"{item_id}: tidak ada siklus yang sedang aktif (PART tidak terpasang sekarang).")
    if not bool(active.iloc[0]["is_initial_model_cohort"]):
        raise ItemNotScorable(f"{item_id}: identitas tipe PART tidak bisa dipastikan dari inventory - tidak diskor.")

    active_cycle = active.iloc[[0]].reset_index(drop=True)
    installed_on = pd.Timestamp(active_cycle.loc[0, "installed_on"])
    age_days = float((dataset_max_event_on - installed_on).total_seconds() / 86400.0)

    events = data_reader.get_events(item_id)

    # --- Landmark TUNGGAL = SEKARANG (observation_on=dataset_max_event_on) -
    # SAMA persis mekanismenya dengan satu baris landmark saat training
    # (features/survival/landmarks.py), hanya satu titik waktu ("sekarang")
    # bukan banyak.
    observations = active_cycle.copy()
    observations["observation_on"] = dataset_max_event_on
    observations["days_since_installation"] = age_days
    observations["landmark_age_days"] = age_days  # attach_dynamic_extra() butuh nama kolom ini (lihat landmarks.py)
    observations = install_context.attach_install_context(observations, events)
    terminal_raw = data_reader.get_terminal_context(item_id)
    observations = features.attach_terminal_extra(observations, terminal_raw)
    observations = feature_builder.attach_history(observations, events)
    # attach_fleet_snapshot (BUKAN attach_fleet + get_cycles() tanpa filter)
    # SENGAJA - landmark di sini SELALU "sekarang" (observation_on=
    # dataset_max_event_on), titik waktu yang PERSIS sama dengan snapshot
    # armada yang sudah di-cache predict/failure.py (dipakai CatBoost) -
    # attach_fleet_snapshot() dijamin identik dengan attach_fleet() point-in-time
    # untuk kasus ini (lihat docstring features/fleet.py attach_fleet_snapshot).
    # Dulu di sini memanggil get_cycles() TANPA filter (seluruh armada,
    # puluhan ribu baris) di SETIAP request satu-PART - di native itu cepat,
    # tapi lewat jaringan Docker (host.docker.internal) terukur >50 detik per
    # assessment. predict_failure.fleet_snapshot() sudah dicache per-proses
    # dan diinvalidasi otomatis lewat data_state (clear_fleet_cache()) -
    # pakai cache yang sama, jangan query fleet-wide kedua kalinya.
    observations = feature_builder.attach_fleet_snapshot(observations, predict_failure.fleet_snapshot(dataset_max_event_on))
    # attach_dynamic_extra dan audit_previous_cycle_features juga groupby
    # per item_identifier_clean - hasilnya identik dipakaikan cycles_for_item
    # (riwayat PART INI saja, sudah diambil di atas) daripada seluruh armada,
    # baris item lain tidak pernah mempengaruhi hasil baris item ini.
    observations = features.attach_dynamic_extra(observations, cycles_for_item, events)
    # compute_features() di bawah memanggil feature_builder.build_features()
    # sebagai utilitas bersama - butuh log_previous_cycle_count
    # (config.DEGRADATION_FEATURES). attach_dynamic_extra() di atas sudah
    # punya previous_cycle_count RAW (dipakai survival sendiri) - tinggal
    # di-log1p.
    observations["log_previous_cycle_count"] = np.log1p(observations["previous_cycle_count"].astype(float))
    # Sama, compute_features() butuh LOCAL_DENSITY_FEATURES ada. Landmark di
    # sini SELALU "sekarang" - pakai potret cache milik CatBoost
    # (predict_failure.item_type_density_snapshot), BUKAN dihitung ulang
    # fleet-wide (itu justru bug latency yang sudah diperbaiki di predict()
    # - lihat komentar attach_fleet_snapshot di atas).
    observations = feature_builder.attach_item_type_density_snapshot(
        observations, events, predict_failure.item_type_density_snapshot(dataset_max_event_on)
    )

    pc = previous_cycle.audit_previous_cycle_features(cycles_for_item)
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
        raw_risk = {h: None for h in HORIZONS_DAYS}
    else:
        raw_risk = {
            h: 1.0 - curves.eval_survival_at(times_grid, curve, float(h))
            for h in HORIZONS_DAYS
        }
    risk = {f"risk_{h}d": (round(v, 4) if v is not None else None) for h, v in raw_risk.items()}
    # Kalibrasi horizon (Fase R1 upgrade RSF - lihat _calibrate_risk()) -
    # isotonic per horizon dari fit_calibrators(), TERPISAH dari risk_Nd
    # mentah di atas supaya kejujuran "ini sudah dikalibrasi atau belum"
    # tetap eksplisit lewat nama field, bukan ditimpa diam-diam.
    calibrated = _calibrate_risk(raw_risk, calibrators)
    calibrated_risk = {f"calibrated_risk_{h}d": calibrated[h] for h in HORIZONS_DAYS}
    median_days_remaining = curves.median_survival_time(times_grid, curve)
    # median SERING None (kebanyakan PART aktif belum cukup lama untuk S(t)
    # turun sampai separuh dalam rentang follow-up training - diukur lewat
    # score_batch(): 5,3% dari populasi aktif) - ambang 90% jauh lebih sering
    # tercapai dan tetap actionable, lihat survival.curves docstring.
    days_until_90pct_remaining = curves.survival_time_at_threshold(times_grid, curve, 0.9)

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
        **calibrated_risk,
        "median_days_remaining_from_now": (
            round(median_days_remaining, 1) if median_days_remaining is not None else None
        ),
        "days_until_survival_90pct_from_now": (
            round(days_until_90pct_remaining, 1) if days_until_90pct_remaining is not None else None
        ),
        "estimated_survival_curve_from_now": curve_points,
        "model_name": model_name,
        "note": (
            "Fitur dihitung pada KONDISI SEKARANG (observation_on=as_of), TERMASUK riwayat/armada "
            "terbaru sampai hari ini - beda dari predict/failure.py (baseline instalasi) yang fiturnya "
            "beku di installed_on. risk_Nd = P(gagal dalam N hari ke depan | kondisi sekarang)."
        ),
    }


def score_batch(
    rows: pd.DataFrame, events: pd.DataFrame, cycles: pd.DataFrame, episodes: pd.DataFrame,
    terminal_raw: pd.DataFrame, model, encoder, metadata: dict,
) -> pd.DataFrame:
    """`median_days_to_failure` untuk BANYAK PART aktif sekaligus - field
    advisory dipakai `serving/batch_predictor.py` (mode aditif, TIDAK
    menentukan risk_level/urutan). `rows` = potret PART aktif
    (`feature_builder.current_observations(cycles)`, kolom sama dengan yang
    dipakai model classification) - fitur event-based dibangun PADA
    observation_on tiap baris lewat `training.landmark_eval` (mekanisme SAMA
    dengan `predict()` di atas, cuma divektorkan).

    `days_until_survival_90pct` (umur saat S(t) turun sampai 90%, bukan 50%)
    disertakan berdampingan - median sering None (kebanyakan PART aktif
    belum cukup lama untuk S(t) turun sampai separuh dalam rentang follow-up
    training), ambang 90% jauh lebih sering tercapai dan tetap actionable
    (lihat docstring `survival.curves.survival_time_at_threshold`).

    Kurva penuh SENGAJA tidak disertakan di sini (payload batch akan melipat
    puluhan kali untuk field yang jarang dibutuhkan di daftar prioritas -
    kurva penuh tetap eksklusif endpoint satu PART lewat `predict()`)."""
    feature_frame = landmark_eval.build_landmark_features_at_observation(
        rows, events, cycles, episodes, terminal_raw, metadata
    )
    n = len(feature_frame)
    median_days = np.full(n, np.nan)
    days_until_90pct = np.full(n, np.nan)
    n_chunks = math.ceil(n / BATCH_CHUNK_SIZE)
    for i in range(n_chunks):
        lo, hi = i * BATCH_CHUNK_SIZE, min((i + 1) * BATCH_CHUNK_SIZE, n)
        chunk = feature_frame.iloc[lo:hi]
        x_chunk = features.encode(chunk, encoder)
        times_grid, curve_values = curves.survival_curve_arrays(model, x_chunk)
        for k in range(curve_values.shape[0]):
            median = curves.median_survival_time(times_grid, curve_values[k])
            median_days[lo + k] = np.nan if median is None else median
            at_90pct = curves.survival_time_at_threshold(times_grid, curve_values[k], 0.9)
            days_until_90pct[lo + k] = np.nan if at_90pct is None else at_90pct
        del curve_values

    return pd.DataFrame({
        "item_id": rows["item_identifier_clean"].to_numpy(),
        "median_days_to_failure": median_days,
        "days_until_survival_90pct": days_until_90pct,
    })


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
