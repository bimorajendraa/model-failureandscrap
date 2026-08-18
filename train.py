"""Latih (atau latih ulang) model risiko kerusakan PART 30 hari.

    python train.py

Alurnya:

    database -> observasi + target -> fitur -> latih -> evaluasi -> simpan

Setiap kali dijalankan, hasilnya disimpan sebagai versi BARU di models/vN/.
Model production hanya diganti kalau versi baru tidak lebih buruk pada data
uji; kalau lebih buruk, versinya tetap tersimpan lengkap dengan metriknya
supaya bisa dibandingkan, tetapi production tetap memakai model lama.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import config
import data_reader
import feature_builder

TRAIN, VALIDATION, TEST = "TRAIN", "VALIDATION", "TEST"
CURRENT_POINTER = config.FAILURE_MODEL_DIR / "CURRENT"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def assign_split(observations: pd.DataFrame, data_end: pd.Timestamp) -> pd.Series:
    """Bagi data berdasarkan WAKTU, bukan acak - model harus diuji pada periode
    yang belum pernah dilihatnya.

    Tahun terakhir yang ada di data jadi data uji, setahun sebelumnya jadi
    validasi, sisanya data latih. Di antara blok ada jeda (embargo) selebar
    horizon target: snapshot yang jawabannya baru terungkap di periode
    berikutnya dibuang, supaya jawaban periode uji tidak bocor ke data latih.
    """
    observed = pd.to_datetime(observations["observation_on"])
    resolved = observed + np.timedelta64(config.TARGET_HORIZON_DAYS, "D")

    test_start = pd.Timestamp(year=data_end.year, month=1, day=1)
    validation_start = test_start - pd.DateOffset(years=1)

    split = pd.Series("EXCLUDED_EMBARGO", index=observations.index)
    split[observed < pd.Timestamp(config.MIN_OBSERVATION_DATE)] = "EXCLUDED_TOO_OLD"
    split[
        (observed >= pd.Timestamp(config.MIN_OBSERVATION_DATE))
        & (resolved < validation_start)
    ] = TRAIN
    split[(observed >= validation_start) & (resolved < test_start)] = VALIDATION
    split[observed >= test_start] = TEST
    return split


def build_dataset() -> tuple:
    """Baca database lalu susun observasi, target, dan fitur."""
    print("[1/5] Membaca event dan siklus pemasangan dari database...")
    events = data_reader.get_events()
    cycles = data_reader.get_cycles()
    data_end = pd.Timestamp(cycles["dataset_max_event_on"].max())
    print(f"      {len(events):,} event, {len(cycles):,} siklus, data s/d {data_end}")

    print("[2/5] Menyusun observasi 30-harian dan target...")
    observations = feature_builder.training_observations(cycles)
    observations = feature_builder.attach_history(observations, events)
    # Kondisi armada dihitung point-in-time untuk tiap observasi.
    episodes = data_reader.get_failure_episodes()
    observations = feature_builder.attach_fleet(observations, cycles, episodes)

    # Dukungan historis dihitung dari SELURUH observasi, sebelum penyaringan
    # kelayakan, supaya nilainya benar-benar point-in-time.
    support = feature_builder.cumulative_support(observations)
    support_totals = feature_builder.support_totals(observations)

    eligible = observations["is_eligible"].to_numpy()
    dataset = observations.loc[eligible].reset_index(drop=True)
    dataset["split"] = assign_split(dataset, data_end)
    print(
        f"      {len(observations):,} observasi -> {len(dataset):,} layak dilatih "
        f"({int(dataset['target_failure'].sum()):,} kerusakan)"
    )

    print("[3/5] Menghitung fitur...")
    features = feature_builder.build_features(
        dataset, support.loc[eligible].reset_index(drop=True)
    )
    return dataset, features, support_totals, data_end, events, cycles, episodes


# ---------------------------------------------------------------------------
# Training dan evaluasi
# ---------------------------------------------------------------------------


def evaluate(target: pd.Series, raw: np.ndarray, calibrated: np.ndarray | None = None) -> dict:
    metrics = {
        "rows": int(len(target)),
        "positives": int(target.sum()),
        "roc_auc": float(roc_auc_score(target, raw)),
        "pr_auc": float(average_precision_score(target, raw)),
    }
    if calibrated is not None:
        metrics["brier_raw"] = float(brier_score_loss(target, raw))
        metrics["brier_calibrated"] = float(brier_score_loss(target, calibrated))
    return metrics


def train_model(dataset: pd.DataFrame, features: pd.DataFrame) -> tuple:
    parts = {name: dataset["split"].eq(name).to_numpy() for name in (TRAIN, VALIDATION, TEST)}
    for name, mask in parts.items():
        if not mask.any():
            raise SystemExit(f"Tidak ada baris untuk bagian {name}. Data belum cukup.")

    target = dataset["target_failure"].astype(bool)
    train_x, train_y = features[parts[TRAIN]], target[parts[TRAIN]]
    val_x, val_y = features[parts[VALIDATION]], target[parts[VALIDATION]]
    test_x, test_y = features[parts[TEST]], target[parts[TEST]]

    print(
        f"[4/5] Melatih model: latih={len(train_x):,} validasi={len(val_x):,} uji={len(test_x):,}"
    )
    if test_y.sum() < 30:
        print(
            f"      PERINGATAN: hanya {int(test_y.sum())} kerusakan di data uji - "
            "metrik uji akan sangat berisik. Pertimbangkan menunggu data lebih banyak."
        )

    model = CatBoostClassifier(
        random_seed=config.RANDOM_STATE, **config.CATBOOST_PARAMS
    )
    model.fit(
        Pool(train_x, train_y, cat_features=config.CATEGORICAL_FEATURES),
        eval_set=Pool(val_x, val_y, cat_features=config.CATEGORICAL_FEATURES),
    )

    raw_train = model.predict_proba(train_x)[:, 1]
    raw_val = model.predict_proba(val_x)[:, 1]
    raw_test = model.predict_proba(test_x)[:, 1]

    # Skor mentah CatBoost bagus untuk MENGURUTKAN risiko, tapi tidak bisa
    # dibaca sebagai probabilitas. Kalibrator memetakannya ke persentase yang
    # benar-benar mencerminkan frekuensi kerusakan sungguhan.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val, val_y.astype(int))

    metrics = {
        "train": evaluate(train_y, raw_train),
        "validation": evaluate(val_y, raw_val),
        "test": evaluate(test_y, raw_test, calibrator.predict(raw_test)),
    }
    return model, calibrator, metrics


def active_part_scores(
    model, cycles: pd.DataFrame, events: pd.DataFrame, support_totals: dict[str, int],
    episodes: pd.DataFrame, fleet: pd.DataFrame,
) -> np.ndarray:
    """Skor MENTAH seluruh PART yang saat ini masih terpasang.

    Dua alasan memakai skor mentah, bukan probabilitas terkalibrasi:

    1. Populasinya harus yang sebenarnya dihadapi production - seluruh PART
       aktif - bukan grid observasi data latih yang sudah tersaring aturan
       kelayakan label dan jumlahnya jauh lebih sedikit.
    2. Kalibrator menghasilkan dataran: 16.877 PART hanya menempati sekitar
       30 nilai probabilitas berbeda, sehingga jumlah PART yang tertandai
       melompat dari 97 langsung ke 303 tanpa nilai di antaranya. Skor mentah
       punya ribuan nilai berbeda dengan URUTAN YANG SAMA PERSIS, jadi batas
       kelompok bisa ditaruh tepat sesuai kapasitas.
    """
    snapshot = feature_builder.current_observations(cycles)
    snapshot = feature_builder.attach_history(snapshot, events)
    snapshot = feature_builder.attach_fleet_snapshot(snapshot, fleet)
    support = feature_builder.part_model_support(snapshot, support_totals)
    features = feature_builder.build_features(snapshot, support)
    return model.predict_proba(features)[:, 1]


def choose_cutoffs(active_score: np.ndarray) -> tuple[dict, dict]:
    """Ambang kelompok risiko diturunkan dari KAPASITAS KERJA bisnis.

    Seluruh PART aktif diurutkan menurut risiko, lalu sebanyak kapasitas per
    bulan itulah yang masuk kelompok HIGH. Karena tiap PART dinilai ulang
    setiap 30 hari, jumlah PART di daftar HIGH pada satu saat sama dengan
    beban kerja per bulan.

    Tidak ada label yang dipakai di sini - hanya urutan skor - jadi tidak ada
    kebocoran dari data uji.
    """
    high = _cutoff_for_count(active_score, config.FAILURE_CAPACITY_PER_MONTH)
    medium = _cutoff_for_count(
        active_score,
        int(config.FAILURE_MEDIUM_CAPACITY_MULTIPLIER * config.FAILURE_CAPACITY_PER_MONTH),
    )
    cutoffs = {"high": high, "medium": medium}
    basis = {
        "rule": "kapasitas kerja per bulan yang ditetapkan bisnis",
        "scale": "skor mentah model, bukan probabilitas terkalibrasi",
        "capacity_per_month": config.FAILURE_CAPACITY_PER_MONTH,
        "active_parts_scored": int(len(active_score)),
        # Jumlah yang benar-benar tercapai bisa meleset dari kapasitas karena
        # kalibrator menghasilkan banyak skor kembar - lihat _cutoff_for_count.
        "flagged_high": int((active_score >= high).sum()),
        "flagged_medium_band": int(((active_score >= medium) & (active_score < high)).sum()),
    }
    return cutoffs, basis


def _cutoff_for_count(score: np.ndarray, wanted: int) -> float:
    """Ambang yang jumlah PART tertandainya paling dekat dengan `wanted`.

    Tidak memakai kuantil biasa karena kalibrator menghasilkan dataran skor:
    banyak PART punya nilai persis sama, sehingga menggeser ambang sedikit
    saja bisa menarik ratusan PART sekaligus. Mencari di antara nilai skor
    yang benar-benar ada membuat jumlahnya sedekat mungkin dengan kapasitas.
    """
    candidates = np.unique(score)
    counts = np.array([(score >= value).sum() for value in candidates])
    return float(candidates[int(np.argmin(np.abs(counts - wanted)))])


def next_version() -> str:
    existing = [
        int(path.name[1:])
        for path in config.FAILURE_MODEL_DIR.glob("v*")
        if path.is_dir() and path.name[1:].isdigit()
    ]
    return f"v{max(existing, default=0) + 1}"


def current_version() -> str | None:
    if not CURRENT_POINTER.exists():
        return None
    version = CURRENT_POINTER.read_text(encoding="utf-8").strip()
    return version if (config.FAILURE_MODEL_DIR / version / "metadata.json").exists() else None


def load_metadata(version: str) -> dict:
    path = config.FAILURE_MODEL_DIR / version / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def save_version(
    version: str,
    model: CatBoostClassifier,
    calibrator: IsotonicRegression,
    metrics: dict,
    support_totals: dict[str, int],
    dataset: pd.DataFrame,
    data_end: pd.Timestamp,
    cutoffs: dict,
    cutoff_basis: dict,
    fleet: pd.DataFrame,
) -> dict:
    directory = config.FAILURE_MODEL_DIR / version
    directory.mkdir(parents=True, exist_ok=True)

    model.save_model(str(directory / "model.cbm"))
    joblib.dump(calibrator, directory / "calibrator.joblib")
    # Potret armada ikut disimpan supaya prediksi tidak perlu membangunnya
    # ulang dari nol. Sah dipakai selama data belum bertambah - predict.py
    # memeriksanya lewat dataset_max_event_on.
    #
    # CSV, bukan JSON: kode model seperti "0120201" akan dibaca ulang sebagai
    # angka 120201 oleh pembaca JSON, nol di depannya hilang, dan seluruh
    # pencocokan gagal diam-diam.
    fleet.to_csv(directory / "fleet_snapshot.csv", index=False)

    observed = pd.to_datetime(dataset["observation_on"])
    validation = metrics["validation"]
    metadata = {
        "model_version": version,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "fleet_snapshot_at": str(data_end),
        "training_period": {
            "observation_from": str(observed.min()),
            "observation_to": str(observed.max()),
            "dataset_max_event_on": str(data_end),
            "rows_by_split": dataset["split"].value_counts().to_dict(),
        },
        "target": (
            f"PART mengalami kerusakan dalam {config.TARGET_HORIZON_DAYS} hari "
            "setelah tanggal observasi"
        ),
        "features": config.FEATURE_COLUMNS,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "hyperparameters": {**config.CATBOOST_PARAMS, "random_seed": config.RANDOM_STATE},
        "evaluation_metrics": metrics,
        "validation_base_rate": validation["positives"] / validation["rows"],
        # Ambang kelompok risiko bukan hasil optimasi statistik, melainkan
        # diturunkan dari kapasitas kerja yang ditetapkan bisnis (config.py).
        "risk_cutoffs": cutoffs,
        "cutoff_basis": cutoff_basis,
        # Dibekukan supaya kategori tipe PART saat prediksi persis sama dengan
        # yang dipelajari model. Ikut diperbarui setiap kali training ulang.
        "part_model_support": support_totals,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def decide_promotion(new_score: float, force: bool) -> tuple[bool, str]:
    """Boleh tidaknya model baru menggantikan model production.

    Model baru yang lebih buruk pada data uji TIDAK otomatis dipakai - hasil
    latihnya tetap disimpan supaya bisa dibandingkan, tetapi production tidak
    ikut turun kualitas hanya karena training ulang sudah dijalankan.
    """
    previous = current_version()
    if previous is None:
        return True, "belum ada model production sebelumnya"

    old_score = load_metadata(previous)["evaluation_metrics"]["test"]["roc_auc"]
    comparison = f"ROC-AUC uji {new_score:.4f} vs {previous} {old_score:.4f}"
    if new_score >= old_score:
        return True, comparison
    if force:
        return True, f"{comparison} - dipaksa lewat --force-promote"
    return False, comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Pakai model baru sebagai production walaupun hasil ujinya lebih buruk.",
    )
    args = parser.parse_args()

    config.FAILURE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset, features, support_totals, data_end, events, cycles, episodes = build_dataset()
    model, calibrator, metrics = train_model(dataset, features)
    fleet = feature_builder.fleet_snapshot(cycles, episodes, data_end)
    cutoffs, cutoff_basis = choose_cutoffs(
        active_part_scores(model, cycles, events, support_totals, episodes, fleet))

    version = next_version()
    save_version(version, model, calibrator, metrics, support_totals, dataset,
                 data_end, cutoffs, cutoff_basis, fleet)

    print(f"[5/5] Tersimpan sebagai {version} di {config.FAILURE_MODEL_DIR / version}")
    for name in ("train", "validation", "test"):
        part = metrics[name]
        print(
            f"      {name:10s} baris={part['rows']:>7,} kerusakan={part['positives']:>5,} "
            f"ROC-AUC={part['roc_auc']:.4f} PR-AUC={part['pr_auc']:.4f}"
        )
    print(f"      Brier terkalibrasi (uji) = {metrics['test']['brier_calibrated']:.4f}")

    previous = current_version()
    promote, reason = decide_promotion(metrics["test"]["roc_auc"], args.force_promote)

    if promote:
        CURRENT_POINTER.write_text(version, encoding="utf-8")
        print(f"\n[OK] {version} dipakai sebagai model production ({reason}).")
    else:
        print(
            f"\n[TAHAN] Model production TETAP {previous} - {reason}.\n"
            f"        {version} tetap tersimpan untuk dibandingkan. Untuk tetap "
            f"memakainya: python train.py --force-promote, atau tulis '{version}' "
            f"ke {CURRENT_POINTER}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
