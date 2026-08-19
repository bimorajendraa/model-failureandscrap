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
    return model, calibrator, metrics, raw_test


def capacity_metrics(raw: np.ndarray, target: np.ndarray, window_days: float) -> dict:
    """Precision/recall pada sejumlah PART teratas yang setara kapasitas kerja
    tim untuk rentang waktu window uji ini.

    Konsisten dengan cara ambang HIGH ditetapkan (choose_cutoffs): bukan
    precision/recall pada satu ambang probabilitas yang dikarang, melainkan
    pada JUMLAH yang sanggup ditindaklanjuti tim - diskalakan dari
    FAILURE_CAPACITY_PER_MONTH ke panjang window uji yang sesungguhnya.
    """
    months = max(window_days / 30.0, 1e-9)
    capacity = max(int(round(config.FAILURE_CAPACITY_PER_MONTH * months)), 1)
    capacity = min(capacity, len(raw))
    flagged = np.argsort(-raw)[:capacity]
    true_positive = int(target[flagged].sum())
    return {
        "capacity_evaluated": capacity,
        "precision_at_capacity": true_positive / capacity if capacity else 0.0,
        "recall_at_capacity": true_positive / max(int(target.sum()), 1),
    }


def full_metrics(raw: np.ndarray, calibrated: np.ndarray, target: np.ndarray, window_days: float) -> dict:
    """Satu set metrik yang dipakai SAMA PERSIS untuk kandidat maupun model
    lama (incumbent), supaya keduanya benar-benar dibandingkan dengan formula
    yang identik - bukan cuma window waktu yang sama.

    PR-AUC dan Recall@kapasitas dipakai sebagai dasar promosi (lihat
    decide_promotion); ROC-AUC dan Brier ikut dihitung untuk konteks tetapi
    bukan penentu tunggal - data ini timpang (base rate kerusakan kecil), dan
    ROC-AUC bisa terlihat bagus walau presisi pada kapasitas kerja nyata
    memburuk.
    """
    metrics = {
        "rows": int(len(target)),
        "positives": int(target.sum()),
        "roc_auc": float(roc_auc_score(target, raw)),
        "pr_auc": float(average_precision_score(target, raw)),
        "brier_calibrated": float(brier_score_loss(target, calibrated)),
    }
    metrics.update(capacity_metrics(raw, target, window_days))
    return metrics


def evaluate_incumbent(previous_version: str, dataset: pd.DataFrame) -> dict:
    """Jalankan model CURRENT (bukan kandidat) pada test split yang PERSIS
    SAMA seperti kandidat, supaya keduanya dibandingkan pada window evaluasi
    yang identik.

    Sebelumnya promosi membandingkan skor kandidat dengan metrik LAMA yang
    tersimpan di metadata model production - dihitung pada test split model
    itu SENDIRI saat ia dilatih. Karena test_start dihitung ulang dari tahun
    data_end setiap kali retrain (lihat assign_split), window itu bergeser
    maju setiap tahun - kandidat dan incumbent akhirnya dibandingkan pada dua
    periode yang berbeda. Fungsi ini menutup celah itu: incumbent dijalankan
    ulang pada data BARU, dibatasi ke baris test split yang sama dengan
    kandidat.
    """
    metadata = load_metadata(previous_version)
    directory = config.FAILURE_MODEL_DIR / previous_version
    model = CatBoostClassifier()
    model.load_model(str(directory / "model.cbm"))
    calibrator = joblib.load(directory / "calibrator.joblib")

    test_dataset = dataset.loc[dataset["split"].eq(TEST)]
    # Kategori tipe PART pakai dukungan BEKU milik model lama - persis yang
    # dipakai predict.py saat model ini jadi production - bukan dukungan baru
    # yang dihitung untuk kandidat. Kalau dipakai dukungan baru, kategori yang
    # "dikenal" incumbent bisa berubah dan hasilnya tidak lagi mencerminkan
    # perilaku production sesungguhnya.
    incumbent_support = feature_builder.part_model_support(
        test_dataset, metadata["part_model_support"]
    )
    incumbent_features = feature_builder.build_features(test_dataset, incumbent_support)

    raw = model.predict_proba(incumbent_features)[:, 1]
    calibrated = calibrator.predict(raw)
    target = test_dataset["target_failure"].astype(bool).to_numpy()
    return {
        "model_version": previous_version,
        "raw": raw,
        "calibrated": calibrated,
        "target": target,
    }


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
    promotion_comparison: dict,
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
        # Perbandingan lengkap yang dipakai keputusan promosi - PR-AUC,
        # Recall@kapasitas, ROC-AUC, Brier, keduanya dihitung pada test split
        # yang SAMA PERSIS. Disimpan untuk audit; lihat decide_promotion().
        "promotion_comparison": promotion_comparison,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def decide_promotion(
    candidate: dict, incumbent: dict | None, previous_version: str | None, force: bool
) -> tuple[bool, str, dict]:
    """Boleh tidaknya model baru menggantikan model production.

    Dua syarat harus TIDAK memburuk, bukan satu skor tunggal:

    - PR-AUC: metrik urutan utama untuk data timpang ini (base rate kerusakan
      kecil) - ROC-AUC sendirian bisa terlihat bagus walau presisi pada
      kapasitas kerja nyata memburuk.
    - Recall@kapasitas: yang benar-benar dirasakan tim - dari kerusakan yang
      sungguh terjadi, berapa persen tertangkap pada jumlah PART yang sanggup
      diperiksa tim per bulan.

    ROC-AUC dan Brier tetap dihitung dan disimpan untuk konteks/audit, tetapi
    bukan penentu tunggal. Keduanya (kandidat dan incumbent) dihitung dengan
    `full_metrics()` yang sama persis pada test split yang sama persis - lihat
    `evaluate_incumbent()`.

    Model baru yang lebih buruk TIDAK otomatis dipakai - hasil latihnya tetap
    disimpan supaya bisa dibandingkan, tetapi production tidak ikut turun
    kualitas hanya karena training ulang sudah dijalankan.
    """
    if incumbent is None:
        return True, "belum ada model production sebelumnya", {"candidate": candidate}

    comparison = {
        "candidate": candidate,
        "incumbent": incumbent,
        "incumbent_version": previous_version,
    }
    pr_ok = candidate["pr_auc"] >= incumbent["pr_auc"]
    recall_ok = candidate["recall_at_capacity"] >= incumbent["recall_at_capacity"]
    reason = (
        f"PR-AUC {candidate['pr_auc']:.4f} vs {previous_version} {incumbent['pr_auc']:.4f} | "
        f"Recall@kapasitas {candidate['recall_at_capacity']:.4f} vs "
        f"{incumbent['recall_at_capacity']:.4f} | "
        f"ROC-AUC {candidate['roc_auc']:.4f} vs {incumbent['roc_auc']:.4f} | "
        f"Brier {candidate['brier_calibrated']:.4f} vs {incumbent['brier_calibrated']:.4f}"
    )
    if pr_ok and recall_ok:
        return True, reason, comparison
    if force:
        return True, f"{reason} - dipaksa lewat --force-promote", comparison
    return False, reason, comparison


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
    model, calibrator, metrics, raw_test = train_model(dataset, features)
    fleet = feature_builder.fleet_snapshot(cycles, episodes, data_end)
    cutoffs, cutoff_basis = choose_cutoffs(
        active_part_scores(model, cycles, events, support_totals, episodes, fleet))

    print("[5/5] Menyimpan dan mengevaluasi promosi...")
    for name in ("train", "validation", "test"):
        part = metrics[name]
        print(
            f"      {name:10s} baris={part['rows']:>7,} kerusakan={part['positives']:>5,} "
            f"ROC-AUC={part['roc_auc']:.4f} PR-AUC={part['pr_auc']:.4f}"
        )
    print(f"      Brier terkalibrasi (uji) = {metrics['test']['brier_calibrated']:.4f}")

    # Window uji kandidat - dipakai menskalakan kapasitas kerja ke Recall/
    # Precision@kapasitas, dan HARUS window yang sama dipakai mengevaluasi
    # incumbent supaya perbandingannya adil.
    test_dataset = dataset.loc[dataset["split"].eq(TEST)]
    test_observed = pd.to_datetime(test_dataset["observation_on"])
    window_days = (
        float((test_observed.max() - test_observed.min()).days) if len(test_dataset) else 0.0
    )
    # PENTING: metrik promosi TIDAK memakai raw_test (yang dibangun dengan
    # dukungan point-in-time - lihat build_dataset()). raw_test cocok untuk
    # metrics["test"] yang dilaporkan di atas (perilaku lama, dipertahankan
    # apa adanya), tetapi untuk membandingkan adil dengan incumbent (yang
    # dievaluasi dengan dukungan BEKU miliknya sendiri, persis seperti
    # predict.py melayani production), kandidat juga harus dievaluasi dengan
    # dukungan beku miliknya SENDIRI (support_totals) - bukan dukungan
    # point-in-time. Tanpa ini, kandidat tampak sedikit lebih baik semata-mata
    # karena metodologi fitur yang berbeda, bukan model yang sungguh berbeda.
    candidate_support = feature_builder.part_model_support(test_dataset, support_totals)
    candidate_features = feature_builder.build_features(test_dataset, candidate_support)
    candidate_raw = model.predict_proba(candidate_features)[:, 1]
    candidate_calibrated = calibrator.predict(candidate_raw)
    candidate_metrics = full_metrics(
        candidate_raw, candidate_calibrated,
        test_dataset["target_failure"].astype(bool).to_numpy(), window_days,
    )

    previous = current_version()
    incumbent_metrics = None
    if previous is not None:
        incumbent = evaluate_incumbent(previous, dataset)
        incumbent_metrics = full_metrics(
            incumbent["raw"], incumbent["calibrated"], incumbent["target"], window_days
        )

    promote, reason, comparison = decide_promotion(
        candidate_metrics, incumbent_metrics, previous, args.force_promote
    )

    version = next_version()
    save_version(version, model, calibrator, metrics, support_totals, dataset,
                 data_end, cutoffs, cutoff_basis, fleet, comparison)
    print(f"      Tersimpan sebagai {version} di {config.FAILURE_MODEL_DIR / version}")

    if incumbent_metrics is not None:
        print(
            f"\n      Perbandingan pada window uji yang sama ({window_days:.0f} hari, "
            f"kapasitas setara {candidate_metrics['capacity_evaluated']} PART):"
        )
        print(
            f"      {'':10s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'Recall@cap':>11s} "
            f"{'Precision@cap':>14s} {'Brier':>8s}"
        )
        for label, values in (("kandidat", candidate_metrics), (previous, incumbent_metrics)):
            print(
                f"      {label:10s} {values['pr_auc']:>8.4f} {values['roc_auc']:>8.4f} "
                f"{values['recall_at_capacity']:>11.4f} {values['precision_at_capacity']:>14.4f} "
                f"{values['brier_calibrated']:>8.4f}"
            )

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
