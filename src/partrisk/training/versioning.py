"""Helper generik dipakai training/failure_classification.py dan training/scrap.py.

Hanya bagian yang benar-benar identik di kedua file: penomoran versi model,
metrik evaluasi, dan keputusan promosi. Feature engineering, kandidat model,
dan cara mengevaluasi incumbent TETAP terpisah di masing-masing file - itu
memang berbeda antara model kerusakan dan model scrap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def next_version(model_dir: Path) -> str:
    existing = [
        int(path.name[1:])
        for path in model_dir.glob("v*")
        if path.is_dir() and path.name[1:].isdigit()
    ]
    return f"v{max(existing, default=0) + 1}"


def current_version(model_dir: Path) -> str | None:
    pointer = model_dir / "CURRENT"
    if not pointer.exists():
        return None
    version = pointer.read_text(encoding="utf-8").strip()
    return version if (model_dir / version / "metadata.json").exists() else None


def capacity_metrics(
    raw: np.ndarray,
    target: np.ndarray,
    window_days: float,
    capacity_per_month: float,
    days_per_month: float = 30.0,
) -> dict:
    """Precision/recall pada sejumlah baris teratas yang setara kapasitas
    kerja tim untuk panjang window uji ini - bukan pada satu ambang
    probabilitas yang dikarang.

    `days_per_month` beda sengaja antara train.py (30 - grid observasi tetap
    setiap 30 hari) dan train_scrap.py (30,44 - rata-rata kalender, karena
    kerusakan scrap tersebar bebas di waktu nyata, bukan grid tetap). Jangan
    disamakan - itu bukan bug.
    """
    months = max(window_days / days_per_month, 1e-9)
    capacity = max(int(round(capacity_per_month * months)), 1)
    capacity = min(capacity, len(raw))
    flagged = np.argsort(-raw)[:capacity]
    true_positive = int(target[flagged].sum())
    return {
        "capacity_evaluated": capacity,
        "precision_at_capacity": true_positive / capacity if capacity else 0.0,
        "recall_at_capacity": true_positive / max(int(target.sum()), 1),
    }


def full_metrics(
    raw: np.ndarray,
    calibrated: np.ndarray,
    target: np.ndarray,
    window_days: float,
    capacity_per_month: float,
    days_per_month: float = 30.0,
) -> dict:
    """Satu set metrik dipakai SAMA PERSIS untuk kandidat maupun incumbent,
    supaya keduanya dibandingkan dengan formula yang identik."""
    metrics = {
        "rows": int(len(target)),
        "positives": int(target.sum()),
        "roc_auc": float(roc_auc_score(target, raw)),
        "pr_auc": float(average_precision_score(target, raw)),
        "brier_calibrated": float(brier_score_loss(target, calibrated)),
    }
    metrics.update(
        capacity_metrics(raw, target, window_days, capacity_per_month, days_per_month)
    )
    return metrics


def decide_promotion(
    candidate: dict, incumbent: dict | None, previous_version: str | None, force: bool
) -> tuple[bool, str, dict]:
    """Dua syarat harus TIDAK memburuk - PR-AUC dan Recall@kapasitas - bukan
    satu skor tunggal. Data timpang (base rate kecil) membuat ROC-AUC sendirian
    bisa terlihat bagus walau presisi pada kapasitas kerja nyata memburuk.

    Model baru yang lebih buruk tidak otomatis dipakai. Hasil latihnya tetap
    disimpan untuk dibandingkan, tetapi production tidak ikut turun kualitas.
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


def print_promotion_comparison(
    candidate_metrics: dict, incumbent_metrics: dict | None, previous_version: str | None,
    window_days: float, unit_label: str,
) -> None:
    """Cetak tabel kandidat vs incumbent pada window uji yang sama - dipakai
    train.py dan train_scrap.py (dulu dua salinan identik, Fase B4 dedup).
    Tidak mencetak apa pun kalau belum ada incumbent (retrain pertama)."""
    if incumbent_metrics is None:
        return
    print(
        f"\n      Perbandingan pada window uji yang sama ({window_days:.0f} hari, "
        f"kapasitas setara {candidate_metrics['capacity_evaluated']} {unit_label}):"
    )
    print(
        f"      {'':10s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'Recall@cap':>11s} "
        f"{'Precision@cap':>14s} {'Brier':>8s}"
    )
    for label, values in (("kandidat", candidate_metrics), (previous_version, incumbent_metrics)):
        print(
            f"      {label:10s} {values['pr_auc']:>8.4f} {values['roc_auc']:>8.4f} "
            f"{values['recall_at_capacity']:>11.4f} {values['precision_at_capacity']:>14.4f} "
            f"{values['brier_calibrated']:>8.4f}"
        )
