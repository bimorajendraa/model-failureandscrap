"""Ukur baseline performa CatBoost SEBELUM restrukturisasi (plan restrukturisasi
survival_model, Fase 0.3). Angka ini jadi ambang G5/G6 di Fase A (gerbang
validasi) - model survival dibandingkan terhadap performa NYATA model yang
akan digantikan, bukan angka yang dikarang.

    python scripts/baseline_performance.py
"""

from __future__ import annotations

import gc
import statistics
import time
from pathlib import Path

import psutil

ROOT_DIR = Path(__file__).resolve().parent.parent


def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def main() -> int:
    proc = psutil.Process()
    print(f"RSS sebelum apa pun dimuat: {_rss_mb():.1f} MB")

    print("\n[1/4] Cold model load...")
    from partrisk import predict as failure_model

    t0 = time.time()
    model, calibrator, metadata = failure_model.load_model()
    cold_load_s = time.time() - t0
    rss_after_load = _rss_mb()
    print(f"      cold load: {cold_load_s:.3f} detik")
    print(f"      model_version: {metadata['model_version']}")
    print(f"      RSS setelah load model: {rss_after_load:.1f} MB")

    print("\n[2/4] Ukuran artifact model failure...")
    failure_dir = ROOT_DIR / "models" / "failure" / metadata["model_version"]
    total_bytes = sum(f.stat().st_size for f in failure_dir.glob("*") if f.is_file())
    for f in sorted(failure_dir.glob("*")):
        if f.is_file():
            print(f"      {f.name}: {f.stat().st_size / 1e6:.3f} MB")
    print(f"      TOTAL: {total_bytes / 1e6:.3f} MB")

    print("\n[3/4] Single predict() p50 (20 PART aktif)...")
    from partrisk import data_reader

    cycles = data_reader.get_cycles()
    active = cycles.loc[
        cycles["cycle_end_reason"].eq("RIGHT_CENSORED_AT_DATA_END")
        & cycles["is_initial_model_cohort"].fillna(False)
    ]
    sample_items = active["item_identifier_clean"].drop_duplicates().head(20).tolist()
    if len(sample_items) < 20:
        print(f"      PERINGATAN: hanya {len(sample_items)} PART aktif ditemukan, bukan 20")

    # Satu panggilan pemanasan (fleet snapshot pertama kali selalu lebih lambat).
    failure_model.predict(sample_items[0])

    latencies = []
    for item_id in sample_items:
        t0 = time.time()
        failure_model.predict(item_id)
        latencies.append(time.time() - t0)
    latencies.sort()
    p50 = statistics.median(latencies)
    p90 = latencies[int(len(latencies) * 0.9)] if len(latencies) > 1 else latencies[0]
    print(f"      p50={p50*1000:.1f} ms  p90={p90*1000:.1f} ms  min={min(latencies)*1000:.1f} ms  max={max(latencies)*1000:.1f} ms")

    print("\n[4/4] Batch penuh (seluruh PART aktif)...")
    gc.collect()
    rss_before_batch = _rss_mb()
    from partrisk.serving import batch_predictor

    t0 = time.time()
    batch = batch_predictor.score_active_parts(force_refresh=True)
    batch_s = time.time() - t0
    rss_after_batch = _rss_mb()
    print(f"      {len(batch.frame):,} PART, {batch_s:.1f} detik")
    print(f"      RSS sebelum batch: {rss_before_batch:.1f} MB  sesudah: {rss_after_batch:.1f} MB  (+{rss_after_batch-rss_before_batch:.1f} MB)")

    report = f"""# Baseline performa CatBoost (v2) - SEBELUM restrukturisasi

Diukur {time.strftime('%Y-%m-%d %H:%M:%S')}. Ambang gerbang G5/G6 (Fase A) dihitung dari angka ini.

| Metrik | Nilai |
|---|---|
| model_version | {metadata['model_version']} |
| Ukuran artifact model failure (semua file) | {total_bytes/1e6:.3f} MB |
| Cold model load | {cold_load_s:.3f} s |
| RSS setelah load model | {rss_after_load:.1f} MB |
| Single predict() p50 (20 PART) | {p50*1000:.1f} ms |
| Single predict() p90 (20 PART) | {p90*1000:.1f} ms |
| Batch penuh ({len(batch.frame):,} PART) | {batch_s:.1f} s |
| RSS naik setelah batch penuh | {rss_after_batch-rss_before_batch:.1f} MB |

## Ambang turunan untuk gerbang Fase A

- **G5 (ukuran artifact)**: target keras <=100 MB (baseline CatBoost {total_bytes/1e6:.3f} MB - target ini BUKAN "boleh sebesar CatBoost x N", tapi batas keras production terlepas dari baseline, sesuai plan).
- **G6 (latency)**: cold load <=5s; single predict p50 <= {p50*1.5*1000:.1f} ms (1.5x baseline); batch penuh <= {batch_s*2:.1f}s (2x baseline).
"""
    out_path = ROOT_DIR / "reports"
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "baseline_performance_catboost.md").write_text(report, encoding="utf-8")
    print(f"\n[OK] Laporan: {out_path / 'baseline_performance_catboost.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
