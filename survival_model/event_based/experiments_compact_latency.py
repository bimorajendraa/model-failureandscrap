"""Fase A2 lanjutan: ukur latency RIIL kandidat compact (bukan cuma ukuran
artifact) untuk G6 - cold load, predict_survival_function() single-row dan
chunk, dibandingkan ambang dari `reports/baseline_performance_catboost.md`
(cold load <=5s, single predict p50 <=3467,7 ms, batch <=94,0s).

    python survival_model/event_based/experiments_compact_latency.py
"""

from __future__ import annotations

import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

SURVIVAL_DIR = Path(__file__).resolve().parent.parent
if str(SURVIVAL_DIR) not in sys.path:
    sys.path.insert(0, str(SURVIVAL_DIR))
EVENT_BASED_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVENT_BASED_DIR))

import joblib

import build_dataset
from src import model_fit
from src import utils as survival_utils

from eb_src import features
from experiments_compact_model import COMPACT_RSF_PARAMS, coarsen_duration_days

REPORTS_DIR = EVENT_BASED_DIR / "reports"


def main() -> int:
    print("[1/3] Menyusun dataset (cache) + melatih kandidat compact...")
    os.environ.setdefault("SURVIVAL_BUILD_CACHE", "1")
    built = build_dataset.build()
    dataset, feature_frame = built["dataset"], built["features"]
    train_mask = (dataset["split"] == "TRAIN").to_numpy()
    val_mask = (dataset["split"] == "VALIDATION").to_numpy()

    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    encoder = features.fit_encoder(feature_frame.loc[train_mask])
    x_train = features.encode(feature_frame.loc[train_mask], encoder)
    x_val = features.encode(feature_frame.loc[val_mask], encoder)
    y_train_true = model_fit.make_survival_target(dataset, train_mask)
    y_train_coarse = Surv.from_arrays(event=y_train_true["event"], time=coarsen_duration_days(y_train_true["time"]))
    model = RandomSurvivalForest(**COMPACT_RSF_PARAMS).fit(x_train, y_train_coarse)
    model.n_jobs = 1  # lihat catatan evaluate.py soal loky hang setelah unpickle

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model_compact.joblib"
        joblib.dump(model, path)
        size_mb = path.stat().st_size / 1e6

        print("[2/3] Cold load...")
        t0 = time.time()
        loaded = joblib.load(path)
        loaded.n_jobs = 1
        cold_load_s = time.time() - t0

    print("[3/3] Latency predict_survival_function() - single-row dan chunk 2.000...")
    single = x_val.iloc[:1]
    loaded.predict_survival_function(single, return_array=False)  # warmup
    latencies = []
    for i in range(20):
        row = x_val.iloc[[i % len(x_val)]]
        t0 = time.time()
        survival_utils.survival_curve_arrays(loaded, row)
        latencies.append(time.time() - t0)
    latencies.sort()
    p50 = statistics.median(latencies)
    p90 = latencies[int(len(latencies) * 0.9)]

    chunk = x_val.iloc[: min(2000, len(x_val))]
    t0 = time.time()
    survival_utils.survival_curve_arrays(loaded, chunk)
    chunk_s = time.time() - t0
    per_row_ms = (chunk_s / len(chunk)) * 1000
    extrapolated_16877_s = per_row_ms * 16877 / 1000

    print(f"      Ukuran artifact: {size_mb:.1f} MB")
    print(f"      Cold load: {cold_load_s:.3f} s")
    print(f"      Single predict p50={p50*1000:.1f} ms  p90={p90*1000:.1f} ms")
    print(f"      Chunk {len(chunk):,} baris: {chunk_s:.1f} s ({per_row_ms:.3f} ms/baris)")
    print(f"      Ekstrapolasi batch 16.877 PART (skala baseline CatBoost): {extrapolated_16877_s:.1f} s")

    report = f"""# Fase A2 lanjutan: latency kandidat compact (G6)

Model: kandidat compact A2 (n_estimators={COMPACT_RSF_PARAMS['n_estimators']},
min_samples_leaf={COMPACT_RSF_PARAMS['min_samples_leaf']}, grid dikasarkan).
Diukur pada `x_val` (fitur VALIDATION landmark, sudah di-encode) - proxy realistis
untuk bentuk data yang dilihat `predict.py` (jumlah kolom fitur sama).

| Metrik | Nilai | Ambang G6 (dari baseline_performance_catboost.md) | Status |
|---|---|---|---|
| Ukuran artifact | {size_mb:.1f} MB | <=100 MB (keras) | {"LULUS" if size_mb <= 100 else "GAGAL"} |
| Cold load | {cold_load_s:.3f} s | <=5 s | {"LULUS" if cold_load_s <= 5 else "GAGAL"} |
| Single predict p50 | {p50*1000:.1f} ms | <=3467,7 ms (1,5x baseline) | {"LULUS" if p50*1000 <= 3467.7 else "GAGAL"} |
| Batch (ekstrapolasi 16.877 PART) | {extrapolated_16877_s:.1f} s | <=94,0 s (2x baseline) | {"LULUS" if extrapolated_16877_s <= 94.0 else "GAGAL"} |

Batch di atas EKSTRAPOLASI linier dari chunk {len(chunk):,} baris ({per_row_ms:.3f} ms/baris) -
BUKAN pengukuran end-to-end lewat pipeline fitur production (itu perlu serving code
Fase C yang sesungguhnya, termasuk pembacaan database dan pembangunan fitur - di luar
cakupan studi kelayakan ini).
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "gate_a2_compact_latency.md").write_text(report, encoding="utf-8")
    print(f"\n[OK] Laporan: {REPORTS_DIR / 'gate_a2_compact_latency.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
