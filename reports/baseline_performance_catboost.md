# Baseline performa CatBoost (v2) - SEBELUM restrukturisasi

Diukur 2026-08-21 14:49:41. Ambang gerbang G5/G6 (Fase A) dihitung dari angka ini.

| Metrik | Nilai |
|---|---|
| model_version | v2 |
| Ukuran artifact model failure (semua file) | 0.157 MB |
| Cold model load | 0.862 s |
| RSS setelah load model | 171.5 MB |
| Single predict() p50 (20 PART) | 2311.8 ms |
| Single predict() p90 (20 PART) | 2453.1 ms |
| Batch penuh (16,877 PART) | 47.0 s |
| RSS naik setelah batch penuh | 98.2 MB |

## Ambang turunan untuk gerbang Fase A

- **G5 (ukuran artifact)**: target keras <=100 MB (baseline CatBoost 0.157 MB - target ini BUKAN "boleh sebesar CatBoost x N", tapi batas keras production terlepas dari baseline, sesuai plan).
- **G6 (latency)**: cold load <=5s; single predict p50 <= 3467.7 ms (1.5x baseline); batch penuh <= 94.0s (2x baseline).
