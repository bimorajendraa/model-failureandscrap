# Fase A5: Keputusan Gerbang - Survival Event-Based vs CatBoost v2

Tanggal: 2026-08-21. Ditandatangani berdasarkan seluruh angka A1-A4 di bawah.
Semua kriteria G1-G8 didefinisikan di plan restrukturisasi (`reports/` sesi ini).

## Ringkasan keputusan

**TIDAK cutover.** G1 lulus tapi **G2 dan G3 gagal secara nyata** (bukan
marginal) pada evaluasi production-realistic (A1) - sesuai aturan plan
*"G1 atau G2 gagal -> JANGAN cutover"*, model survival event-based **TIDAK**
menggantikan CatBoost v2 sebagai mesin keputusan utama.

**Restrukturisasi (`src/partrisk`) tetap jalan** (keputusan plan, tidak
bersyarat gerbang). Survival masuk mode **aditif**: CatBoost tetap memiliki
`failure_probability_*`, `risk_level`, `tier_score`; survival menyuplai
`median_days_to_failure` + `survival_curve` sebagai field advisory
(`advisory: true`) - kontrak API tetap diperluas, cuma tidak menggantikan
jalur keputusan.

## G1-G4: akurasi operasional (A1, populasi TEST classification identik, N=38.451)

Beda dari angka lama (`reports/ensemble_operational.md`, PR-AUC 0,1824): fitur
event-based di sini dihitung PADA `observation_on` tiap baris (kondisi PART
saat snapshot classification diambil), bukan dibekukan di `installed_on` -
cara yang benar-benar merepresentasikan `predict.py`. Lihat
`reports/gate_a1_landmark_operational.md`.

| # | Kriteria | Ambang | Event-based | CatBoost v2 | Status |
|---|---|---|---|---|---|
| G1 | PR-AUC | survival >= catboost | 0,1643 | 0,1444 | **LULUS** |
| G2 | Recall@kapasitas 200/bln | survival >= catboost | 0,2816 | 0,3348 | **GAGAL** |
| G3 | Precision@kapasitas | >= 0,95 x catboost (0,2039) | 0,1805 | 0,2146 | **GAGAL** |
| G4 | Brier (kalibrasi 30d) | <= 1,10 x catboost (0,02365) | 0,0214 | 0,0215 | LULUS |

ROC-AUC (dilaporkan, TIDAK menggerbang - sama seperti `decide_promotion`
produksi): event-based 0,7437 vs catboost 0,8165, di bawah ambang
pertimbangan 0,78. Karena keputusan di sini SUDAH "tidak cutover" dari G2/G3,
penerimaan tertulis formal soal ROC-AUC (disyaratkan plan KALAU cutover)
tidak relevan - dicatat di sini demi transparansi, bukan didiamkan.

## G5-G6: ukuran artifact + latency (A2, kandidat compact)

Artifact produksi awal 5,26 GB (RSF 100 pohon, `min_samples_leaf=30`, grid
waktu penuh). Lever: perkasar target `duration_days` yang dilihat `.fit()`
(harian s/d 120 hari, kelipatan 60 hari di atasnya) + `n_estimators=50`,
`min_samples_leaf=100`. Evaluasi TETAP pakai `duration_days` asli (tidak
dikasarkan) - lihat `reports/gate_a2_compact_model.md` dan
`reports/gate_a2_compact_latency.md`.

| # | Kriteria | Ambang | Nilai | Status |
|---|---|---|---|---|
| G5 | Ukuran artifact | <=100 MB | 66,2 MB | **LULUS** |
| G6 | Cold load | <=5 s | 0,174 s | **LULUS** |
| G6 | Single predict p50 | <=3.467,7 ms (1,5x baseline CatBoost) | 2,7 ms | **LULUS** |
| G6 | Batch (ekstrapolasi 16.877 PART) | <=94,0 s (2x baseline CatBoost) | 2,7 s | **LULUS** |

Bonus tak terduga: kandidat compact justru **C-index VALIDATION lebih tinggi**
dari baseline produksi (0,8417 vs 0,8290) dan AUC@30 lebih tinggi (0,8509 vs
0,8357) - grid yang lebih kasar bertindak sebagai regularisasi, bukan
kompromi akurasi-vs-ukuran murni. IBS/Brier@30 nyaris identik.

Latency di atas diukur pada `predict_survival_function()` murni (model sudah
di-load, fitur sudah di-encode) - BUKAN end-to-end lewat pipeline database +
pembangunan fitur (itu perlu serving code Fase C yang sesungguhnya).

## G7: instalasi `scikit-survival` di image production (A4)

**LULUS, dengan mitigasi tercatat.** `pip install scikit-survival==0.28.0`
polos GAGAL di `python:3.13-slim` - `ecos` (dependensi, HANYA dipakai
`SurvivalSVM` yang tidak dipakai proyek ini) tidak punya wheel py3.13 dan
`gcc` tidak ada di image (`error: [Errno 2] No such file or directory: 'gcc'`).
Diperbaiki PERSIS seperti mitigasi R3 di plan: install dependensi riil
`scikit-survival` secara eksplisit (numpy, scipy, pandas, scikit-learn,
joblib, numexpr, osqp - osqp PUNYA wheel, tidak masalah) lalu
`pip install --no-deps scikit-survival==0.28.0`. Diverifikasi BUKAN cuma
import - `RandomSurvivalForest.fit()` + `predict_survival_function()`
dijalankan sungguhan di dalam container dan berhasil (`sksurv OK 0.28.0`).

`scikit-learn==1.9.0` (lock file sekarang) SAMA PERSIS dengan yang dipasang
di test ini - kompatibel, tidak perlu upgrade.

## G8: reproduksibilitas lintas snapshot DB

**BELUM diukur di sini - DITUNDA ke Fase C, bukan diabaikan.** A2/A3 sengaja
memakai `SURVIVAL_BUILD_CACHE` (dataset event-based dari cache lokal) untuk
kecepatan pencarian hyperparameter - itu valid untuk MENCARI konfigurasi
(bukan mengevaluasi keputusan fitur, beda dengan pelajaran
`reports/short_window.md`), tapi belum membuktikan reproduksibilitas lintas
pembacaan DB fresh. Setiap skrip A2/A3 mencatat ini eksplisit di output:
*"konfigurasi terpilih WAJIB dilatih ulang dari DB fresh sebelum jadi
artifact produksi"*. Karena keputusan di sini sudah "tidak cutover", G8 jadi
syarat SEBELUM model compact ini benar-benar di-ship sebagai artifact
`v3` (Fase C), bukan syarat gerbang hari ini.

## Konfigurasi pemenang (dicatat untuk Fase C)

```
RandomSurvivalForest(
    n_estimators=50, min_samples_split=140, min_samples_leaf=100,
    max_features="sqrt", n_jobs=1, random_state=42,
)
# target .fit() dikasarkan: harian s/d 120 hari, kelipatan 60 hari di atasnya
# (evaluasi/serving TETAP pakai duration_days asli, tidak dikasarkan)
# + 4 IsotonicRegression per horizon [30,60,90,120] + cummax lintas horizon
```

## Yang TIDAK diubah oleh keputusan ini

- `models/failure/CURRENT` tetap v2 CatBoost. Tidak ada perubahan production.
- Dashboard/API tidak disentuh oleh keputusan ini sendiri (field advisory,
  kalau jadi diimplementasikan di Fase C/D, aditif murni - lihat plan
  bagian Kontrak API).
- Restrukturisasi `src/partrisk` (Fase B) berjalan terlepas dari keputusan
  ini, sesuai plan.
