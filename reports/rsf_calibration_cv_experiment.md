# Eksperimen: kalibrasi RSF via 5-fold CV (TRAIN+VALIDATION) untuk perbaiki underestimate di bucket risiko tertinggi

## Hipotesis

`reports/rsf_r1_evaluation.md` #3 menemukan `calibrators.joblib` production
*underestimate* signifikan di bucket risiko tertinggi (risk_30d: prediksi 0,17 vs
aktual 0,28; risk_90d: prediksi 0,35 vs aktual 0,49), sementara 3 bucket risiko
rendah terkalibrasi baik. Dugaan akar masalah: calibrator SEKARANG cuma dilatih di
VALIDATION (5.540 baris, 534 event_observed total lintas semua landmark) - terlalu
sedikit event untuk mengunci isotonic dengan baik persis di ekor yang jarang data,
dan isotonic dikenal rawan overfit di daerah sparse seperti ini.

Hipotesis perbaikan: kalau calibrator dilatih pada populasi yang jauh lebih besar
lewat out-of-fold prediction (bukan cuma VAL), estimasi di ekor risiko tinggi akan
lebih stabil dan tidak underestimate.

## Metodologi

- 5-fold CV di level **LIFECYCLE** (bukan baris landmark) pada TRAIN+VALIDATION
  gabungan (97.838 baris, 17.296 lifecycle, 11.741 event total) - split di level
  lifecycle mencegah kebocoran antar-fold, sama seperti disiplin split
  TRAIN/VAL/TEST asli (`features/survival/landmarks.py`).
- Tiap fold: fit RSF **compact params production** (`COMPACT_RSF_PARAMS`, sama
  persis dengan yang dipakai model yang di-deploy) pada 4/5 bagian, prediksi
  raw_risk pada 1/5 sisanya (out-of-fold, encoder PRODUCTION dipakai apa adanya,
  tidak di-refit per fold - one-hot encoding tidak memakai target jadi tidak ada
  risiko leakage dari situ).
- Prediksi OOF dikumpulkan utk SELURUH TRAIN+VAL (97.838 baris -~18x lebih banyak
  event daripada VAL sendirian: 11.741 vs 534), lalu isotonic final di-fit di situ.
- Model RSF yang DI-DEPLOY **tidak diganti** - eksperimen ini hanya menyoal
  `calibrators.joblib`. TEST tetap 100% tidak tersentuh CV (tetap holdout jujur).
- Evaluasi: reliability table yang SAMA persis dengan `rsf_r1_evaluation.md`
  (5 bucket TEST, kalibrasi lama vs kalibrasi CV baru berdampingan).

## Hasil

**risk_30d, TEST (n_label=4.382):**

| Bucket | n | aktual | pred lama | gap lama | pred CV | gap CV |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 877 | 0.0000 | 0.0000 | 0.0000 | 0.0005 | 0.0005 |
| 2 | 877 | 0.0011 | 0.0000 | 0.0011 | 0.0013 | 0.0002 |
| 3 | 876 | 0.0046 | 0.0021 | 0.0025 | 0.0077 | 0.0031 |
| 4 | 876 | 0.0731 | 0.0396 | 0.0335 | 0.0412 | 0.0319 |
| **5 (tertinggi)** | 876 | **0.2808** | 0.1699 | **0.1109** | 0.2267 | **0.0541** |

**risk_90d, TEST (n_label=2.430):**

| Bucket | n | aktual | pred lama | gap lama | pred CV | gap CV |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 486 | 0.0021 | 0.0000 | 0.0021 | 0.0014 | 0.0007 |
| 2 | 486 | 0.0062 | 0.0000 | 0.0062 | 0.0031 | 0.0031 |
| 3 | 486 | 0.0329 | 0.0160 | 0.0169 | 0.0272 | 0.0057 |
| 4 | 486 | 0.2428 | 0.1266 | 0.1162 | 0.0968 | **0.1460 (memburuk)** |
| **5 (tertinggi)** | 486 | **0.4877** | 0.3466 | 0.1411 | 0.3242 | **0.1635 (memburuk)** |

## Interpretasi

**Campuran, bukan kemenangan bersih.** risk_30d membaik jelas di bucket tertinggi
(gap 0,111 -> 0,054, hampir separuh) dan sedikit di bucket 3. Tapi risk_90d justru
**memburuk** di dua bucket teratas (bucket 4: gap 0,116 -> 0,146; bucket 5: gap
0,141 -> 0,164) - berlawanan arah dengan hipotesis.

Dugaan penyebab: OOF raw_risk dihasilkan dari **5 model RSF sementara** (tiap fold
model beda, dilatih di subset data yang beda), bukan dari model production yang
sesungguhnya di-deploy. Kalau distribusi raw score model-model sementara ini sedikit
berbeda dari model production (meski hyperparameter identik - variasi wajar antar
fit RSF dengan random_state, ukuran train set, dan komposisi lifecycle berbeda),
pemetaan isotonic yang dipelajari dari situ tidak otomatis pas saat diterapkan ke
raw_risk model production yang sebenarnya. Efek ini kemungkinan lebih terasa di
horizon 90d daripada 30d karena alasan yang belum jelas - tidak diselidiki lebih
jauh sesuai semangat "jangan over-invest" Fase R1.

## Verdict: **DITOLAK sebagai pengganti penuh** `calibrators.joblib`

`calibrators.joblib` production **tidak diubah**. Kandidat CV (skrip di scratchpad,
`calibrators_cv_candidate.joblib` sengaja dihapus setelah eksperimen - gitignored,
reproducible dari skrip kalau perlu diulang) memperbaiki satu horizon tapi
memperburuk horizon lain dengan besaran yang sebanding - bukan trade-off yang jelas
menguntungkan, dan mengganti seluruh kalibrator berbasis hasil campuran seperti ini
berisiko lebih besar daripada manfaatnya.

Underestimate di bucket risiko tertinggi risk_30d/90d **tetap ada** dan didokumentasikan
sebagai keterbatasan diketahui (lihat `rsf_r1_evaluation.md` #3) - bukan sesuatu yang
disembunyikan, hanya belum ada perbaikan yang terbukti aman untuk dipasang. Opsi
lanjutan yang belum dicoba (di luar cakupan "cepat, ROI tinggi" R1, kalau suatu saat
relevan): kalibrasi per-horizon terpisah dengan metode CV yang menghasilkan OOF dari
model FINAL (bukan model sementara per-fold) - lebih rumit, butuh 2 tahap CV
bersarang, tidak dikerjakan di sini.
