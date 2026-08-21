# Fase A3: studi kalibrasi (event-based, konfigurasi compact A2)

Populasi: VALIDATION landmark rows (aturan proyek - keputusan tidak pernah dari TEST).
Model: kandidat compact A2 (n_estimators=50,
min_samples_leaf=100, grid dikasarkan).

## Brier per horizon, mentah vs terkalibrasi (isotonic independen per horizon)

| Horizon | Baris terpakai | Kejadian | Brier mentah | Brier terkalibrasi |
|---|---|---|---|---|
| 30d | 5,147 | 227 | 0.0374 | 0.0367 |
| 60d | 4,154 | 330 | 0.0614 | 0.0594 |
| 90d | 3,763 | 400 | 0.0769 | 0.0728 |
| 120d | 3,374 | 441 | 0.0880 | 0.0829 |

## Monotonisitas lintas horizon (30<=60<=90<=120)

- Pelanggaran SEBELUM cummax: 527/5,540 baris
  (isotonic per horizon dikalibrasi TERPISAH, jadi ini diharapkan bukan 0 - lihat
  docstring skrip). cummax lintas [30,60,90,120] WAJIB, sama seperti
  `tests/test_parity.py:132-139` menegaskan untuk CatBoost hazard-chaining.
- Pelanggaran SESUDAH cummax: 0 (harus 0 - diverifikasi lewat assert).

## Dampak ke risk_cutoffs (HIGH>=0.25, MEDIUM>=0.15) pada risiko 30 hari

| | HIGH | MEDIUM |
|---|---|---|
| Skor mentah (1-S(30)) | 141 | 245 |
| Skor terkalibrasi | 245 | 163 |

Populasi VALIDATION landmark (5,540 baris) - BUKAN populasi PART aktif
production (itu perlu skor pada `predict.py`-style observation_on, lihat A1),
cuma untuk melihat ARAH dan BESAR pergeseran akibat kalibrasi.
