# Langkah B — Kurva S(t) terkalibrasi konsisten, diwire ke production

Lanjutan `reports/rsf_median_curve_baseline.md` (Langkah A). Rencana user
eksplisit meminta: *"Setelah kalibrasi horizon, bangun ulang / sesuaikan cara
baca median dari kurva yang sudah disesuaikan skala-nya (jangan median dari
kurva mentah + risk dari kurva kalibrasi — inkonsisten)."*

## Bug ditemukan sebelum hasil dipercaya

Prototipe pertama (`calibrated_survival_matrix()`) memakai interval TERBUKA
di kedua ujung (`times > h_lo) & (times < h_hi)`) untuk menginterpolasi antar
horizon terlatih. Titik grid yang PERSIS sama dengan horizon terlatih (t=60,
t=90) tidak masuk region MANA PUN - diverifikasi grid harian RSF production
MEMANG memuat t=60/t=90 persis (index 59/89 di grid 185 titik). Kolom itu
terisi `np.empty_like()` (memori tidak diinisialisasi), lalu ikut ter-`cummax`
ke titik-titik setelahnya - berisiko mencemari hasil di luar kolom itu sendiri.

Diperbaiki: interval setengah-terbuka `(h_lo, h_hi]` yang konsisten (setiap
titik grid masuk TEPAT SATU region) + `np.full(..., np.nan)` (bukan
`np.empty_like`) + `assert not np.isnan(...)` eksplisit - kalau ada celah
lagi di masa depan, gagal keras, bukan diam-diam salah. Dua run (sebelum/
sesudah fix) menghasilkan angka yang HAMPIR identik (garbage kebetulan
ter-overwrite cummax di kasus ini) - tapi bug-nya nyata dan bisa memberi hasil
salah di kondisi data lain, jadi tetap diperbaiki, bukan diabaikan karena
"kebetulan tidak berdampak kali ini". Regression test ditambahkan
(`tests/test_survival_curves.py`, 5 test, logic murni tanpa database).

## Metode

`curves.calibrate_curve(times, curve_values, calibrators)` (baru,
`src/partrisk/survival/curves.py`): raw_risk(t)=1-S(t) dipetakan lewat
isotonic per horizon TERLATIH (calibrators.joblib, TIDAK dilatih ulang) -
interpolasi linear antar dua horizon terdekat, flat-extrapolation di luar
rentang (t<=30 pakai calibrator_30, t>120 pakai calibrator_120), cummax WAJIB
di SELURUH grid. Dipakai KONSISTEN oleh `predict()` (median/p90/kurva
ditampilkan) dan `score_batch()` (median/p90) - `calibrated_risk_Nd` (4 titik
diskrit, Fase R1a) TIDAK diubah/tidak disentuh (tetap dihitung lewat
`_calibrate_risk()` yang sudah teruji, supaya nol risiko regresi di situ).

## Hasil (TEST landmark, sama persis populasi/definisi dengan Langkah A)

| Metrik | RAW (baseline) | TERKALIBRASI (Langkah B) | Perubahan |
|---|---:|---:|---:|
| **Seluruh TEST** — % median null | 79.3% | 60.4% | -18.9pp (lebih sering terisi) |
| — n usable (event & median terisi) | 257 | 386 | +50% |
| — MAE median | 751.9 hari | 450.0 hari | **-40.1%** |
| — Bias median (signed) | +751.9 hari | +448.5 hari | **-40.3%** |
| — % over-predict | 99.6% | 98.7% | ~tidak berubah |
| **ANCHOR saja (mirip serving)** — % median null | 87.8% | 75.0% | -12.8pp |
| — n usable | 18 | 27 | +50% |
| — MAE median | 1303.8 hari | 609.7 hari | **-53.2%** |
| — Bias median (signed) | +1303.8 hari | +609.7 hari | **-53.2%** |
| — % over-predict | 100.0% | 100.0% | tidak berubah |

Kalibrasi kurva (mean S(d) prediksi vs proporsi empiris masih hidup, gap):

| Horizon | Gap RAW | Gap TERKALIBRASI | Perubahan |
|---|---:|---:|---:|
| 30d | 0.028 | 0.030 | ~tidak berubah |
| 60d | 0.040 | 0.032 | -20% |
| 90d | 0.077 | 0.054 | -30% |
| 120d | 0.094 | 0.067 | -29% |
| 180d | 0.463 | 0.370 | -20% (lihat catatan follow-up window di Langkah A) |

Sanity: kurva tetap monoton non-increasing di semua baris (`np.diff(...) <=
1e-9` untuk semua kolom) - kalibrasi + interpolasi TIDAK merusak properti
dasar S(t).

## Verdict: **DITERIMA, diwire ke production**

Perbaikan nyata dan konsisten di HAMPIR SEMUA metrik: MAE turun 40-53%, gap
kalibrasi turun 20-30% di 60-180d, cakupan (% non-null) membaik. Ini dicapai
TANPA retrain model apa pun - murni memakai kalibrator yang SUDAH ada secara
konsisten di seluruh kurva, bukan cuma 4 titik diskrit seperti sebelumnya.

**Bias TIDAK hilang sepenuhnya** (median masih optimis rata-rata +448 s/d +609
hari, over-predict rate nyaris tidak bergerak dari ~99-100%). Ini SESUAI
EKSPEKTASI, bukan kegagalan: kalibrator hanya terlatih untuk horizon
30-120 hari, sementara median (saat terisi) rata-rata jatuh di ratusan hari -
jauh di luar rentang yang benar-benar dikalibrasi (flat-extrapolation di luar
120d menaikkan skala tapi tidak membentuk ulang LAJU penurunan kurva di sana).
Memperbaiki ini lebih lanjut butuh kalibrator terlatih di horizon lebih jauh
(180/365d) - TIDAK dikerjakan di sini (di luar cakupan "cepat, ROI tinggi";
lihat Langkah D/E rencana user untuk arah lanjutan) - didokumentasikan
sebagai keterbatasan diketahui, bukan diklaim selesai.

## Perubahan kode

- `src/partrisk/survival/curves.py`: fungsi baru `calibrate_curve()`.
- `src/partrisk/predict/survival.py`: `predict()` - median/p90/kurva
  ditampilkan dari kurva terkalibrasi; field baru `curve_is_calibrated`
  (True kalau calibrators ada). `score_batch()` - parameter baru
  `calibrators=None`, median/p90 dari kurva terkalibrasi kalau tersedia.
- `src/partrisk/serving/batch_predictor.py::_score_survival_advisory()` -
  meneruskan `calibrators` ke `score_batch()` (dulu sengaja dibuang/tidak
  dipakai, komentar lama sudah dihapus karena sekarang salah).
- `src/partrisk/serving/predictor.py::_survival_advisory_fields()` -
  `curve_is_calibrated` dibaca dari `predict()` (dulu hardcode `False`).
- `calibrated_risk_Nd` (Fase R1a, `_calibrate_risk()`) **TIDAK diubah** -
  tetap jalur terpisah yang sudah teruji, sekarang KONSISTEN secara nilai
  dengan kurva penuh di titik anchor (30/60/90/120), bukan cuma "kebetulan
  sama-sama terkalibrasi tapi dari sumber berbeda".
- Test baru: `tests/test_survival_curves.py` (5 test, unit murni, termasuk
  regression test untuk bug interval di atas) +
  `tests/test_parity.py::test_survival_kurva_terkalibrasi_monoton_turun_dan_flag_benar`
  (real model, DB-backed).
