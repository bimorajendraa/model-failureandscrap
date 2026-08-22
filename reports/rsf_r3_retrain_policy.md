# Fase R3 — Stabilitas & retrain policy RSF (opsional)

Tiga poin dari rencana upgrade RSF user, masing-masing dicek/diselesaikan:

## 1. Retrain lebih jarang (bulanan / kalau `data_end` bergeser jauh)

RSF model ini **advisory** - tidak menentukan ranking/urutan inspeksi (itu tetap
CatBoost, lihat gate_decision.md), jadi tidak perlu retrain tiap minggu seperti
CatBoost. Kebijakan ini sekarang tertulis eksplisit di docstring modul
`training/failure_survival.py`: retrain wajar bulanan, atau kapan pun `data_end`
sudah bergeser jauh (>60 hari) dari training terakhir - keduanya operasional
(dijalankan manual/cron di luar repo), bukan sesuatu yang perlu dipaksa lewat kode.

## 2. Compact artifact + n_jobs=1 di serve

Sudah ada, diverifikasi tetap ada, TIDAK diubah:
- `COMPACT_RSF_PARAMS` (`n_estimators=50, min_samples_leaf=100, ..., n_jobs=1`) -
  artifact ~66 MB (vs 5,26 GB konfigurasi riset lama), lihat docstring modul.
- `predict/survival.py::load_model()` eksplisit set `model.n_jobs = 1` setelah
  `joblib.load()`, terlepas dari nilai n_jobs saat training - mencegah RSF
  ter-unpickle + `n_jobs=-1` hang tanpa error saat `predict_survival_function()`
  (R7 di rencana restrukturisasi awal).

## 3. Gate ringan: promote HANYA kalau Brier@30/90 tidak memburuk

**Diimplementasikan sebagai kode, bukan cuma dokumentasi** - `decide_survival_promotion()`
(baru, `training/failure_survival.py`), dipanggil dari `main()` SEBELUM
`joblib.dump()`/artifact ditulis:

- Kalau belum ada artifact production (`metadata.json` belum ada): lolos otomatis
  (training pertama kali).
- Kalau sudah ada: baca `evaluation_metrics_full_landmark_rows.random_survival_forest.test`
  dari `metadata.json` LAMA sebagai incumbent, bandingkan dengan metrik TEST
  kandidat yang baru saja dilatih di run yang sama - **Brier@30d DAN Brier@90d
  kandidat harus <= incumbent**, keduanya, bukan salah satu.
- Kalau gagal: `main()` mencetak alasan, **TIDAK menimpa artifact** (`ARTIFACTS_DIR`
  tetap berisi model lama), return exit code 1 - training gagal secara EKSPLISIT,
  bukan diam-diam menimpa model yang lebih buruk.
- Sengaja BUKAN dual-gate PR-AUC/Recall@kapasitas ala `training.versioning.decide_promotion`
  (CatBoost) - model ini tidak dipakai untuk ranking, jadi metrik ranking (PR-AUC/
  Recall@kapasitas) tidak relevan untuk gate-nya; Brier per horizon (kualitas
  probabilitas mentah) yang relevan untuk field advisory `risk_Nd`/median/90pct.
- Normalisasi kunci horizon int vs string ditangani eksplisit (`metadata.json`
  hasil `json.load()` punya kunci `"30"`/`"90"` string, metrik live dari
  `model_fit.evaluate_models()` punya kunci int 30/90) - diuji lewat
  `tests/test_promotion.py::test_survival_incumbent_dari_metadata_json_kunci_string_tetap_terbaca`.
- Unit test lengkap (6 kasus baru: lolos pertama kali, kandidat lebih baik,
  Brier@30d memburuk menahan walau 90d membaik dan sebaliknya, identik lolos,
  normalisasi kunci) - `tests/test_promotion.py`, semua hijau tanpa database
  (logic murni dengan dict sintetis, pola sama dengan `decide_promotion()`).

## Perbaikan kecil yang ikut ditemukan: metadata stale

`metadata.json`'s `calibration.applied_to_advisory_fields` masih tertulis `false`
sejak Fase R1(a) mengaktifkan kalibrasi ke `calibrated_risk_*d` (commit `be83a03`) -
field itu HANYA menjelaskan status pemakaian, R1(a) tidak retrain ulang jadi
metadata di disk tidak ikut ter-update otomatis. Diperbaiki dua tempat: file
`metadata.json` yang ada sekarang (patch langsung, factual correction, bukan
retrain), dan sumber di `training/failure_survival.py` (supaya retrain berikutnya
otomatis menulis `true`).

## Status Fase R3

Selesai. Tidak ada eksperimen ablation di sini (murni kebijakan + satu gate kode
kecil) - sesuai sifatnya yang ditandai opsional oleh user, tidak ada alasan
menginvestasikan lebih dari ini.
