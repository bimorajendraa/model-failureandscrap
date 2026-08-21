# I_plus_short_window_7_14d: jendela corrective sangat dekat (Fase 3)

Baseline (0_baseline_production) ada di reports/fleet_hierarchy.md (VAL t0=0,7985, AUC30=0,7862) - dibandingkan di sini tanpa diulang.

| Experiment | Model | VAL C-index (full) | VAL C-index (t0-only, ADIL) | VAL t0 IBS | VAL t0 AUC-30d | VAL t0 AUC-90d |
|---|---|---|---|---|---|---|
| I_plus_short_window_7_14d | random_survival_forest | 0.8352 | 0.8058 | 0.0778 | 0.7948 | 0.8351 |
| I_plus_short_window_7_14d | cox_ph | 0.7933 | 0.7639 | 0.0933 | 0.7437 | 0.7947 |

## TINDAK LANJUT: DIBATALKAN setelah verifikasi retrain penuh

Tabel di atas dihitung dari `build_dataset.build()` yang memakai **cache lama**
(`SURVIVAL_BUILD_CACHE=1`, tidak membaca ulang database). Setelah fitur ini
di-wire ke produksi dan di-retrain dengan **cache dihapus** (database dibaca
FRESH), hasilnya menunjukkan **REGRESI di semua metrik** dibanding baseline
sebelum perubahan ini (VAL t0-only C-index turun ke 0,7974, TEST turun ke
0,8065, Recall@kapasitas turun ke 0,3345, PR-AUC turun ke 0,1817) - melanggar
syarat "tidak boleh lebih buruk dari sebelumnya".

**Kesimpulan**: jendela SANGAT pendek (7/14 hari) rupanya terlalu sensitif
terhadap kapan tepatnya data ditarik dari database yang terus berjalan
(live) - snapshot cache lama dan snapshot database fresh menghasilkan nilai
fitur yang cukup berbeda untuk baris-baris TEST/VALIDATION yang
`observation_on`-nya dekat dengan "sekarang", sampai membalik kesimpulan
menang/kalah. **DIBATALKAN, dikembalikan ke jendela 60/90 hari yang stabil**
(lihat `eb_src/dynamic_history.py` - `windowed_corrective_extra()` default
kembali ke `(60, 90)`, dan `eb_src/features.py`). Fitur produksi final tetap
di VAL t0-only 0,7985 / TEST 0,8105 / Recall@kapasitas 0,3401 (lihat
`reports/evaluation_report.md`).

**Pelajaran metodologis**: fitur berbasis jendela waktu yang SANGAT sempit,
diuji lewat cache yang bisa basi, TIDAK cukup divalidasi - butuh retrain
penuh pada snapshot data yang SAMA PERSIS dengan yang akan dipakai produksi
sebelum benar-benar diadopsi, bukan hanya lolos di satu kali percobaan ablasi.