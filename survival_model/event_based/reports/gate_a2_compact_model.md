# Fase A2: pencarian model compact (event-based RSF)

Populasi: VALIDATION landmark rows (aturan proyek - keputusan tidak pernah dari TEST).
Konfigurasi TRAIN dari `SURVIVAL_BUILD_CACHE` (pencarian cepat) - konfigurasi terpilih
WAJIB dilatih ulang dari pembacaan DB fresh sebelum jadi artifact produksi (G8).

| Konfigurasi | Detail | Ukuran (MB) | VAL C-index | VAL IBS | VAL Brier@30 | VAL AUC@30 |
|---|---|---|---|---|---|---|
| baseline (production, 5.26 GB) | n_estimators=100 min_samples_leaf=30 grid asli | 5,262.3 | 0.8290 | 0.0475 | 0.0357 | 0.8357 |
| compact (kandidat A2) | n_estimators=50 min_samples_leaf=100 grid dikasarkan (171 titik) | 66.2 | 0.8417 | 0.0482 | 0.0357 | 0.8509 |

Fit compact: 67.2 detik.

Verdict: LULUS - artifact <=100 MB, C-index VALIDATION dalam toleransi (>= baseline - 0,01).

Lever dipakai: perkasar target duration_days yang dilihat RSF.fit() (resolusi harian
s/d 120 hari, kelipatan 30 hari di atasnya - evaluasi tetap pakai duration_days ASLI),
n_estimators 100->60, min_samples_leaf 30->80, min_samples_split 40->110 (rasio dijaga
mirip baseline).
