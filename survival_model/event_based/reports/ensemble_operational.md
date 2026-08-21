# Ensemble operasional: model statis + event-based (Fase 4)

Populasi irisan (kedua model punya skor pada baris yang sama): 37,923 baris, window 211 hari, kapasitas 200/bulan. `static_only`/`event_based_only` di sini dihitung ULANG pada populasi IRISAN (bukan angka lama dari evaluation_report.md masing-masing, yang populasinya beda) - supaya perbandingan adil.

| Kandidat | PR-AUC | ROC-AUC | Recall@cap | Precision@cap | Brier |
|---|---|---|---|---|---|
| static_only (populasi irisan) | 0.1633 | 0.6871 | 0.3108 | 0.1962 | 0.0214 |
| event_based_only (populasi irisan) | 0.1824 | 0.6961 | 0.3401 | 0.2146 | 0.0212 |
| ensemble_avg_raw | 0.1756 | 0.6882 | 0.3311 | 0.2090 | 0.0213 |
| ensemble_avg_rank | 0.1756 | 0.6913 | 0.3322 | 0.2097 | 0.3192 |
| ensemble_max | 0.1751 | 0.6860 | 0.3255 | 0.2054 | 0.0212 |