# Fase A1: evaluasi production-realistic event-based vs CatBoost (Gate)

Populasi: 38,451 baris TEST classification (window 211 hari, kapasitas 200/bulan) - SAMA PERSIS untuk kedua model (rows_matched = seluruh populasi, tidak ada baris yang gugur di sisi event-based).

Beda dari angka lama (reports/ensemble_operational.md, PR-AUC 0,1824): fitur event-based DIHITUNG PADA `observation_on` tiap baris (kondisi PART SAAT snapshot classification itu diambil), BUKAN dibekukan di `installed_on`. Ini evaluasi yang merepresentasikan cara `predict.py` benar-benar dipakai (skor kondisi SEKARANG).

| Model | PR-AUC | ROC-AUC | Recall@cap | Precision@cap | Brier |
|---|---|---|---|---|---|
| event-based (observation_on) | 0.1643 | 0.7437 | 0.2816 | 0.1805 | 0.0214 |
| catboost v2 (incumbent, dihitung ulang) | 0.1444 | 0.8165 | 0.3348 | 0.2146 | 0.0215 |

rows_matched = 38,451 (populasi identik untuk kedua model)