# Concept drift: jendela tahun TRAIN (event-based, fitur produksi final)

TRAIN dipangkas berdasar `installed_on` LIFECYCLE (bukan observation_on landmark) - satu lifecycle tidak pernah terpotong di tengah. VALIDATION identik di semua baris (t0-only, sama seperti reports/evaluation_report.md).

| Experiment | Model | VAL C-index (full) | VAL C-index (t0-only, ADIL) | VAL t0 IBS |
|---|---|---|---|---|
| 2014-2024 (penuh) (14,980 lifecycle) | random_survival_forest | 0.7963 | 0.7963 | 0.0777 |
| 2018-2024 (10,712 lifecycle) | random_survival_forest | 0.8065 | 0.8065 | 0.0780 |
| 2020-2024 (6,806 lifecycle) | random_survival_forest | 0.7961 | 0.7961 | 0.0843 |
| 2022-2024 (5,285 lifecycle) | random_survival_forest | 0.8067 | 0.8067 | 0.0830 |