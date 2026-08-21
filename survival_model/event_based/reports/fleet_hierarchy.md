# H_plus_fleet_hierarchy: fleet failure rate level item_type (Fase 3)

**Strategi baru**: C-index TIDAK dikejar lagi (sudah terbukti mentok ~0,80 lewat banyak percobaan) - baris `0_baseline_production` adalah PAGAR (floor), kandidat HANYA layak diadopsi kalau C-index-nya TIDAK TURUN dari baseline DAN AUC-30d/90d (proxy murah untuk Recall@kapasitas operasional, tidak perlu bangun ulang populasi TEST classification 1,4 juta baris) NAIK.

| Experiment | Model | VAL C-index (full) | VAL C-index (t0-only, ADIL) | VAL t0 IBS | VAL t0 AUC-30d | VAL t0 AUC-90d |
|---|---|---|---|---|---|---|
| 0_baseline_production | random_survival_forest | 0.8290 | 0.7985 | 0.0781 | 0.7862 | 0.8265 |
| 0_baseline_production | cox_ph | 0.7915 | 0.7651 | 0.0907 | 0.7453 | 0.7947 |
| H_plus_fleet_hierarchy | random_survival_forest | 0.8300 | 0.7994 | 0.0796 | 0.7865 | 0.8285 |
| H_plus_fleet_hierarchy | cox_ph | 0.7771 | 0.7527 | 0.0919 | 0.7326 | 0.7805 |