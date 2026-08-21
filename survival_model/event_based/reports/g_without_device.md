# G_combined_without_device (tanpa dependency schema analytics)

Bandingkan dengan F_combined_all (VAL t0-only RSF=0,8036) di reports/dynamic_ablation.md.

| Experiment | Model | VAL C-index (full) | VAL C-index (t0-only, ADIL) | VAL t0 IBS |
|---|---|---|---|---|
| G_combined_without_device | random_survival_forest | 0.8270 | 0.7954 | 0.0777 |
| G_combined_without_device | cox_ph | 0.7961 | 0.7684 | 0.0908 |