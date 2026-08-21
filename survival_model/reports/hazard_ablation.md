# Fitur hazard baru: prior survival empiris per grup (Fase 2)

F1 dari plan peningkatan C-index - untuk tiap lifecycle: di antara lifecycle LAIN pada grup yang sama (part model/item type/client) yang SUDAH BERAKHIR sebelum installed_on baris ini (point-in-time, mekanisme sama dengan feature_builder.attach_fleet - dihitung dari populasi is_initial_model_cohort PENUH, bukan dibatasi lifecycle eligible survival), berapa yang berakhir FAILURE dan berapa median durasinya. A_final = fitur produksi saat ini (tidak berubah, baseline). A_plus_* mengisolasi kontribusi 1 grup saja. Bandingkan dengan reports/uncertainty_baseline.md - hanya dianggap menang kalau naiknya di luar rentang ketidakpastian baseline di sana. Lihat src/hazard_features.py untuk definisi lengkap.

| Experiment | VAL C-index | TEST C-index | Uno C | AUC30 | AUC90 | IBS | PR-AUC30(op) | ROC-AUC(op) | Recall@cap | Precision@cap |
|---|---|---|---|---|---|---|---|---|---|
| A_final (random_survival_forest) | 0.8114 | 0.8082 | 0.8083 | 0.8424 | 0.8827 | 0.0811 | N/A | N/A | N/A | N/A |
| A_final (cox_ph) | 0.7819 | 0.7722 | 0.7724 | 0.8027 | 0.8449 | 0.0950 | N/A | N/A | N/A | N/A |
| A_plus_partmodel_prior (random_survival_forest) | 0.8135 | 0.8056 | 0.8057 | 0.8387 | 0.8774 | 0.0834 | N/A | N/A | N/A | N/A |
| A_plus_partmodel_prior (cox_ph) | 0.7743 | 0.7600 | 0.7600 | 0.7911 | 0.8225 | 0.0930 | N/A | N/A | N/A | N/A |
| A_plus_itemtype_prior (random_survival_forest) | 0.8136 | 0.7975 | 0.7975 | 0.8332 | 0.8752 | 0.0887 | N/A | N/A | N/A | N/A |
| A_plus_itemtype_prior (cox_ph) | 0.7806 | 0.7508 | 0.7510 | 0.7769 | 0.8218 | 0.1067 | N/A | N/A | N/A | N/A |
| A_plus_client_prior (random_survival_forest) | 0.8108 | 0.8099 | 0.8099 | 0.8473 | 0.8882 | 0.0852 | N/A | N/A | N/A | N/A |
| A_plus_client_prior (cox_ph) | 0.6318 | 0.7546 | 0.7547 | 0.7792 | 0.8195 | 0.1104 | N/A | N/A | N/A | N/A |
| A_plus_all_priors (random_survival_forest) | 0.8124 | 0.8044 | 0.8045 | 0.8435 | 0.8793 | 0.0881 | N/A | N/A | N/A | N/A |
| A_plus_all_priors (cox_ph) | 0.6267 | 0.7754 | 0.7756 | 0.8002 | 0.8356 | 0.1079 | N/A | N/A | N/A | N/A |