# Keluarga model: RSF vs Cox PH vs ExtraSurvivalTrees vs GBSA vs ComponentwiseGBSA

Semua model dilatih pada fitur FINAL PRODUKSI yang SAMA PERSIS (src/features.py, threshold item_model=200/item_type=300 - lihat README poin 4 & 8), hyperparameter default per keluarga (BUKAN hasil tuning - tuning per-keluarga adalah langkah terpisah). Tujuannya mengisolasi kontribusi KELUARGA MODEL saja, terpisah dari kontribusi fitur (yang sudah diaudit habis di feature_ablation.md/previous_cycle_audit.md - lihat README poin 5-6 dan 11).

`GradientBoostingSurvivalAnalysis(loss='ipcwls'/'squared')` DIUJI lewat smoke test dan DIBUANG dari registry (bukan dilewati tanpa dicoba): loss selain 'coxph' tidak punya baseline hazard model, `predict_survival_function()`-nya melempar ValueError - tidak kompatibel dengan seluruh pipeline di sini (evaluate.py IBS/Brier/AUC, predict.py) yang butuh kurva S(t) di SETIAP model, bukan cuma skor risiko. Lihat catatan di src/model_fit.py.

Bandingkan angka di sini dengan reports/uncertainty_baseline.md - kandidat hanya dianggap menang kalau naiknya di luar rentang ketidakpastian baseline di sana, bukan menang tipis.

| Experiment | VAL C-index | TEST C-index | Uno C | AUC30 | AUC90 | IBS | PR-AUC30(op) | ROC-AUC(op) | Recall@cap | Precision@cap |
|---|---|---|---|---|---|---|---|---|---|
| model_family (random_survival_forest) | 0.8114 | 0.8082 | 0.8083 | 0.8424 | 0.8827 | 0.0811 | N/A | N/A | N/A | N/A |
| model_family (cox_ph) | 0.7819 | 0.7722 | 0.7724 | 0.8027 | 0.8449 | 0.0950 | N/A | N/A | N/A | N/A |
| model_family (extra_survival_trees) | 0.8080 | 0.8111 | 0.8112 | 0.8474 | 0.8848 | 0.0833 | N/A | N/A | N/A | N/A |
| model_family (gbsa_coxph) | 0.8072 | 0.8256 | 0.8257 | 0.8692 | 0.9038 | 0.0885 | N/A | N/A | N/A | N/A |
| model_family (componentwise_gbsa) | 0.7937 | 0.8314 | 0.8315 | 0.8698 | 0.9049 | 0.0874 | N/A | N/A | N/A | N/A |