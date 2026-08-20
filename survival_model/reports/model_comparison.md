# RSF tuning & perbandingan model final

Pencarian KECIL coordinate-wise (bukan grid penuh 2x4x3x3=72) di sekitar titik hyperparameter current - satu sumbu diubah per langkah, dipertahankan hanya kalau menaikkan VAL C-index. TEST hanya untuk pelaporan akhir.

Kolom operasional (PR-AUC30/ROC-AUC/Recall/Precision@kapasitas) kosong di sini dengan sengaja - membangun ulang populasi TEST classification (1,4 juta baris) terbukti langkah paling berat & paling rentan macet di lingkungan eksperimen ini. Angka operasional untuk konfigurasi final didapat dengan menjalankan `python evaluate.py` SETELAH `train.py` diupdate ke konfigurasi ini - lihat `reports/evaluation_report.md`.

## Pencarian tuning
| Experiment | VAL C-index | TEST C-index | Uno C | AUC30 | AUC90 | IBS | PR-AUC30(op) | ROC-AUC(op) | Recall@cap | Precision@cap |
|---|---|---|---|---|---|---|---|---|---|
| tuning=current(baseline) (random_survival_forest) | 0.8120 | 0.8096 | 0.8097 | 0.8467 | 0.8859 | 0.0809 | N/A | N/A | N/A | N/A |
| tuning=n_estimators=200 (random_survival_forest) | 0.8095 | 0.8084 | 0.8085 | 0.8450 | 0.8836 | 0.0807 | N/A | N/A | N/A | N/A |
| tuning=n_estimators=400 (random_survival_forest) | 0.8100 | 0.8096 | 0.8097 | 0.8456 | 0.8840 | 0.0804 | N/A | N/A | N/A | N/A |
| tuning=min_samples_leaf=10 (random_survival_forest) | 0.8100 | 0.8082 | 0.8083 | 0.8454 | 0.8844 | 0.0798 | N/A | N/A | N/A | N/A |
| tuning=min_samples_leaf=20 (random_survival_forest) | 0.8102 | 0.8074 | 0.8075 | 0.8441 | 0.8831 | 0.0803 | N/A | N/A | N/A | N/A |
| tuning=min_samples_leaf=50 (random_survival_forest) | 0.8069 | 0.8083 | 0.8084 | 0.8456 | 0.8844 | 0.0811 | N/A | N/A | N/A | N/A |
| tuning=max_features=0.5 (random_survival_forest) | 0.8069 | 0.7945 | 0.7946 | 0.8286 | 0.8697 | 0.0813 | N/A | N/A | N/A | N/A |
| tuning=max_features=1.0 (random_survival_forest) | 0.8087 | 0.7805 | 0.7807 | 0.8097 | 0.8537 | 0.0839 | N/A | N/A | N/A | N/A |
| tuning=max_depth=8 (random_survival_forest) | 0.8100 | 0.8091 | 0.8092 | 0.8454 | 0.8828 | 0.0810 | N/A | N/A | N/A | N/A |
| tuning=max_depth=12 (random_survival_forest) | 0.8058 | 0.8072 | 0.8073 | 0.8447 | 0.8840 | 0.0807 | N/A | N/A | N/A | N/A |

## Model final (RSF vs Cox PH, fitur+threshold+hyperparameter terpilih)
| Experiment | VAL C-index | TEST C-index | Uno C | AUC30 | AUC90 | IBS | PR-AUC30(op) | ROC-AUC(op) | Recall@cap | Precision@cap |
|---|---|---|---|---|---|---|---|---|---|
| FINAL (random_survival_forest) (random_survival_forest) | 0.8120 | 0.8096 | 0.8097 | 0.8467 | 0.8859 | 0.0809 | N/A | N/A | N/A | N/A |
| FINAL (cox_ph) (cox_ph) | 0.7858 | 0.7963 | 0.7964 | 0.8342 | 0.8755 | 0.0942 | N/A | N/A | N/A | N/A |