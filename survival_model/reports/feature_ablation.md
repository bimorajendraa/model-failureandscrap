# Feature ablation: current vs context-only vs combined

A = fitur classification warisan (19 kolom, tidak berubah). B = HANYA konteks instalasi (part model/client/item type/lokasi, threshold khusus survival dari category_threshold.md, TANPA riwayat/armada/lifecycle). C = A + item_type_at_install + place_at_install (part_model/client sudah ada di A). A_plus_* mengisolasi kontribusi 1 fitur baru saja.

| Experiment | VAL C-index | TEST C-index | Uno C | AUC30 | AUC90 | IBS | PR-AUC30(op) | ROC-AUC(op) | Recall@cap | Precision@cap |
|---|---|---|---|---|---|---|---|---|---|
| A_current (random_survival_forest) | 0.8078 | 0.8051 | 0.8052 | 0.8377 | 0.8820 | 0.0801 | N/A | N/A | N/A | N/A |
| A_current (cox_ph) | 0.7706 | 0.8149 | 0.8150 | 0.8564 | 0.8929 | 0.0868 | N/A | N/A | N/A | N/A |
| B_context_only (random_survival_forest) | 0.6547 | 0.6227 | 0.6226 | 0.6408 | 0.6415 | 0.1044 | N/A | N/A | N/A | N/A |
| B_context_only (cox_ph) | 0.6678 | 0.6245 | 0.6246 | 0.6372 | 0.6454 | 0.1064 | N/A | N/A | N/A | N/A |
| A_plus_item_type (random_survival_forest) | 0.8118 | 0.8034 | 0.8035 | 0.8378 | 0.8815 | 0.0811 | N/A | N/A | N/A | N/A |
| A_plus_item_type (cox_ph) | 0.7809 | 0.7965 | 0.7966 | 0.8365 | 0.8778 | 0.0940 | N/A | N/A | N/A | N/A |
| A_plus_place (random_survival_forest) | 0.8089 | 0.8091 | 0.8092 | 0.8436 | 0.8846 | 0.0814 | N/A | N/A | N/A | N/A |
| A_plus_place (cox_ph) | 0.7625 | 0.7984 | 0.7984 | 0.8394 | 0.8743 | 0.0908 | N/A | N/A | N/A | N/A |
| C_combined (random_survival_forest) | 0.8074 | 0.8036 | 0.8037 | 0.8365 | 0.8775 | 0.0826 | N/A | N/A | N/A | N/A |
| C_combined (cox_ph) | 0.7716 | 0.7826 | 0.7827 | 0.8221 | 0.8606 | 0.0962 | N/A | N/A | N/A | N/A |