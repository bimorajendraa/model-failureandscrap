# Audit previous_cycle_lifetime_mean

`previous_cycle_lifetime_mean` (dari `data_reader.get_cycles()` SQL) TERBUKTI mencampur rata-rata durasi siklus sebelumnya APAPUN cara berakhirnya (FAILURE, RIGHT_CENSORED_AT_DATA_END, REINSTALL_WITHOUT_RECORDED_FAILURE) - bukan murni "lifetime sampai gagal" seperti namanya. Diuji di atas konfigurasi terbaik (A_plus_item_type) dari tahap ablation.

| Experiment | VAL C-index | TEST C-index | Uno C | AUC30 | AUC90 | IBS | PR-AUC30(op) | ROC-AUC(op) | Recall@cap | Precision@cap |
|---|---|---|---|---|---|---|---|---|---|
| prev_cycle=existing (random_survival_forest) | 0.8109 | 0.8015 | 0.8016 | 0.8358 | 0.8781 | 0.0809 | N/A | N/A | N/A | N/A |
| prev_cycle=confirmed_failure_only (random_survival_forest) | 0.8120 | 0.8096 | 0.8097 | 0.8467 | 0.8859 | 0.0809 | N/A | N/A | N/A | N/A |
| prev_cycle=last_confirmed_failure (random_survival_forest) | 0.8093 | 0.8106 | 0.8107 | 0.8479 | 0.8863 | 0.0804 | N/A | N/A | N/A | N/A |
| prev_cycle=confirmed_failure_only+end_reason (random_survival_forest) | 0.8113 | 0.8108 | 0.8109 | 0.8474 | 0.8857 | 0.0810 | N/A | N/A | N/A | N/A |

Varian terpilih: **confirmed_failure_only** (VAL C-index 0.8120 vs existing 0.8109). previous_cycle_end_reason TIDAK dipertahankan (VAL C-index 0.8113 <= 0.8120).