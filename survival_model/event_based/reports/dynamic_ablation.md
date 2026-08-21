# Ablation lanjutan event-based: dynamic history + device/terminal (Fase 2)

Semua konfigurasi ditambahkan DI ATAS A_t0_baseline (fitur event-based final saat ini, VAL t0-only 0,7849 - lihat reports/evaluation_report.md). Keputusan dari **VAL t0-only** (kolom ke-4, SEBANDING dengan C-index model statis) - VAL full (kolom ke-3) TIDAK dipakai memilih (repeated measures, lihat README). E_plus_device_terminal memakai schema `analytics` (riset lama, BUKAN live production - lihat config.py) dengan filter `parent_link_quality_status=='VALID_POINT_IN_TIME_RELATION'` di observasi PALING AWAL tiap cycle - cycle yang relasinya baru diketahui SETELAH instalasi diberi UNKNOWN, bukan diam-diam dipakai.

| Experiment | Model | VAL C-index (full) | VAL C-index (t0-only, ADIL) | VAL t0 IBS |
|---|---|---|---|---|
| A_t0_baseline | random_survival_forest | 0.8173 | 0.7849 | 0.0774 |
| A_t0_baseline | cox_ph | 0.7905 | 0.7612 | 0.0885 |
| B_plus_degradation_trend | random_survival_forest | 0.8199 | 0.7890 | 0.0775 |
| B_plus_degradation_trend | cox_ph | 0.7890 | 0.7593 | 0.0895 |
| C_plus_cumulative_history | random_survival_forest | 0.8242 | 0.7910 | 0.0783 |
| C_plus_cumulative_history | cox_ph | 0.7974 | 0.7702 | 0.0889 |
| D_plus_windowed_corrective | random_survival_forest | 0.8224 | 0.7929 | 0.0772 |
| D_plus_windowed_corrective | cox_ph | 0.7900 | 0.7607 | 0.0897 |
| E_plus_device_terminal | random_survival_forest | 0.8206 | 0.7884 | 0.0776 |
| E_plus_device_terminal | cox_ph | 0.7327 | 0.7119 | 0.0916 |
| F_combined_all | random_survival_forest | 0.8313 | 0.8036 | 0.0783 |
| F_combined_all | cox_ph | 0.7391 | 0.7207 | 0.0939 |