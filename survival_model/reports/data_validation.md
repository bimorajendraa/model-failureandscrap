# Validasi dataset survival

## Jumlah lifecycle
Cohort (is_initial_model_cohort, durasi positif): 23,927

Eligible untuk survival (lolos aturan censoring per-split): 20,116 (84.1% dari cohort)

## Lifecycle yang di-exclude, per split & alasan siklus berakhir
           split                   cycle_end_reason  count
EXCLUDED_TOO_OLD                            FAILURE    915
EXCLUDED_TOO_OLD REINSTALL_WITHOUT_RECORDED_FAILURE    348
EXCLUDED_TOO_OLD         RIGHT_CENSORED_AT_DATA_END   2146
            TEST REINSTALL_WITHOUT_RECORDED_FAILURE     28
           TRAIN REINSTALL_WITHOUT_RECORDED_FAILURE    342
      VALIDATION REINSTALL_WITHOUT_RECORDED_FAILURE     32

## Event vs censored per split
            censored (0)  event (1)
split                              
TEST                2450        370
TRAIN              11939       3041
VALIDATION          1931        385

## Distribusi duration_days
min=1.0  p25=198.0  median=492.0  p75=1898.0  p99=3402.0  max=4011.0

## Cek integritas (semua harus 0)
duration_days <= 0: 0

installation_cycle_id duplikat: 0

failure_onset_on < installed_on (pada lifecycle event=1): 0

installed_on di masa depan (> sekarang): 0

## Tipe PART yang cuma muncul di VALIDATION (tidak pernah di TRAIN)
20 tipe dari 58 tipe di VALIDATION (catatan: part_model_category sudah mengelompokkan tipe bersupport rendah jadi satu kategori bersama - lihat config.MIN_PART_MODEL_SUPPORT; OneHotEncoder di train.py juga diberi handle_unknown='ignore' sebagai pengaman kedua)

## Tipe PART yang cuma muncul di TEST (tidak pernah di TRAIN)
12 tipe dari 56 tipe di TEST (catatan: part_model_category sudah mengelompokkan tipe bersupport rendah jadi satu kategori bersama - lihat config.MIN_PART_MODEL_SUPPORT; OneHotEncoder di train.py juga diberi handle_unknown='ignore' sebagai pengaman kedua)

## PART dengan lifecycle di lebih dari satu split
1,234 item (7.5% dari item unik) punya >1 lifecycle yang jatuh di split berbeda (mis. cycle 1 di TRAIN, cycle 3 di TEST). Bukan leakage temporal (urutan waktu tetap terjaga), tapi potensi model 'mengenali' identitas item lintas split lewat fitur previous_cycle_lifetime_mean/has_previous_cycle. Didokumentasikan sebagai keterbatasan, tidak diperbaiki dengan grouped split (lihat README).

## Base rate event (failure) per split - untuk melihat pergeseran
split
TEST          0.131206
TRAIN         0.203004
VALIDATION    0.166235
