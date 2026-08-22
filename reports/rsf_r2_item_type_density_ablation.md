# Fase R2: ablation LOCAL_DENSITY_FEATURES (item_type 90/180d) di RSF

## Hipotesis

`LOCAL_DENSITY_FEATURES` (item_type_at_install, 90/180 hari, log count + rate) adalah
satu-satunya fitur baru yang menang di SEMUA 5 metrik CatBoost v4
(`reports/local_density_experiment.md`). Rencana R2 user meminta feature ini
di-port ke RSF ("samakan sinyal yang sudah menang di CatBoost") dengan ablation wajib.

Catatan penting: kolom-kolom ini SUDAH ditempel ke `landmarks` di
`training/datasets/survival.py` (baris 115) dan ke observations di
`predict/survival.py`/`training/landmark_eval.py` - tapi HANYA untuk memenuhi
kebutuhan kolom `feature_builder.build_features()` (dipakai bersama classification),
BUKAN karena `features/survival/builder.py`'s `FEATURE_COLUMNS` sudah memakainya.
Jadi kolomnya sudah "ada" di pipeline tapi belum benar-benar jadi fitur RSF.

## Metodologi

- Dataset survival dibangun ulang fresh dari DB (`training.datasets.survival.build()`).
- BASELINE = `features.compute_features()` apa adanya (production hari ini, TANPA density).
- CANDIDATE = feature frame yang SAMA + 4 kolom `config.LOCAL_DENSITY_FEATURES`
  (diambil langsung dari `landmarks`, sudah tersedia point-in-time, tidak perlu
  dihitung ulang).
- **Encoder categorical SAMA PERSIS** untuk baseline dan candidate (CATEGORICAL_FEATURES
  tidak berubah) - satu-satunya beda adalah 4 kolom numeric density, lewat parameter
  `numeric_columns` di `features.encode()`.
- RSF compact params PRODUCTION (`COMPACT_RSF_PARAMS`, `random_state=42` tetap) untuk
  kedua varian - satu-satunya variabel yang berubah adalah fitur input.
- Kalibrasi isotonic (`fit_calibrators`) dilatih ulang di VALIDATION masing-masing varian,
  dievaluasi di TEST (holdout, tidak pernah dipakai fit di kedua varian).
- Gate: **R3 milik rencana upgrade RSF** - promote hanya kalau Brier@30d DAN Brier@90d
  TEST tidak memburuk, dan kalibrasi horizon tidak rusak.

## Hasil

| Metrik (TEST) | Baseline | Candidate (+density) | Selisih |
|---|---:|---:|---:|
| C-index | 0.8625 | 0.8550 | -0.0075 |
| IBS | 0.0517 | 0.0518 | +0.0001 |
| Brier@30d | 0.0496 | 0.0499 | **+0.0003 (memburuk)** |
| Brier@60d | 0.0519 | 0.0523 | +0.0004 (memburuk) |
| Brier@90d | 0.0529 | 0.0530 | **+0.0001 (memburuk)** |
| Brier@120d | 0.0510 | 0.0506 | -0.0004 (membaik) |
| AUC@30d | 0.8949 | 0.8928 | -0.0021 |
| AUC@60d | 0.9127 | 0.9110 | -0.0017 |
| AUC@90d | 0.8828 | 0.8806 | -0.0022 |
| AUC@120d | 0.9055 | 0.9027 | -0.0028 |

Kalibrasi bucket tertinggi (reliability, TEST) - relatif tidak berubah dibanding
baseline (bukan perbaikan, bukan kerusakan besar): risk_30d bucket 5 pred 0,1699->0,1645
(aktual 0,2842->0,2911); risk_90d bucket 5 pred 0,3466->0,3399 (aktual 0,4877->0,4938) -
underestimate yang sama besarnya seperti baseline, tidak membaik maupun memburuk
secara berarti.

## Verdict: **DITOLAK**

Gate R3 gagal di dua horizon yang jadi syarat (Brier@30d DAN Brier@90d, keduanya
memburuk, walau tipis). C-index dan AUC juga konsisten memburuk di semua horizon
kecuali Brier@120d. Perbedaannya kecil (~0,0001-0,0075) tapi KONSISTEN arahnya
(hampir semua metrik memburuk) dan reproducible (`random_state=42` tetap di kedua
varian, satu-satunya variabel adalah fitur) - bukan noise run-to-run.

**Fitur TIDAK diwire ke `features/survival/builder.py`.** Kolom density tetap ada
di `landmarks`/observations sebagai artefak pipeline (dibutuhkan
`feature_builder.build_features()`), tapi tetap tidak dipakai FEATURE_COLUMNS RSF -
tidak ada perubahan kode production dari eksperimen ini.

Temuan ini konsisten dengan pola berulang yang sudah didokumentasikan di sisi CatBoost
(`reports/error_analysis_and_age_history_rate.md` dkk): item_type density adalah sinyal
BARU yang genuinely membantu CLASSIFICATION 30-hari (menang di 5 metrik CatBoost v4),
tapi RSF sudah punya cara berbeda melihat "tekanan lokal" lewat kombinasi
`log_days_since_installation` + riwayat/armada landmark-nya sendiri - menambahkan sinyal
yang sama lewat jalur berbeda tidak otomatis membantu model yang arsitekturnya berbeda.
Fase R2 dianggap SELESAI dengan hasil "tidak ada yang diwire" - bukan kegagalan proses,
melainkan hasil sah dari ablation yang jujur (sama semangat dengan 9 eksperimen CatBoost
sebelumnya yang ditolak).
