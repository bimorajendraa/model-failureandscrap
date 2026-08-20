# Laporan evaluasi survival model

## Lapis 1 - native survival (dari t=0=installed_on)

### VALIDATION
- **random_survival_forest**: rows=2,316 events=385 C-index=0.8078 IBS=0.07711840691544726
  - Brier per horizon: 30d=0.0652, 60d=0.0767, 90d=0.0806, 120d=0.0829
  - Time-dependent AUC per horizon: 30d=0.7956, 60d=0.8201, 90d=0.8395, 120d=0.8524
- **cox_ph**: rows=2,316 events=385 C-index=0.7706 IBS=0.09214491962070051
  - Brier per horizon: 30d=0.0754, 60d=0.0927, 90d=0.0959, 120d=0.1003
  - Time-dependent AUC per horizon: 30d=0.7560, 60d=0.7801, 90d=0.8012, 120d=0.8099

### TEST
- **random_survival_forest**: rows=2,820 events=370 C-index=0.8051 IBS=0.08012462289869494
  - Brier per horizon: 30d=0.0810, 60d=0.0818, 90d=0.0795, 120d=0.0771
  - Time-dependent AUC per horizon: 30d=0.8377, 60d=0.8681, 90d=0.8820, 120d=0.9054
- **cox_ph**: rows=2,820 events=370 C-index=0.8149 IBS=0.08675647385680031
  - Brier per horizon: 30d=0.0843, 60d=0.0884, 90d=0.0867, 120d=0.0861
  - Time-dependent AUC per horizon: 30d=0.8564, 60d=0.8787, 90d=0.8929, 120d=0.9087

## Lapis 2 - perbandingan adil vs classification model (populasi TEST classification)

37,923 dari 38,451 baris TEST classification cocok dengan lifecycle survival (211 hari window, kapasitas 200/bulan).

```
model                          PR-AUC  ROC-AUC  Recall@cap  Precision@cap    Brier
random_survival_forest         0.1558   0.7065      0.3142         0.1983   0.0214
cox_ph                         0.1115   0.7063      0.2173         0.1372   0.0221
classification (v2)            0.1607   0.8206      0.3359         0.2154   0.0215
```

Catatan: skor survival di sini pakai fitur baseline INSTALASI (bukan fitur yang di-refresh ke tanggal snapshot seperti classification) - lihat README bagian "Keterbatasan: baseline instalasi vs kondisi sekarang". Perbandingan adil dari sisi horizon/populasi/label, tapi classification model punya keuntungan struktural (fitur lebih segar).