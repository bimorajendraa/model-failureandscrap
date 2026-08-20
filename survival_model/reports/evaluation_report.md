# Laporan evaluasi survival model

## Lapis 1 - native survival (dari t=0=installed_on)

### VALIDATION
- **random_survival_forest**: rows=2,316 events=385 C-index(Harrell)=0.8114 C-index(Uno/IPCW)=0.8117 IBS=0.07636189083277563
  - Brier per horizon: 30d=0.0627, 60d=0.0755, 90d=0.0805, 120d=0.0836
  - Time-dependent AUC per horizon: 30d=0.7991, 60d=0.8261, 90d=0.8450, 120d=0.8546
- **cox_ph**: rows=2,316 events=385 C-index(Harrell)=0.7819 C-index(Uno/IPCW)=0.7821 IBS=0.0858761371675983
  - Brier per horizon: 30d=0.0684, 60d=0.0853, 90d=0.0899, 120d=0.0965
  - Time-dependent AUC per horizon: 30d=0.7683, 60d=0.7943, 90d=0.8166, 120d=0.8249

### TEST
- **random_survival_forest**: rows=2,820 events=370 C-index(Harrell)=0.8082 C-index(Uno/IPCW)=0.8083 IBS=0.08112253928754527
  - Brier per horizon: 30d=0.0807, 60d=0.0827, 90d=0.0808, 120d=0.0790
  - Time-dependent AUC per horizon: 30d=0.8424, 60d=0.8690, 90d=0.8827, 120d=0.9051
- **cox_ph**: rows=2,820 events=370 C-index(Harrell)=0.7722 C-index(Uno/IPCW)=0.7724 IBS=0.09497357568616568
  - Brier per horizon: 30d=0.0912, 60d=0.0963, 90d=0.0952, 120d=0.0958
  - Time-dependent AUC per horizon: 30d=0.8027, 60d=0.8288, 90d=0.8449, 120d=0.8650

## Lapis 2 - perbandingan adil vs classification model (populasi TEST classification)

37,923 dari 38,451 baris TEST classification cocok dengan lifecycle survival (211 hari window, kapasitas 200/bulan).

```
model                          PR-AUC  ROC-AUC  Recall@cap  Precision@cap    Brier
random_survival_forest         0.1633   0.6871      0.3108         0.1962   0.0214
cox_ph                         0.0939   0.6893      0.1869         0.1180   0.0224
classification (v2)            0.1607   0.8206      0.3359         0.2154   0.0215
```

Catatan: skor survival di sini pakai fitur baseline INSTALASI (bukan fitur yang di-refresh ke tanggal snapshot seperti classification) - lihat README bagian "Keterbatasan: baseline instalasi vs kondisi sekarang". Perbandingan adil dari sisi horizon/populasi/label, tapi classification model punya keuntungan struktural (fitur lebih segar).