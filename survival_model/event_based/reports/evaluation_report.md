# Laporan evaluasi event-based survival (Tahap 6-9)

Tiga lapis - lihat docstring evaluate.py untuk definisi lengkap tiap lapis dan kenapa dipisah. **Lapis 1b (t0-only) adalah angka yang SEBANDING dengan C-index model statis** (survival_model/reports/evaluation_report.md) - Lapis 1 (full landmark) TIDAK sebanding langsung (repeated measures per lifecycle).

## Lapis 1 - native, SEMUA baris landmark (bukan perbandingan apples-to-apples)

### VALIDATION
- **random_survival_forest**: rows=5,540 events=534 C-index(Harrell)=0.8290 IBS=0.04746517713658578
- **cox_ph**: rows=5,540 events=534 C-index(Harrell)=0.7915 IBS=0.05362900428460551

### TEST
- **random_survival_forest**: rows=4,890 events=412 C-index(Harrell)=0.8477 IBS=0.05088881845427292
- **cox_ph**: rows=4,890 events=412 C-index(Harrell)=0.7618 IBS=0.06501409221599042

## Lapis 1b - native, T0-ONLY (satu baris/lifecycle, SEBANDING dengan model statis)

### VALIDATION
- **random_survival_forest**: rows=2,316 events=385 C-index(Harrell)=0.7985 IBS=0.07807458999016595
- **cox_ph**: rows=2,316 events=385 C-index(Harrell)=0.7651 IBS=0.09072887662894609

### TEST
- **random_survival_forest**: rows=2,820 events=370 C-index(Harrell)=0.8105 IBS=0.07870647169642243
- **cox_ph**: rows=2,820 events=370 C-index(Harrell)=0.7405 IBS=0.10281293234218483

## Lapis 2 - perbandingan adil vs classification model (fitur t0-only, populasi TEST classification)

Window 211 hari, kapasitas 200/bulan.
- **random_survival_forest**: PR-AUC=0.1824 ROC-AUC=0.6961 Recall@cap=0.3401 Precision@cap=0.2146 Brier=0.0212
- **cox_ph**: PR-AUC=0.0590 ROC-AUC=0.6832 Recall@cap=0.1160 Precision@cap=0.0732 Brier=0.0233