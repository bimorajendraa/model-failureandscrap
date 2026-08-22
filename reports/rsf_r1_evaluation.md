# Fase R1 — Evaluasi RSF dengan metrik yang lebih mudah dipahami

## Hipotesis / tujuan

Fase R1 tidak mengejar C-index lebih tinggi. Tiga pertanyaan yang mau dijawab:

1. Seberapa bisa dipercaya angka "sisa umur" (median / days-to-90%)?
2. Seberapa terkalibrasi `risk_30d`…`risk_120d` setelah isotonic + cummax (Fase R1 item a,
   commit `be83a03`)?
3. Kapan `median_days_to_failure` kosong, dan apakah UI sudah menangani itu dengan jujur?

Item (a) — aktivasi kalibrasi ke field advisory — sudah selesai dan sudah diverifikasi lewat
`test_survival_calibrated_risk_monoton_naik` + smoke test manual sebelum laporan ini ditulis.
Laporan ini adalah item (b) dan (c).

## Metodologi

- Split TEST landmark dibangun ulang dari DB fresh (`training.datasets.survival.build()`) -
  4.890 baris, 412 di antaranya `event_observed=True`.
- Median prediksi per baris: `curves.median_survival_time()` dari
  `curves.survival_curve_arrays()` pada model + encoder yang sama dengan production
  (`predict_survival.load_model()`).
- Error median dihitung HANYA pada baris `event_observed=True` DAN median terisi - untuk baris
  censored, "error" tidak bisa dihitung jujur karena durasi asli belum diketahui.
- % null armada aktif diukur dengan jalur serving sungguhan (`predict_survival.score_batch()`
  pada `feature_builder.current_observations(cycles)`, populasi PART aktif hari ini), bukan
  populasi TEST - supaya cocok dengan apa yang benar-benar dilihat user di dashboard.
- Kalibrasi risk@30d/90d diukur dengan reliability table: TEST landmark rows dikelompokkan
  jadi 5 bucket berdasarkan risk_30d/90d TERKALIBRASI, dibandingkan rata-rata prediksi vs
  rata-rata label biner aktual (`event & duration<=horizon` -> 1, `duration>=horizon` -> 0,
  censored-sebelum-horizon dikeluarkan) - populasi dan aturan label PERSIS sama dengan yang
  dipakai `fit_calibrators()` saat training, hanya split-nya TEST bukan VAL, supaya jadi ukuran
  generalisasi kalibrasi yang jujur.

## Hasil

### 1. Brier & AUC per horizon (dari metadata, populasi TEST landmark)

| Horizon | Brier | AUC waktu-bergantung |
|---|---:|---:|
| 30d | 0.0496 | 0.8949 |
| 60d | 0.0519 | 0.9127 |
| 90d | 0.0529 | 0.8828 |
| 120d | 0.0510 | 0.9055 |

C-index (Harrell) = 0.8625, C-index (Uno/IPCW) = 0.8623, Integrated Brier Score = 0.0517.
Brier < 0.05 di semua horizon secara agregat terlihat baik - tapi angka ini dihitung dari
S(t) MENTAH (belum dikalibrasi), dan agregat menutupi masalah di sub-populasi (lihat #3).

### 2. Median sisa umur - jarang terisi, dan ketika terisi, jauh dari akurat

| Populasi | % median = null |
|---|---:|
| TEST landmark (4.890 baris) | 79.3% |
| **Armada aktif sekarang (16.877 PART)** | **94.7%** |

Untuk 257 baris TEST yang berkejadian nyata (`event_observed=True`) DAN median terisi (257/412
= 62%):

| Metrik error `|median prediksi - durasi aktual|` | Nilai |
|---|---:|
| Median | 439 hari |
| Mean | 751.9 hari |
| Persentil 25 / 75 | 250 / 1.078 hari |

**Kesimpulan: `median_days_to_failure` tidak layak dijadikan angka andalan.** Bukan cuma jarang
terisi (94.7% None di armada aktif - S(t) memang jarang turun sampai separuh dalam rentang
follow-up training, ini bawaan struktural data, bukan bug), tapi bahkan ketika terisi, meleset
median 439 hari dari durasi aktual. Ini konsisten dengan ekspektasi realistis yang sudah ditulis
di rencana awal ("median days lebih stabil/lebih sering terisi" TIDAK terbukti - justru makin
jarang terisi di populasi aktif dibanding TEST landmark, karena armada aktif condong lebih muda).

`days_until_survival_90pct` jauh lebih sering terisi: 32.9% null di armada aktif (vs 94.7% untuk
median) - field inilah yang lebih pantas jadi headline "sisa umur", bukan median.

### 3. Kalibrasi risk_30d / risk_90d - underestimate di bucket risiko tertinggi

Reliability table, TEST landmark, risk TERKALIBRASI (isotonic + cummax), 5 bucket sama besar:

**risk_30d** (n_label=4.382):

| Bucket (rendah→tinggi) | n | pred mentah | pred terkalibrasi | rate aktual |
|---|---:|---:|---:|---:|
| 1 | 877 | 0.0024 | 0.0000 | 0.0000 |
| 2 | 877 | 0.0037 | 0.0000 | 0.0034 |
| 3 | 876 | 0.0077 | 0.0021 | 0.0023 |
| 4 | 876 | 0.0483 | 0.0396 | 0.0696 |
| **5 (tertinggi)** | 876 | 0.1578 | **0.1699** | **0.2842** |

**risk_90d** (n_label=2.430):

| Bucket (rendah→tinggi) | n | pred mentah | pred terkalibrasi | rate aktual |
|---|---:|---:|---:|---:|
| 1 | 486 | 0.0048 | 0.0000 | 0.0041 |
| 2 | 486 | 0.0041 | 0.0000 | 0.0041 |
| 3 | 486 | 0.0305 | 0.0160 | 0.0391 |
| 4 | 486 | 0.1028 | 0.1266 | 0.2366 |
| **5 (tertinggi)** | 486 | 0.2460 | **0.3466** | **0.4877** |

Tiga bucket risiko rendah terkalibrasi cukup baik (dekat nol, cocok dengan aktual). **Dua bucket
teratas underestimate cukup jauh** - persis di bagian yang paling menentukan keputusan (PART mana
yang benar-benar berisiko): risk_30d bucket teratas bilang 17%, aktualnya 28%; risk_90d bucket
teratas bilang 35%, aktualnya 49%.

Penyebab paling mungkin: kalibrator isotonic dilatih di split VALIDATION, dievaluasi di sini pada
TEST - pergeseran antar split (bukan bug kalibrasi itu sendiri) + bucket VAL kemungkinan
lebih sedikit sampel ekstrem-tinggi dibanding TEST. Brier agregat (#1) tidak menangkap ini karena
bucket 5 hanya ~20% populasi dan errornya "hanya" sekitar 0.11-0.14 absolut - cukup kecil untuk
tenggelam di rata-rata keseluruhan, tapi cukup besar untuk menyesatkan keputusan di level PART
individual berisiko tinggi.

### 4. Fallback UI saat median = null - SUDAH ada, terverifikasi

`dashboard/ui.py::survival_advisory()` (baris 257-275) sudah menampilkan
`days_until_survival_90pct` berdampingan dengan median, plus caption
`median_days_to_failure_basis` saat median kosong menjelaskan alasannya. Docstring dan komentar
di kode itu bahkan sudah mencatat estimasi "~5% PART aktif" untuk median - cocok persis dengan
angka terukur 5.3% (100% - 94.7%) di laporan ini. **Tidak ada perubahan kode diperlukan untuk
item (c) - sudah memenuhi.**

## Kesimpulan & rekomendasi untuk R2

1. **Jangan jadikan median sebagai metrik sukses R2.** Ia terlalu jarang terisi dan terlalu
   tidak presisi bahkan saat terisi - ini keterbatasan struktural (follow-up belum cukup panjang
   untuk sebagian besar PART), bukan sesuatu yang bisa diperbaiki lewat fitur baru.
   `days_until_survival_90pct` adalah field yang lebih realistis untuk terus ditingkatkan.
2. **Kalibrasi di bucket risiko tertinggi adalah temuan baru yang layak ditindaklanjuti** - bisa
   jadi kandidat kecil untuk R1 lanjutan (mis. re-fit calibrator dengan data VAL+TEST gabungan,
   atau lebih banyak bucket di ekor atas) SEBELUM masuk R2 (LOCAL_DENSITY_FEATURES), karena ini
   murni masalah kalibrasi - tidak menambah fitur.
3. Brier/AUC/C-index agregat sehat dan konsisten dengan ekspektasi rencana awal (~0,86)- tidak
   ada regresi mengejutkan dari sisi itu.
