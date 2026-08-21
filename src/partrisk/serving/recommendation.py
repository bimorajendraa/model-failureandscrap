"""Menerjemahkan kelompok risiko model menjadi tindakan operasional.

BUKAN model, dan sengaja tidak memakai ambang sendiri. Satu-satunya masukan
adalah kelompok risiko (LOW/MEDIUM/HIGH) yang sudah ditetapkan model - ambang
angkanya ditentukan saat training (lihat config.py: FAILURE_HIGH/MEDIUM_
PROBABILITY_THRESHOLD untuk kerusakan, SCRAP_CAPACITY_PER_MONTH untuk risiko
rusak total), bukan dikarang di lapisan ini.

Isinya satu tabel keputusan supaya seluruh aturan terlihat sekaligus dan bisa
diganti tanpa menyentuh kode lain.

Dua sumbu yang dipakai:

    failure_risk_level  - seberapa mungkin PART rusak (dari predict.py)
    scrap_risk_level    - kalau rusak, seberapa mungkin tidak bisa diperbaiki
                          (dari predict_scrap.py; BERSYARAT, bukan peluang
                          PART ini rusak)

Karena risiko scrap bersifat bersyarat, dia tidak pernah menaikkan prioritas
sendirian: PART dengan risiko rusak LOW tetap MONITOR walaupun scrap-nya HIGH.
Yang diubah risiko scrap hanya SIAPKAN PENGGANTI atau tidak.
"""

from __future__ import annotations

RISK_LEVELS = ("LOW", "MEDIUM", "HIGH")

# (risiko kerusakan, risiko scrap) -> (prioritas, tindakan, penjelasan)
_DECISION_TABLE: dict[tuple[str, str], tuple[str, str, str]] = {
    ("HIGH", "HIGH"): (
        "CRITICAL",
        "INSPECT_AND_PREPARE_REPLACEMENT",
        "Risiko kerusakan tinggi dan kecil kemungkinan bisa diperbaiki bila "
        "rusak. Periksa lebih awal dan siapkan unit pengganti.",
    ),
    ("HIGH", "MEDIUM"): (
        "HIGH",
        "PRIORITIZE_INSPECTION",
        "Risiko kerusakan tinggi. Dahulukan pemeriksaan, dan cek ketersediaan "
        "unit pengganti.",
    ),
    ("HIGH", "LOW"): (
        "HIGH",
        "PRIORITIZE_INSPECTION",
        "Risiko kerusakan tinggi, tetapi bila rusak umumnya masih bisa "
        "diperbaiki. Dahulukan pemeriksaan.",
    ),
    ("MEDIUM", "HIGH"): (
        "MEDIUM",
        "SCHEDULE_INSPECTION_AND_REVIEW_STOCK",
        "Risiko kerusakan sedang, tetapi bila rusak kecil kemungkinan bisa "
        "diperbaiki. Jadwalkan pemeriksaan dan tinjau stok pengganti.",
    ),
    ("MEDIUM", "MEDIUM"): (
        "MEDIUM",
        "SCHEDULE_INSPECTION",
        "Risiko kerusakan sedang. Jadwalkan pemeriksaan pada siklus terdekat.",
    ),
    ("MEDIUM", "LOW"): (
        "MEDIUM",
        "SCHEDULE_INSPECTION",
        "Risiko kerusakan sedang. Jadwalkan pemeriksaan pada siklus terdekat.",
    ),
    ("LOW", "HIGH"): (
        "LOW",
        "MONITOR",
        "Risiko kerusakan rendah. Belum perlu tindakan, tetapi bila nanti "
        "rusak kemungkinan besar tidak bisa diperbaiki.",
    ),
    ("LOW", "MEDIUM"): ("LOW", "MONITOR", "Risiko kerusakan rendah. Cukup dipantau."),
    ("LOW", "LOW"): ("LOW", "MONITOR", "Risiko kerusakan rendah. Cukup dipantau."),
}

# Dipakai kalau risiko scrap tidak tersedia (PART belum punya riwayat yang
# bisa dinilai model scrap). Tidak ditebak - hanya sumbu kerusakan yang dipakai.
_FAILURE_ONLY: dict[str, tuple[str, str, str]] = {
    "HIGH": (
        "HIGH",
        "PRIORITIZE_INSPECTION",
        "Risiko kerusakan tinggi. Dahulukan pemeriksaan. Risiko scrap belum "
        "bisa dinilai untuk PART ini.",
    ),
    "MEDIUM": (
        "MEDIUM",
        "SCHEDULE_INSPECTION",
        "Risiko kerusakan sedang. Jadwalkan pemeriksaan. Risiko scrap belum "
        "bisa dinilai untuk PART ini.",
    ),
    "LOW": (
        "LOW",
        "MONITOR",
        "Risiko kerusakan rendah. Cukup dipantau. Risiko scrap belum bisa "
        "dinilai untuk PART ini.",
    ),
}

PRIORITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def recommend(failure_risk_level: str, scrap_risk_level: str | None = None) -> dict:
    """Tindakan operasional untuk satu PART.

    Melempar ValueError kalau kelompok risikonya tidak dikenal - lebih baik
    gagal terang-terangan daripada diam-diam menyarankan MONITOR untuk PART
    yang sebenarnya berisiko tinggi.
    """
    if failure_risk_level not in RISK_LEVELS:
        raise ValueError(f"Kelompok risiko kerusakan tidak dikenal: {failure_risk_level!r}")

    if scrap_risk_level is None:
        priority, action, message = _FAILURE_ONLY[failure_risk_level]
    else:
        if scrap_risk_level not in RISK_LEVELS:
            raise ValueError(f"Kelompok risiko scrap tidak dikenal: {scrap_risk_level!r}")
        priority, action, message = _DECISION_TABLE[
            (failure_risk_level, scrap_risk_level)
        ]

    return {
        "priority": priority,
        "action": action,
        "message": message,
        "based_on": {
            "failure_risk_level": failure_risk_level,
            "scrap_risk_level": scrap_risk_level,
        },
    }


def is_replacement_candidate(failure_risk_level: str, scrap_risk_level: str | None) -> bool:
    """PART yang layak masuk perencanaan penggantian.

    Bukan vonis bahwa PART akan dibuang - hanya kombinasi dua risiko yang
    membuat penyiapan pengganti lebih awal masuk akal.
    """
    return (
        failure_risk_level in ("MEDIUM", "HIGH")
        and scrap_risk_level == "HIGH"
    )
