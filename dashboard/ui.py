"""Potongan tampilan yang dipakai beberapa halaman.

Bahasa yang ditampilkan ke pengguna sengaja diterjemahkan dari kode mentah
backend (mis. "PRIORITIZE_INSPECTION", "HIGH") ke istilah biasa yang konsisten
di seluruh dashboard - lihat *_LABELS di bawah. Nilai mentahnya sendiri (yang
dikirim balik ke API sebagai filter, atau dipakai mencocokkan baris) tidak
disentuh, hanya cara menampilkannya.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client

# ---------------------------------------------------------------------------
# Istilah: kode mentah backend -> bahasa biasa, konsisten di seluruh halaman
# ---------------------------------------------------------------------------

RISK_LEVEL_LABELS = {"HIGH": "Tinggi", "MEDIUM": "Sedang", "LOW": "Rendah"}
RISK_LEVEL_COLORS = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}

PRIORITY_LABELS = {
    "CRITICAL": "Sangat mendesak",
    "HIGH": "Mendesak",
    "MEDIUM": "Sedang",
    "LOW": "Rendah",
}
PRIORITY_COLORS = {"CRITICAL": "red", "HIGH": "orange", "MEDIUM": "yellow", "LOW": "green"}

ACTION_LABELS = {
    "INSPECT_AND_PREPARE_REPLACEMENT": "Periksa & siapkan pengganti",
    "PRIORITIZE_INSPECTION": "Segera periksa",
    "SCHEDULE_INSPECTION_AND_REVIEW_STOCK": "Jadwalkan periksa & cek stok",
    "SCHEDULE_INSPECTION": "Jadwalkan pemeriksaan",
    "MONITOR": "Pantau saja",
}

PROBABILITY_COLUMNS = [
    "failure_probability_30d",
    "failure_probability_60d",
    "failure_probability_90d",
    "failure_probability_120d",
    "scrap_probability",
    "death_probability_30d",
]

# Kolom yang isinya kode mentah backend dan perlu diterjemahkan sebelum
# ditampilkan di tabel - lihat _translate_values().
_TRANSLATED_COLUMNS = {
    "failure_risk_level": RISK_LEVEL_LABELS,
    "scrap_risk_level": RISK_LEVEL_LABELS,
    "priority": PRIORITY_LABELS,
    "recommended_action": ACTION_LABELS,
}

COLUMN_LABELS = {
    "rank": "#",
    "item_id": "Item ID",
    "item_type": "Jenis",
    "item_model_code": "Model",
    "client": "Client",
    "location": "Lokasi",
    "installation_age_days": "Umur pasang (hari)",
    "failure_probability_30d": "Risiko 30H",
    "failure_probability_60d": "Risiko 60H",
    "failure_probability_90d": "Risiko 90H",
    "failure_probability_120d": "Risiko 120H",
    "failure_risk_level": "Tingkat risiko rusak",
    "scrap_probability": "Peluang rusak total",
    "scrap_risk_level": "Tingkat risiko rusak total",
    "death_probability_30d": "Peluang harus diganti (30H)",
    "priority": "Prioritas",
    "recommended_action": "Tindakan",
    "active_parts": "PART aktif",
    "high_risk_parts": "Risiko tinggi",
    "medium_risk_parts": "Risiko sedang",
    "replacement_candidates": "Kandidat penggantian",
    "checked": "Status",
    "date": "Tanggal",
    "status": "Status",
    "first_seen": "Pertama tercatat",
    "last_seen": "Terakhir tercatat",
    "events": "Jumlah catatan",
}


def page_setup(title: str, icon: str = "🔧") -> None:
    st.set_page_config(page_title=f"{title} - Predictive Maintenance", page_icon=icon, layout="wide")


def sidebar_status() -> None:
    """Versi model dan kesegaran data - supaya tidak ada yang membaca angka
    lama tanpa sadar."""
    with st.sidebar:
        st.caption(f"API: {api_client.API_BASE_URL}")
        try:
            status = api_client.health()
        except api_client.ApiError as error:
            st.error(str(error))
            return
        versions = status["model_version"]
        st.caption(
            f"Model kerusakan: **{versions.get('failure')}** · "
            f"Model rusak total: **{versions.get('scrap')}**"
        )
        cache = status["batch_cache"]
        if cache["ready"]:
            st.caption(f"Data s/d **{cache['data_through']}**")


def risk_badge(level: str | None) -> str:
    """Lencana kelompok risiko kerusakan/rusak total (HIGH/MEDIUM/LOW)."""
    if level not in RISK_LEVEL_LABELS:
        return "-"
    return f":{RISK_LEVEL_COLORS[level]}-badge[{RISK_LEVEL_LABELS[level]}]"


def priority_badge(priority: str | None) -> str:
    """Lencana prioritas tindakan (CRITICAL/HIGH/MEDIUM/LOW)."""
    if priority not in PRIORITY_LABELS:
        return "-"
    return f":{PRIORITY_COLORS[priority]}-badge[{PRIORITY_LABELS[priority]}]"


def action_label(action: str | None) -> str:
    """Kode tindakan mentah -> kalimat biasa."""
    return ACTION_LABELS.get(action, action or "-")


# RGBA (0-255) untuk titik peta - dipisah jadi konstanta supaya kalimat
# legenda peta dan warna sungguhan tidak pernah berbeda.
MAP_HIGH_COLOR = [192, 57, 43, 200]
MAP_MEDIUM_COLOR = [214, 151, 22, 200]
MAP_LOW_COLOR = [74, 144, 217, 170]


def risk_marker_color(high_risk_parts: int, medium_risk_parts: int) -> list[int]:
    """Warna titik peta menurut kombinasi risiko yang ada di satu lokasi."""
    if high_risk_parts > 0:
        return MAP_HIGH_COLOR
    if medium_risk_parts > 0:
        return MAP_MEDIUM_COLOR
    return MAP_LOW_COLOR


def risk_marker_radius(high_risk_parts: int) -> int:
    """Radius titik peta (meter) - makin besar kalau makin banyak PART
    risiko tinggi di lokasi itu."""
    return 120 + int(high_risk_parts) * 60


def _translate_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Kode mentah (HIGH, PRIORITIZE_INSPECTION, ...) -> bahasa biasa, hanya
    untuk kolom yang memang berisi kode semacam itu."""
    frame = frame.copy()
    for column, labels in _TRANSLATED_COLUMNS.items():
        if column in frame.columns:
            frame[column] = frame[column].map(lambda value: labels.get(value, value))
    return frame


def priority_table(items: list[dict], columns: list[str], key: str = "priority_table") -> None:
    """Tabel daftar PART dengan probabilitas ditampilkan sebagai persen dan
    kode mentah diterjemahkan ke bahasa biasa.

    Klik satu baris untuk memilihnya, lalu tombol "Lihat Detail" muncul di
    bawah tabel dan membawa ke halaman Detail PART untuk PART tersebut.
    `key` harus unik per pemanggilan di halaman yang sama (di sini tiap
    halaman hanya memanggilnya sekali, jadi bawaannya sudah aman).
    """
    if not items:
        st.info("Tidak ada PART yang cocok dengan filter ini.")
        return

    frame = pd.DataFrame(items)
    present = [column for column in columns if column in frame.columns]
    display = _translate_values(frame[present])
    for column in PROBABILITY_COLUMNS:
        if column in display.columns:
            display[column] = display[column].map(api_client.percent)
    if "installation_age_days" in display.columns:
        display["installation_age_days"] = display["installation_age_days"].round(0)

    event = st.dataframe(
        display.rename(columns=COLUMN_LABELS),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )

    rows = event.selection.rows if event and event.selection else []
    if not rows:
        st.caption("Klik satu baris untuk melihat detail PART tersebut.")
        return

    selected_id = str(frame.iloc[rows[0]]["item_id"])
    if st.button(f"🔍 Lihat detail PART {selected_id}", key=f"{key}_open_detail"):
        # session_state dibagi antar halaman dalam satu sesi Streamlit, jadi
        # halaman Detail PART bisa membaca ID ini sebagai nilai awal kotak
        # pencarian tanpa perlu URL/query param.
        st.session_state["detail_item_id"] = selected_id
        st.switch_page("pages/2_Detail_PART.py")


def labeled_table(items: list[dict], columns: list[str], empty_message: str = "") -> None:
    """Tabel sederhana dengan kolom dilabeli lewat COLUMN_LABELS - tanpa
    seleksi baris atau format persen (itu urusan priority_table)."""
    if not items:
        if empty_message:
            st.caption(empty_message)
        return
    frame = pd.DataFrame(items)
    present = [column for column in columns if column in frame.columns]
    display = _translate_values(frame[present])
    st.dataframe(
        display.rename(columns=COLUMN_LABELS),
        width="stretch",
        hide_index=True,
    )


def horizon_metrics(failure: dict) -> None:
    """Empat horizon risiko berdampingan.

    Selalu diberi label "dalam N hari" - model memperkirakan PELUANG, bukan
    tanggal kerusakan, dan tampilan tidak boleh membuatnya terbaca lain.
    """
    columns = st.columns(4)
    for column, days in zip(columns, (30, 60, 90, 120)):
        key = f"failure_probability_{days}d"
        if key in failure:
            column.metric(f"Rusak dalam {days} hari", api_client.percent(failure[key]))


def probability_caption() -> None:
    st.caption(
        "Angka di atas adalah PELUANG kerusakan dalam jangka waktu tersebut - "
        "model tidak memperkirakan tanggal kerusakan."
    )
