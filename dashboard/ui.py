"""Potongan tampilan yang dipakai beberapa halaman."""

from __future__ import annotations

import pandas as pd
import streamlit as st

import api_client

RISK_BADGES = {
    "CRITICAL": ":red[**CRITICAL**]",
    "HIGH": ":red[**HIGH**]",
    "MEDIUM": ":orange[**MEDIUM**]",
    "LOW": ":green[**LOW**]",
}

PROBABILITY_COLUMNS = [
    "failure_probability_30d",
    "failure_probability_60d",
    "failure_probability_90d",
    "failure_probability_120d",
    "scrap_probability",
    "death_probability_30d",
]

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
    "failure_risk_level": "Kelompok risiko",
    "scrap_probability": "Risiko scrap",
    "scrap_risk_level": "Kelompok scrap",
    "death_probability_30d": "Risiko mati 30H",
    "priority": "Prioritas",
    "recommended_action": "Tindakan",
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
            f"Model scrap: **{versions.get('scrap')}**"
        )
        cache = status["batch_cache"]
        if cache["ready"]:
            st.caption(f"Data s/d **{cache['data_through']}**")


def risk_badge(level: str | None) -> str:
    return RISK_BADGES.get(level, "-")


# RGBA (0-255) untuk titik peta - dipisah jadi konstanta supaya kalimat
# legenda peta dan warna sungguhan tidak pernah berbeda.
MAP_HIGH_COLOR = [192, 57, 43, 200]
MAP_MEDIUM_COLOR = [39, 143, 245, 0.8]
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


def priority_table(items: list[dict], columns: list[str], key: str = "priority_table") -> None:
    """Tabel daftar PART dengan probabilitas ditampilkan sebagai persen.

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
    display = frame[present].copy()
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
