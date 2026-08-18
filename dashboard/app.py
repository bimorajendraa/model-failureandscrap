"""Halaman utama dashboard: keadaan armada sekilas.

    streamlit run dashboard/app.py

Dashboard hanya bicara ke FastAPI (lihat dashboard/api_client.py) - tidak
pernah ke database dan tidak pernah memuat model sendiri.
"""

from __future__ import annotations

import streamlit as st

import api_client
import ui

ui.page_setup("Overview")
ui.sidebar_status()

st.title("🔧 Predictive Maintenance")
st.caption(
    "Risiko kerusakan dan risiko scrap untuk seluruh PART yang sedang terpasang."
)

try:
    data = api_client.overview(top=15)
except api_client.ApiError as error:
    st.error(str(error))
    st.stop()

summary = data["summary"]
scored = data["scored_at"]

columns = st.columns(4)
columns[0].metric("PART aktif", f"{summary['active_parts']:,}")
columns[1].metric("Risiko tinggi", f"{summary['high_risk_parts']:,}")
columns[2].metric("Risiko sedang", f"{summary['medium_risk_parts']:,}")
columns[3].metric(
    "Kandidat penggantian",
    f"{summary['replacement_candidates']:,}",
    help=(
        "Risiko kerusakan MEDIUM/HIGH sekaligus risiko scrap HIGH. Bukan vonis "
        "bahwa PART akan dibuang."
    ),
)

st.caption(
    f"Data sampai **{scored['data_through']}** · "
    f"model kerusakan **{scored['model_version']['failure']}**, "
    f"model scrap **{scored['model_version']['scrap']}** · "
    f"dihitung {scored['computed_seconds_ago']} detik lalu"
)

st.subheader("PART paling perlu diperhatikan")
ui.priority_table(
    data["top_priority"],
    key="overview_top",
    columns=[
        "rank",
        "item_id",
        "item_type",
        "client",
        "location",
        "failure_probability_30d",
        "failure_probability_60d",
        "failure_risk_level",
        "scrap_risk_level",
        "priority",
        "recommended_action",
    ],
)
ui.probability_caption()

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Sebaran kelompok risiko kerusakan")
    st.bar_chart(
        {
            "HIGH": summary["high_risk_parts"],
            "MEDIUM": summary["medium_risk_parts"],
            "LOW": summary["low_risk_parts"],
        },
        horizontal=True,
    )
with right:
    st.subheader("Sebaran prioritas tindakan")
    st.bar_chart(summary["priority_counts"], horizontal=True)

st.info(
    "**Cara membaca angka di dashboard ini**\n\n"
    "- *Risiko 30H/60H/90H/120H* adalah PELUANG PART rusak dalam jangka waktu "
    "tersebut. Model tidak memperkirakan tanggal kerusakan.\n"
    "- *Risiko scrap* bersifat BERSYARAT: peluang PART tidak bisa diperbaiki "
    "JIKA rusak - bukan peluang PART ini rusak.\n"
    "- Kelompok LOW/MEDIUM/HIGH memakai ambang yang ditetapkan saat training "
    "dari kapasitas kerja tim per bulan."
)
