"""PART yang layak disiapkan penggantinya lebih awal.

Halaman ini TIDAK menyatakan bahwa sebuah PART akan dibuang. Yang ditampilkan
adalah PART yang risiko rusaknya tinggi DAN - seandainya rusak - kecil
kemungkinannya bisa diperbaiki. Kombinasi itulah yang membuat menyiapkan
pengganti lebih awal masuk akal.
"""

from __future__ import annotations

import streamlit as st

import api_client
import ui

ui.page_setup("Perencanaan Penggantian")
ui.sidebar_status()

st.title("Perencanaan Penggantian")
st.caption(
    "PART dengan risiko kerusakan sedang/tinggi, dan risiko rusak total tinggi."
)

st.warning(
    "Daftar ini "
    "gunanya untuk menyiapkan stok pengganti lebih awal, bukan untuk "
    "memutuskan penggantian."
)

try:
    data = api_client.recommendations(replacement_candidates_only=True, limit=200)
except api_client.ApiError as error:
    st.error(str(error))
    st.stop()

if not data["items"]:
    st.success(
        "Saat ini tidak ada PART yang risiko kerusakan dan risiko rusak "
        "totalnya sama-sama tinggi."
    )
    st.stop()

with st.container(border=True):
    st.metric("Kandidat penggantian", f"{data['total']:,}")
    st.caption(f"Data sampai {data['scored_at']['data_through']}")

ui.priority_table(
    data["items"],
    key="replacement_candidates",
    columns=[
        "rank",
        "item_id",
        "item_type",
        "client",
        "location",
        "installation_age_days",
        "failure_probability_30d",
        "failure_probability_90d",
        "failure_risk_level",
        "scrap_probability",
        "scrap_risk_level",
        "death_probability_30d",
        "priority",
        "recommended_action",
    ],
)

st.caption(
    "*Peluang harus diganti (30H)* = peluang rusak dalam 30 hari dikali "
    "peluang rusak total kalau sampai rusak. Kejadiannya jarang, jadi kolom "
    "ini lebih cocok dipakai untuk mengurutkan perencanaan stok daripada "
    "sebagai pemicu tindakan per PART."
)

st.divider()
st.subheader("Sebaran jenis PART")
types = {}
for item in data["items"]:
    label = item.get("item_type") or "Tidak diketahui"
    types[label] = types.get(label, 0) + 1
st.bar_chart(types, horizontal=True, color="#2563EB")
