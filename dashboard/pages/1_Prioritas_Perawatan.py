"""Daftar prioritas perawatan, bisa disaring."""

from __future__ import annotations

import streamlit as st

import api_client
import ui

ui.page_setup("Prioritas Perawatan")
ui.sidebar_status()

st.title("Prioritas Perawatan")
st.caption("Seluruh PART aktif, terurut dari yang paling berisiko rusak.")

try:
    available = api_client.filters()
except api_client.ApiError as error:
    st.error(str(error))
    st.stop()


def _choice(label: str, options: list[str], default: str | None = None) -> str | None:
    all_options = ["Semua", *options]
    index = all_options.index(default) if default in all_options else 0
    value = st.selectbox(label, all_options, index=index)
    return None if value == "Semua" else value


# Diisi otomatis kalau datang dari peta risiko (klik titik lokasi -> "Lihat
# daftar PART di sini"). Sekali pakai, sama seperti detail_item_id di halaman
# Detail PART.
default_location = st.session_state.pop("priority_location_filter", None)

search = st.text_input(
    "Cari Item ID",
    placeholder="sebagian ID sudah cukup, mis. 0112011",
    help="Cocok sebagian - tidak perlu mengetik ID lengkap.",
).strip()

row = st.columns(5)
with row[0]:
    risk = _choice("Kelompok risiko", available["risk_levels"])
with row[1]:
    item_type = _choice("Jenis PART", available["item_types"])
with row[2]:
    client = _choice("Client", available["clients"])
with row[3]:
    location = _choice("Lokasi", available["locations"], default=default_location)
with row[4]:
    limit = st.number_input("Jumlah baris", min_value=10, max_value=500, value=50, step=10)

try:
    data = api_client.recommendations(
        risk=risk, item_type=item_type, client=client, location=location,
        search=search or None, limit=int(limit),
    )
except api_client.ApiError as error:
    st.error(str(error))
    st.stop()

st.caption(
    f"**{data['total']:,}** PART cocok dengan filter · menampilkan "
    f"{data['returned']:,} teratas · data sampai {data['scored_at']['data_through']}"
)

ui.priority_table(
    data["items"],
    key="priority_list",
    columns=[
        "rank",
        "item_id",
        "item_type",
        "item_model_code",
        "client",
        "location",
        "installation_age_days",
        "failure_probability_30d",
        "failure_probability_60d",
        "failure_probability_90d",
        "failure_probability_120d",
        "failure_risk_level",
        "scrap_probability",
        "scrap_risk_level",
        "priority",
        "recommended_action",
    ],
)
ui.probability_caption()

st.caption(
    "Buka halaman **Detail PART** dan masukkan Item ID untuk melihat faktor "
    "risiko di balik angkanya."
)
