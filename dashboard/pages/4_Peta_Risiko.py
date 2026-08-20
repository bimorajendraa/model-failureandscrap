"""Peta sebaran risiko menurut lokasi.

Database ini hanya punya NAMA lokasi ("STASIUN JUANDA"), bukan koordinat GPS.
Titik di peta ini datang dari OpenStreetMap (dicari otomatis di sisi API,
lihat api/services/geocoding_service.py), disaring ketat: hanya nama berpola
stasiun publik yang dicoba, dan hasilnya harus jatuh di dalam kotak
Jabodetabek. Lokasi yang tidak lolos TIDAK dipasang pin - supaya peta ini
tidak pernah menunjukkan tempat yang salah - tetapi tetap dilaporkan di tabel
di bawah peta supaya PART berisiko tinggi di lokasi itu tidak hilang dari
pandangan hanya karena belum ada titiknya.
"""

from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st

import api_client
import ui

ui.page_setup("Peta Risiko", icon="🗺️")
ui.sidebar_status()

st.title("Peta Risiko")
st.caption("Di mana saja PART berisiko tinggi sekarang terpasang.")

if st.button("🔄 Coba cari koordinat lagi", help="Untuk lokasi yang belum punya titik."):
    api_client.locations_map.clear()

try:
    data = api_client.locations_map(resolve=True, budget_seconds=60)
except api_client.ApiError as error:
    st.error(str(error))
    st.stop()

resolved = data["resolved"]
unresolved = data["unresolved"]
checked = [item for item in unresolved if item["checked"]]
pending = [item for item in unresolved if not item["checked"]]

with st.container(border=True):
    columns = st.columns(4)
    columns[0].metric("Lokasi aktif", f"{len(resolved) + len(unresolved):,}")
    columns[1].metric("Sudah ada titik di peta", f"{len(resolved):,}")
    columns[2].metric("Belum ketemu koordinatnya", f"{len(checked):,}")
    columns[3].metric("Belum sempat dicoba", f"{len(pending):,}")

if pending:
    st.info(
        f"{len(pending)} lokasi belum sempat dicari koordinatnya (anggaran waktu "
        "habis). Klik \"Coba cari koordinat lagi\" di atas untuk melanjutkan - "
        "lokasi yang sudah pernah dicoba tidak diulang."
    )

if resolved:
    st.divider()
    st.subheader("Peta")
    st.caption(
        "Klik satu titik untuk melihat rinciannya, lalu tombol untuk membuka "
        "daftar PART di lokasi itu."
    )

    frame = pd.DataFrame(resolved)
    frame["color"] = frame.apply(
        lambda row: ui.risk_marker_color(row["high_risk_parts"], row["medium_risk_parts"]),
        axis=1,
    )
    frame["radius"] = frame["high_risk_parts"].map(ui.risk_marker_radius)

    layer = pdk.Layer(
        "ScatterplotLayer",
        id="risk-points",
        data=frame,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius="radius",
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(
        latitude=float(frame["lat"].mean()),
        longitude=float(frame["lon"].mean()),
        zoom=10,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": (
                "<b>{location}</b><br/>"
                "PART aktif: {active_parts}<br/>"
                "Risiko tinggi: {high_risk_parts}<br/>"
                "Risiko sedang: {medium_risk_parts}"
            )
        },
    )
    event = st.pydeck_chart(
        deck, on_select="rerun", selection_mode="single-object", key="risk_map"
    )

    selected = []
    if event and event.selection:
        selected = event.selection.objects.get("risk-points", [])

    if selected:
        point = selected[0]
        with st.container(border=True):
            st.markdown(f"### 📍 {point['location']}")
            detail = st.columns(4)
            detail[0].metric("PART aktif", f"{point['active_parts']:,}")
            detail[1].metric("Risiko tinggi", f"{point['high_risk_parts']:,}")
            detail[2].metric("Risiko sedang", f"{point['medium_risk_parts']:,}")
            detail[3].metric("Kandidat penggantian", f"{point['replacement_candidates']:,}")
            if st.button(f"📋 Lihat daftar PART di {point['location']}"):
                st.session_state["priority_location_filter"] = point["location"]
                st.switch_page("pages/1_Prioritas_Perawatan.py")
    else:
        st.caption("Belum ada titik dipilih.")

    st.caption(
        ":red-badge[merah] = ada PART risiko tinggi · "
        ":orange-badge[oranye] = ada PART risiko sedang, tidak ada yang tinggi · "
        ":blue-badge[biru] = tidak ada PART risiko tinggi/sedang di lokasi ini. "
        "Ukuran titik mengikuti jumlah PART risiko tinggi."
    )

    with st.expander(f"Tabel lokasi di peta ({len(resolved)})"):
        sorted_resolved = sorted(resolved, key=lambda item: -item["high_risk_parts"])
        ui.labeled_table(
            sorted_resolved,
            columns=["location", "active_parts", "high_risk_parts", "medium_risk_parts", "replacement_candidates"],
        )
else:
    st.info("Belum ada lokasi yang berhasil dipetakan.")

if unresolved:
    st.divider()
    st.subheader("Lokasi yang belum punya titik di peta")
    st.caption(
        "Tetap diurutkan berdasarkan risiko supaya tidak terlewat hanya karena "
        "belum ada koordinatnya."
    )
    status_label = {True: "Sudah dicoba, tidak ketemu", False: "Belum dicoba"}
    sorted_unresolved = [
        {**item, "checked": status_label[item["checked"]]}
        for item in sorted(unresolved, key=lambda item: -item["high_risk_parts"])
    ]
    ui.labeled_table(
        sorted_unresolved,
        columns=["location", "active_parts", "high_risk_parts", "medium_risk_parts", "checked"],
    )
