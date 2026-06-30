import pandas as pd
import streamlit as st
import plotly.express as px

from utils.data_loader import load_eda_data
from components.kpi_card import render_kpi_row


def kolom_tersedia(dataframe_target, nama_kolom):
    '''Mengembalikan True apabila nama_kolom tersedia pada dataframe_target.'''
    return nama_kolom in dataframe_target.columns


def hitung_statistik_utama(df_eda_data):
    '''Menghitung 4 statistik utama: jumlah video, jumlah komentar, jumlah like, dan rentang waktu.'''
    jumlah_video = df_eda_data["video_title"].nunique() if kolom_tersedia(df_eda_data, "video_title") else None
    jumlah_komentar = len(df_eda_data)
    jumlah_like = int(df_eda_data["likes"].sum()) if kolom_tersedia(df_eda_data, "likes") else None

    if kolom_tersedia(df_eda_data, "published_at"):
        tanggal = pd.to_datetime(df_eda_data["published_at"], errors="coerce")
        rentang_waktu = f"{tanggal.min().date()} s.d. {tanggal.max().date()}" if tanggal.notna().any() else "-"
    else:
        rentang_waktu = "-"

    return {
        "jumlah_video": jumlah_video,
        "jumlah_komentar": jumlah_komentar,
        "jumlah_like": jumlah_like,
        "rentang_waktu": rentang_waktu,
    }


def render_kpi_overview(statistik):
    '''Menampilkan kartu KPI 4 statistik utama Dataset Overview.'''
    daftar_kpi = [
        {"icon": "🎬", "label": "Jumlah Video", "value": f"{statistik['jumlah_video']:,}" if statistik["jumlah_video"] is not None else "-"},
        {"icon": "💬", "label": "Jumlah Komentar", "value": f"{statistik['jumlah_komentar']:,}"},
        {"icon": "👍", "label": "Jumlah Like", "value": f"{statistik['jumlah_like']:,}" if statistik["jumlah_like"] is not None else "-"},
        {"icon": "📅", "label": "Rentang Waktu", "value": statistik["rentang_waktu"]},
    ]
    render_kpi_row(daftar_kpi, jumlah_kolom=4)


def render_tab_pratinjau(df_eda_data):
    '''Menampilkan pratinjau dataframe, dimensi, struktur tipe data, dan missing value.'''
    st.markdown("##### Pratinjau Dataset EDA")
    st.dataframe(df_eda_data.head(20), use_container_width=True)

    kolom_kiri, kolom_kanan = st.columns(2)
    with kolom_kiri:
        st.metric("Jumlah Baris", f"{df_eda_data.shape[0]:,}")
    with kolom_kanan:
        st.metric("Jumlah Kolom", df_eda_data.shape[1])

    st.markdown("##### Struktur Tipe Data & Missing Value")
    tabel_struktur = pd.DataFrame({
        "Kolom": df_eda_data.columns,
        "Tipe Data": df_eda_data.dtypes.astype(str).values,
        "Missing Value": df_eda_data.isnull().sum().values,
    })
    st.dataframe(tabel_struktur, use_container_width=True, hide_index=True)


def render_tab_statistik(df_eda_data):
    '''Menampilkan statistik deskriptif numerik Dataset EDA.'''
    kolom_numerik = df_eda_data.select_dtypes(include=["number"]).columns.tolist()
    if kolom_numerik:
        st.markdown(f"##### Statistik Deskriptif Kolom Numerik: {', '.join(kolom_numerik)}")
        st.dataframe(df_eda_data[kolom_numerik].describe(), use_container_width=True)
    else:
        st.info("Tidak ditemukan kolom numerik pada Dataset EDA.")


def render_tab_distribusi(df_eda_data):
    '''Menampilkan grafik distribusi query, channel/komentator, dan video.'''
    if kolom_tersedia(df_eda_data, "query_source"):
        st.markdown("##### Distribusi Query Pencarian")
        distribusi_query = df_eda_data["query_source"].value_counts().reset_index()
        distribusi_query.columns = ["query", "jumlah"]
        fig_query = px.bar(distribusi_query, x="query", y="jumlah", color_discrete_sequence=["#1B5E20"])
        st.plotly_chart(fig_query, use_container_width=True)

    kolom_kiri, kolom_kanan = st.columns(2)
    with kolom_kiri:
        if kolom_tersedia(df_eda_data, "author"):
            st.markdown("##### Top 10 Channel/Komentator")
            distribusi_channel = df_eda_data["author"].value_counts().head(10).reset_index()
            distribusi_channel.columns = ["author", "jumlah"]
            fig_channel = px.bar(
                distribusi_channel.sort_values("jumlah"), x="jumlah", y="author", orientation="h",
                color_discrete_sequence=["#FF8F00"],
            )
            st.plotly_chart(fig_channel, use_container_width=True)

    with kolom_kanan:
        if kolom_tersedia(df_eda_data, "video_title"):
            st.markdown("##### Top 10 Video Terbanyak Dikomentari")
            distribusi_video = df_eda_data["video_title"].value_counts().head(10).reset_index()
            distribusi_video.columns = ["video", "jumlah"]
            fig_video = px.bar(
                distribusi_video.sort_values("jumlah"), x="jumlah", y="video", orientation="h",
                color_discrete_sequence=["#3498DB"],
            )
            st.plotly_chart(fig_video, use_container_width=True)


# ============================================================
# RENDER HALAMAN DATASET OVERVIEW
# ============================================================
st.markdown("<div class='page-header-title'>📊 Dataset Overview</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-header-subtitle'>Gambaran umum Dataset EDA hasil scraping & preprocessing komentar YouTube</div>",
    unsafe_allow_html=True,
)

df_eda_halaman = load_eda_data()

if df_eda_halaman.empty:
    st.warning("Dataset EDA belum tersedia. Pastikan berkas data/dataset_eda.csv telah disiapkan.")
else:
    statistik_utama = hitung_statistik_utama(df_eda_halaman)
    render_kpi_overview(statistik_utama)
    st.divider()

    tab_pratinjau, tab_statistik, tab_distribusi = st.tabs(["🔍 Pratinjau & Struktur", "📈 Statistik Numerik", "📊 Distribusi"])
    with tab_pratinjau:
        render_tab_pratinjau(df_eda_halaman)
    with tab_statistik:
        render_tab_statistik(df_eda_halaman)
    with tab_distribusi:
        render_tab_distribusi(df_eda_halaman)
