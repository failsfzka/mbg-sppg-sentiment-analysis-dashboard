import os
import pandas as pd
import streamlit as st

from config import OUTPUTS_DIR
from utils.data_loader import load_integrated_data, load_labeling_data


def kolom_tersedia(dataframe_target, nama_kolom):
    '''Mengembalikan True apabila nama_kolom tersedia pada dataframe_target.'''
    return nama_kolom in dataframe_target.columns


def muat_dataset_explorer():
    '''Memuat dataset terbaik yang tersedia untuk halaman Explorer: Integrated > Labeling.'''
    df_integrated = load_integrated_data()
    if not df_integrated.empty:
        return df_integrated, "Integrated Dataset"
    df_labeling = load_labeling_data()
    return df_labeling, "Dataset Final Labeling (fallback)"


def render_sidebar_filter(df_explorer):
    '''Menampilkan seluruh widget filter pada sidebar dan mengembalikan nilai yang dipilih pengguna.'''
    with st.sidebar:
        st.markdown("### 🔎 Filter Dataset")

        kata_kunci = st.text_input("Cari dalam teks komentar", placeholder="Ketik kata kunci...")

        filter_sentimen = st.multiselect(
            "Filter Sentimen",
            options=sorted(df_explorer["sentiment_prediction"].dropna().unique().tolist())
            if kolom_tersedia(df_explorer, "sentiment_prediction") else [],
            default=[],
        )

        filter_sarkasme = st.multiselect(
            "Filter Sarkasme",
            options=sorted(df_explorer["sarcasm_prediction"].dropna().unique().tolist())
            if kolom_tersedia(df_explorer, "sarcasm_prediction") else [],
            default=[],
        )

        filter_isu = st.multiselect(
            "Filter Isu Kebijakan",
            options=sorted(df_explorer["issue_category"].dropna().unique().tolist())
            if kolom_tersedia(df_explorer, "issue_category") else [],
            default=[],
        )

        filter_threat = st.multiselect(
            "Filter Tingkat Ancaman",
            options=df_explorer["threat_level"].dropna().unique().tolist()
            if kolom_tersedia(df_explorer, "threat_level") else [],
            default=[],
        )

        filter_query = st.multiselect(
            "Filter Query Pencarian",
            options=sorted(df_explorer["query_source"].dropna().unique().tolist())
            if kolom_tersedia(df_explorer, "query_source") else [],
            default=[],
        )

        filter_tanggal_aktif = False
        tanggal_mulai = None
        tanggal_akhir = None
        if kolom_tersedia(df_explorer, "published_at"):
            tanggal_series = pd.to_datetime(df_explorer["published_at"], errors="coerce").dropna()
            if tanggal_series.notna().any():
                filter_tanggal_aktif = True
                tgl_min = tanggal_series.min().date()
                tgl_maks = tanggal_series.max().date()
                tanggal_mulai = st.date_input("Tanggal Mulai", value=tgl_min, min_value=tgl_min, max_value=tgl_maks)
                tanggal_akhir = st.date_input("Tanggal Akhir", value=tgl_maks, min_value=tgl_min, max_value=tgl_maks)

        return {
            "kata_kunci": kata_kunci,
            "sentimen": filter_sentimen,
            "sarkasme": filter_sarkasme,
            "isu": filter_isu,
            "threat": filter_threat,
            "query": filter_query,
            "tanggal_aktif": filter_tanggal_aktif,
            "tanggal_mulai": tanggal_mulai,
            "tanggal_akhir": tanggal_akhir,
        }


def terapkan_filter(df_explorer, pilihan_filter):
    '''Menerapkan semua filter secara berantai (AND) pada dataset Explorer.'''
    df_hasil = df_explorer.copy()

    if pilihan_filter["kata_kunci"]:
        kolom_teks_explorer = "text_clean" if kolom_tersedia(df_hasil, "text_clean") else df_hasil.columns[0]
        df_hasil = df_hasil[
            df_hasil[kolom_teks_explorer].astype(str).str.contains(
                pilihan_filter["kata_kunci"], case=False, na=False, regex=False
            )
        ]

    if pilihan_filter["sentimen"] and kolom_tersedia(df_hasil, "sentiment_prediction"):
        df_hasil = df_hasil[df_hasil["sentiment_prediction"].isin(pilihan_filter["sentimen"])]

    if pilihan_filter["sarkasme"] and kolom_tersedia(df_hasil, "sarcasm_prediction"):
        df_hasil = df_hasil[df_hasil["sarcasm_prediction"].isin(pilihan_filter["sarkasme"])]

    if pilihan_filter["isu"] and kolom_tersedia(df_hasil, "issue_category"):
        df_hasil = df_hasil[df_hasil["issue_category"].isin(pilihan_filter["isu"])]

    if pilihan_filter["threat"] and kolom_tersedia(df_hasil, "threat_level"):
        df_hasil = df_hasil[df_hasil["threat_level"].isin(pilihan_filter["threat"])]

    if pilihan_filter["query"] and kolom_tersedia(df_hasil, "query_source"):
        df_hasil = df_hasil[df_hasil["query_source"].isin(pilihan_filter["query"])]

    if pilihan_filter["tanggal_aktif"] and kolom_tersedia(df_hasil, "published_at"):
        tanggal_series = pd.to_datetime(df_hasil["published_at"], errors="coerce")
        df_hasil = df_hasil[
            (tanggal_series.dt.date >= pilihan_filter["tanggal_mulai"]) &
            (tanggal_series.dt.date <= pilihan_filter["tanggal_akhir"])
        ]

    return df_hasil


def render_tombol_download(df_hasil_filter):
    '''Menampilkan tombol unduh CSV dari hasil filter.'''
    csv_bytes = df_hasil_filter.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Hasil Filter sebagai CSV",
        data=csv_bytes,
        file_name="hasil_filter_dataset_mbg_sppg.csv",
        mime="text/csv",
    )


# ============================================================
# RENDER HALAMAN DATASET EXPLORER
# ============================================================
st.markdown("<div class='page-header-title'>🔎 Dataset Explorer</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-header-subtitle'>Jelajahi dan filter dataset secara interaktif — gunakan panel filter di sidebar kiri</div>",
    unsafe_allow_html=True,
)

df_explorer_halaman, nama_dataset_explorer = muat_dataset_explorer()

if df_explorer_halaman.empty:
    st.warning("Dataset tidak tersedia untuk Explorer.")
else:
    st.caption(f"Dataset yang digunakan: **{nama_dataset_explorer}** | {len(df_explorer_halaman):,} baris total")

    pilihan_filter_pengguna = render_sidebar_filter(df_explorer_halaman)
    df_terfilter = terapkan_filter(df_explorer_halaman, pilihan_filter_pengguna)

    kolom_info, kolom_unduh = st.columns([3, 1])
    with kolom_info:
        st.metric("Baris Ditemukan", f"{len(df_terfilter):,}", delta=f"{len(df_terfilter) - len(df_explorer_halaman):,}")
    with kolom_unduh:
        if not df_terfilter.empty:
            render_tombol_download(df_terfilter)

    if df_terfilter.empty:
        st.info("Tidak ada data yang sesuai dengan kombinasi filter yang dipilih. Coba ubah atau hapus beberapa filter.")
    else:
        st.dataframe(df_terfilter.reset_index(drop=True), use_container_width=True, height=520)
