import streamlit as st
import plotly.express as px

from config import APP_TITLE, APP_SUBTITLE, WARNA_SENTIMEN, DAFTAR_HALAMAN
from utils.data_loader import load_labeling_data, load_integrated_data
from components.kpi_card import render_kpi_row


def render_header():
    '''Menampilkan judul, subjudul, dan garis pemisah halaman Home.'''
    st.markdown(f"<div class='page-header-title'>🏠 {APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-header-subtitle'>{APP_SUBTITLE}</div>", unsafe_allow_html=True)


def render_sidebar_info():
    '''Menampilkan informasi ringkas penelitian pada sidebar.'''
    with st.sidebar:
        st.markdown("### 🍱 Tentang Dashboard")
        st.write(
            "Dashboard ini menyajikan hasil analisis sentimen dan deteksi sarkasme pada "
            "komentar YouTube terkait Program Makan Bergizi Gratis (MBG-SPPG), menggunakan "
            "model fine-tuned IndoBERT dan XLM-RoBERTa."
        )
        st.divider()
        st.caption("Gunakan menu navigasi di atas untuk menjelajahi setiap halaman analitik.")


def hitung_statistik_ringkasan(df_labeling_data, df_integrated_data):
    '''Menghitung kumpulan statistik ringkasan (KPI) dari dataset labeling & dataset terintegrasi.'''
    total_komentar = len(df_labeling_data)
    persen_positif = (df_labeling_data["sentiment_prediction"] == "Positif").mean() * 100 if total_komentar else 0
    persen_sarkasme = (df_labeling_data["sarcasm_prediction"] == "Sarkasme").mean() * 100 if total_komentar else 0
    jumlah_ancaman_tinggi = int((df_labeling_data["threat_level"] == "High Threat (Kritis)").sum())

    if not df_integrated_data.empty and "video_title" in df_integrated_data.columns:
        jumlah_video = df_integrated_data["video_title"].nunique()
    else:
        jumlah_video = None

    return {
        "total_komentar": total_komentar,
        "persen_positif": persen_positif,
        "persen_sarkasme": persen_sarkasme,
        "jumlah_ancaman_tinggi": jumlah_ancaman_tinggi,
        "jumlah_video": jumlah_video,
    }


def render_kpi_section(statistik):
    '''Menampilkan baris kartu KPI utama pada halaman Home.'''
    nilai_video = f"{statistik['jumlah_video']:,}" if statistik["jumlah_video"] is not None else "-"
    daftar_kpi = [
        {"icon": "💬", "label": "Total Komentar", "value": f"{statistik['total_komentar']:,}"},
        {"icon": "🎬", "label": "Total Video Dianalisis", "value": nilai_video},
        {"icon": "😊", "label": "Sentimen Positif", "value": f"{statistik['persen_positif']:.1f}%"},
        {"icon": "🚨", "label": "Komentar Risiko Tinggi", "value": f"{statistik['jumlah_ancaman_tinggi']:,}"},
    ]
    render_kpi_row(daftar_kpi, jumlah_kolom=4)


def render_grafik_ringkasan(df_labeling_data):
    '''Menampilkan grafik donut sentimen dan grafik batang isu kebijakan menggunakan Plotly.'''
    kolom_kiri, kolom_kanan = st.columns(2)

    with kolom_kiri:
        st.markdown("##### 📊 Distribusi Sentimen Komentar")
        distribusi_sentimen = df_labeling_data["sentiment_prediction"].value_counts().reset_index()
        distribusi_sentimen.columns = ["sentimen", "jumlah"]
        fig_donut = px.pie(
            distribusi_sentimen, names="sentimen", values="jumlah", hole=0.45,
            color="sentimen", color_discrete_map=WARNA_SENTIMEN,
        )
        fig_donut.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
        st.plotly_chart(fig_donut, use_container_width=True)

    with kolom_kanan:
        st.markdown("##### 🗂️ Distribusi Kategori Isu Kebijakan")
        distribusi_isu = df_labeling_data["issue_category"].value_counts().reset_index()
        distribusi_isu.columns = ["isu", "jumlah"]
        fig_bar = px.bar(
            distribusi_isu.sort_values("jumlah"), x="jumlah", y="isu", orientation="h",
            color_discrete_sequence=["#1B5E20"],
        )
        fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title="", xaxis_title="Jumlah Komentar")
        st.plotly_chart(fig_bar, use_container_width=True)


def render_navigasi_cepat():
    '''Menampilkan grid tautan navigasi cepat ke seluruh halaman dashboard lainnya.'''
    st.markdown("##### 🧭 Navigasi Cepat")
    halaman_lainnya = [h for h in DAFTAR_HALAMAN if h["title"] != "Home"]
    jumlah_kolom = 3
    kolom = st.columns(jumlah_kolom)
    for index, halaman in enumerate(halaman_lainnya):
        with kolom[index % jumlah_kolom]:
            st.page_link(halaman["file"], label=halaman["title"], icon=halaman["icon"])


# ============================================================
# RENDER HALAMAN HOME
# ============================================================
render_header()
render_sidebar_info()

df_labeling_aktif = load_labeling_data()
df_integrated_aktif = load_integrated_data()

if df_labeling_aktif.empty:
    st.warning("Dataset labeling belum tersedia. Pastikan berkas data/dataset_labeling.csv telah disiapkan.")
else:
    statistik_ringkasan = hitung_statistik_ringkasan(df_labeling_aktif, df_integrated_aktif)
    render_kpi_section(statistik_ringkasan)
    st.divider()
    render_grafik_ringkasan(df_labeling_aktif)
    st.divider()
    render_navigasi_cepat()
