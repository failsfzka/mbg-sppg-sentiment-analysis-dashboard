import pandas as pd
import streamlit as st
import plotly.express as px

from utils.system_info import (
    INFORMASI_PENELITIAN,
    METODOLOGI_PIPELINE,
    INFORMASI_MODEL_SENTIMEN,
    INFORMASI_MODEL_SARKASME,
    METRIK_EVALUASI_SENTIMEN,
    DAFTAR_REFERENSI,
    ringkasan_metrik_dataframe,
)
from components.kpi_card import render_kpi_row


def render_profil_penelitian():
    '''Menampilkan profil, judul, dan deskripsi singkat penelitian.'''
    st.markdown("### 🔬 Profil Penelitian")
    st.markdown(
        f"<div style='background:#F4F8F4;border-left:5px solid #1B5E20;padding:1.2rem 1.5rem;"
        f"border-radius:8px;margin-bottom:1rem;'>"
        f"<h4 style='margin:0;color:#1B5E20;'>{INFORMASI_PENELITIAN['judul']}</h4>"
        f"<p style='margin-top:0.7rem;color:#4a4a4a;'>{INFORMASI_PENELITIAN['deskripsi']}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    kolom_kiri, kolom_kanan = st.columns(2)
    with kolom_kiri:
        st.markdown(f"**Sumber Data:** {INFORMASI_PENELITIAN['sumber_data']}")
    with kolom_kanan:
        st.markdown(f"**Objek Penelitian:** {INFORMASI_PENELITIAN['objek_penelitian']}")


def render_informasi_dataset():
    '''Menampilkan kartu ringkasan dua dataset yang digunakan dalam penelitian.'''
    st.markdown("### 🗃️ Dataset Penelitian")
    kolom_kiri, kolom_kanan = st.columns(2)

    with kolom_kiri:
        st.markdown(
            "<div class='kpi-card'>"
            "<div class='kpi-icon'>📊</div>"
            "<div class='kpi-value'>Dataset EDA</div>"
            "<div class='kpi-label'>dataset_siap_eda_mbg_sppg.csv</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Dataset hasil scraping YouTube Data API + preprocessing teks "
            "(case folding, normalisasi, stopword removal). "
            "Berisi metadata video & komentar. Digunakan pada halaman Dataset Overview & Text Analytics."
        )

    with kolom_kanan:
        st.markdown(
            "<div class='kpi-card'>"
            "<div class='kpi-icon'>🏷️</div>"
            "<div class='kpi-value'>Dataset Final Labeling</div>"
            "<div class='kpi-label'>dataset_final_mbg_sppg.csv — 69.079 baris</div>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Dataset hasil inferensi model IndoBERT (sentimen) & XLM-RoBERTa (sarkasme), "
            "dilengkapi kategori isu kebijakan (7 kelas) dan tingkat ancaman (4 kelas). "
            "Digunakan pada halaman Model Analytics, Sentiment, Sarcasm, Policy Risk, "
            "Dataset Explorer, dan Live Prediction."
        )


def render_metodologi():
    '''Menampilkan 10 tahap metodologi penelitian dalam format timeline/expander.'''
    st.markdown("### 🔄 Metodologi Penelitian")
    st.caption("Penelitian mengikuti pipeline 10 tahap dari pengumpulan data hingga analisis pasca-pelabelan.")

    for tahap_info in METODOLOGI_PIPELINE:
        with st.expander(tahap_info["tahap"], expanded=False):
            st.write(tahap_info["deskripsi"])


def render_performa_model():
    '''Menampilkan metrik evaluasi model IndoBERT dan informasi model sarkasme.'''
    st.markdown("### 🤖 Performa Model")

    # ---- Model Sentimen: IndoBERT ----
    st.markdown("#### Model Sentimen — IndoBERT Fine-Tuned")
    st.markdown(
        f"**Nama Model:** `{INFORMASI_MODEL_SENTIMEN['nama_model']}`  \n"
        f"**Tipe:** {INFORMASI_MODEL_SENTIMEN['tipe']}  \n"
        f"**Data Latih:** {INFORMASI_MODEL_SENTIMEN['jumlah_data_latih']:,} komentar  \n"
        f"**Data Uji:** {INFORMASI_MODEL_SENTIMEN['jumlah_data_uji']:,} komentar  \n"
        f"**Panjang Token Maks:** {INFORMASI_MODEL_SENTIMEN['panjang_token_maksimum']}  \n"
        f"**Learning Rate:** {INFORMASI_MODEL_SENTIMEN['learning_rate']}  \n"
        f"**Status:** {INFORMASI_MODEL_SENTIMEN['status_pelatihan']}"
    )

    # --- 4 Kartu Metrik Utama ---
    daftar_metrik_kpi = [
        {"icon": "🎯", "label": "Accuracy",        "value": f"{METRIK_EVALUASI_SENTIMEN['accuracy']:.2%}"},
        {"icon": "📐", "label": "Macro Precision", "value": f"{METRIK_EVALUASI_SENTIMEN['macro_precision']:.2%}"},
        {"icon": "📏", "label": "Macro Recall",    "value": f"{METRIK_EVALUASI_SENTIMEN['macro_recall']:.2%}"},
        {"icon": "⚡", "label": "Macro F1-Score",  "value": f"{METRIK_EVALUASI_SENTIMEN['macro_f1']:.2%}"},
    ]
    render_kpi_row(daftar_metrik_kpi, jumlah_kolom=4)

    st.markdown("")

    # --- Tabel Detail Per Kelas ---
    st.markdown("##### Tabel Evaluasi Detail Per Kelas Sentimen")
    df_metrik = pd.DataFrame(ringkasan_metrik_dataframe())
    df_metrik["precision"] = df_metrik["precision"].apply(lambda x: f"{x:.2%}")
    df_metrik["recall"] = df_metrik["recall"].apply(lambda x: f"{x:.2%}")
    df_metrik["f1_score"] = df_metrik["f1_score"].apply(lambda x: f"{x:.2%}")
    df_metrik = df_metrik.rename(columns={
        "kelas": "Kelas", "precision": "Precision",
        "recall": "Recall", "f1_score": "F1-Score", "support": "Support"
    })
    st.dataframe(df_metrik, use_container_width=True, hide_index=True)

    # --- Grafik Batang Perbandingan Metrik Per Kelas ---
    df_grafik = pd.DataFrame(ringkasan_metrik_dataframe())
    df_grafik_long = df_grafik.melt(
        id_vars=["kelas"],
        value_vars=["precision", "recall", "f1_score"],
        var_name="metrik", value_name="skor"
    )
    df_grafik_long["metrik"] = df_grafik_long["metrik"].map(
        {"precision": "Precision", "recall": "Recall", "f1_score": "F1-Score"}
    )
    fig_metrik = px.bar(
        df_grafik_long, x="kelas", y="skor", color="metrik", barmode="group",
        range_y=[0.85, 1.0],
        color_discrete_sequence=["#1B5E20", "#FF8F00", "#3498DB"],
        title="Perbandingan Metrik Evaluasi Per Kelas Sentimen",
        labels={"kelas": "Kelas Sentimen", "skor": "Skor", "metrik": "Metrik"},
    )
    fig_metrik.update_layout(yaxis_tickformat=".0%")
    st.plotly_chart(fig_metrik, use_container_width=True)

    st.divider()

    # ---- Model Sarkasme: XLM-RoBERTa ----
    st.markdown("#### Model Sarkasme — XLM-RoBERTa Pretrained")
    st.markdown(
        f"**Nama Model:** `{INFORMASI_MODEL_SARKASME['nama_model']}`  \n"
        f"**Tipe:** {INFORMASI_MODEL_SARKASME['tipe']}  \n"
        f"**Status:** {INFORMASI_MODEL_SARKASME['status_pelatihan']}"
    )
    st.info(
        "Model sarkasme digunakan langsung tanpa fine-tuning ulang. "
        "Model ini telah dilatih sebelumnya pada data Twitter berbahasa Indonesia "
        "untuk mendeteksi sarkasme secara biner (Sarkasme / Non-Sarkasme)."
    )


def render_referensi():
    '''Menampilkan daftar referensi ilmiah yang mendasari penelitian.'''
    st.markdown("### 📚 Referensi")
    for i, referensi in enumerate(DAFTAR_REFERENSI, start=1):
        st.markdown(f"{i}. {referensi}")


# ============================================================
# RENDER HALAMAN ABOUT
# ============================================================
st.markdown("<div class='page-header-title'>ℹ️ About</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-header-subtitle'>Profil penelitian, dataset, metodologi, dan performa model yang mendasari dashboard ini</div>",
    unsafe_allow_html=True,
)

render_profil_penelitian()
st.divider()
render_informasi_dataset()
st.divider()
render_metodologi()
st.divider()
render_performa_model()
st.divider()
render_referensi()
