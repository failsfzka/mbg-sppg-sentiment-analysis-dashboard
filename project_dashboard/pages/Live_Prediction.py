import streamlit as st
import plotly.express as px
import pandas as pd

from config import WARNA_SENTIMEN, WARNA_SARKASME, WARNA_THREAT
from utils.predictor import (
    muat_model_sentimen, muat_model_sarkasme,
    prediksi_sentimen, prediksi_sarkasme,
    petakan_isu_strategis, evaluasi_tingkat_ancaman,
)
from components.kpi_card import render_badge


def render_grafik_probabilitas(distribusi_prob, peta_warna, judul):
    '''Menampilkan grafik batang horizontal distribusi probabilitas per kelas.'''
    df_prob = pd.DataFrame(list(distribusi_prob.items()), columns=["kelas", "probabilitas"])
    df_prob["persentase"] = (df_prob["probabilitas"] * 100).round(2)
    fig = px.bar(
        df_prob, x="persentase", y="kelas", orientation="h",
        color="kelas", color_discrete_map=peta_warna,
        title=judul, text=df_prob["persentase"].astype(str) + "%",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="Probabilitas (%)", yaxis_title="", xaxis_range=[0, 115])
    st.plotly_chart(fig, use_container_width=True)


def render_hasil_prediksi(label_sentimen, conf_sentimen, prob_sentimen,
                          label_sarkasme, conf_sarkasme, prob_sarkasme,
                          kategori_isu, tingkat_ancaman):
    '''Menampilkan 4 kotak hasil prediksi + grafik probabilitas.'''
    st.markdown("#### 📋 Hasil Prediksi")

    kol1, kol2, kol3, kol4 = st.columns(4)
    warna_s = {"Positif": "#2ECC71", "Netral": "#F1C40F", "Negatif": "#E74C3C"}
    warna_k = {"Sarkasme": "#E67E22", "Non-Sarkasme": "#3498DB"}
    warna_t = {"No Threat (Dukungan Positif)": "#2ECC71", "Low Threat (Monitoring)": "#F1C40F",
                "Medium Threat (Waspada)": "#E67E22", "High Threat (Kritis)": "#E74C3C"}

    with kol1:
        st.markdown("**Sentimen**")
        badge_s = render_badge(label_sentimen, warna_s.get(label_sentimen, "#999"))
        st.markdown(badge_s, unsafe_allow_html=True)
        st.caption(f"Confidence: {conf_sentimen:.2%}")
    with kol2:
        st.markdown("**Sarkasme**")
        badge_k = render_badge(label_sarkasme, warna_k.get(label_sarkasme, "#999"))
        st.markdown(badge_k, unsafe_allow_html=True)
        st.caption(f"Confidence: {conf_sarkasme:.2%}")
    with kol3:
        st.markdown("**Isu Kebijakan**")
        badge_i = render_badge(kategori_isu, "#1B5E20")
        st.markdown(badge_i, unsafe_allow_html=True)
        st.caption("Berbasis kata kunci")
    with kol4:
        st.markdown("**Tingkat Ancaman**")
        badge_t = render_badge(tingkat_ancaman, warna_t.get(tingkat_ancaman, "#999"))
        st.markdown(badge_t, unsafe_allow_html=True)
        st.caption("Rule-based analysis")

    st.divider()
    kol_prob_a, kol_prob_b = st.columns(2)
    with kol_prob_a:
        render_grafik_probabilitas(prob_sentimen, WARNA_SENTIMEN, "Distribusi Probabilitas Sentimen")
    with kol_prob_b:
        render_grafik_probabilitas(prob_sarkasme, WARNA_SARKASME, "Distribusi Probabilitas Sarkasme")


# ============================================================
# RENDER HALAMAN LIVE PREDICTION
# ============================================================
st.markdown("<div class='page-header-title'>🔮 Live Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-header-subtitle'>Prediksi sentimen, sarkasme, isu, dan ancaman pada komentar baru secara real-time</div>",
    unsafe_allow_html=True,
)

st.info(
    "Halaman ini menggunakan model IndoBERT (sentimen) dan XLM-RoBERTa (sarkasme). "
    "Pemuatan model pertama kali membutuhkan beberapa menit. "
    "Untuk prediksi sentimen yang akurat, simpan model hasil fine-tuning ke folder **models/sentiment_model/**."
)

with st.expander("Cara Menyimpan Model Fine-Tuned ke Folder models/sentiment_model/", expanded=False):
    st.code(
        'trainer.save_model("project_dashboard/models/sentiment_model/")\n'
        'tokenizer.save_pretrained("project_dashboard/models/sentiment_model/")',
        language="python",
    )
    st.code(
        'from transformers import pipeline\n'
        'model_sarkasme.save_pretrained("project_dashboard/models/sarcasm_model/")\n'
        'tokenizer_sarkasme.save_pretrained("project_dashboard/models/sarcasm_model/")',
        language="python",
    )

teks_input_pengguna = st.text_area(
    "Masukkan Komentar untuk Dianalisis",
    height=140,
    placeholder='Contoh: "Program MBG bagus tapi implementasinya masih kacau, korupsi di mana-mana!"',
)

if st.button("🔮 Analisis Komentar", type="primary", use_container_width=True):
    if not teks_input_pengguna.strip():
        st.warning("Harap masukkan teks komentar terlebih dahulu.")
    else:
        with st.spinner("Memuat model dan menganalisis komentar..."):
            tokenizer_sentimen_aktif, model_sentimen_aktif = muat_model_sentimen()
            tokenizer_sarkasme_aktif, model_sarkasme_aktif = muat_model_sarkasme()

            hasil_sentimen, conf_sentimen, prob_sentimen = prediksi_sentimen(
                teks_input_pengguna, tokenizer_sentimen_aktif, model_sentimen_aktif
            )
            hasil_sarkasme, conf_sarkasme, prob_sarkasme = prediksi_sarkasme(
                teks_input_pengguna, tokenizer_sarkasme_aktif, model_sarkasme_aktif
            )
            isu = petakan_isu_strategis(teks_input_pengguna)
            ancaman = evaluasi_tingkat_ancaman(hasil_sentimen, hasil_sarkasme, isu)

        render_hasil_prediksi(
            hasil_sentimen, conf_sentimen, prob_sentimen,
            hasil_sarkasme, conf_sarkasme, prob_sarkasme,
            isu, ancaman,
        )

        with st.expander("Detail Teks yang Dianalisis"):
            st.text(teks_input_pengguna)
