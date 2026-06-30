import re
import os
import torch
import streamlit as st
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (
    PATH_SENTIMENT_MODEL,
    MODEL_SENTIMENT_BASE_NAME,
    MODEL_SARCASM_NAME,
    MAX_SEQUENCE_LENGTH,
    LABEL_MAPPING_SENTIMENT,
    DAFTAR_KATEGORI_ISU,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PREPROCESSING TEKS
# ============================================================
def preprocessing_teks(teks_input):
    '''
    Menerapkan tahap preprocessing dasar yang konsisten dengan notebook Fine-Tuning:
    lowercase, normalisasi spasi, hapus karakter non-alfanumerik berlebih.
    '''
    teks = teks_input.strip().lower()
    teks = re.sub(r"\s+", " ", teks)
    teks = re.sub(r"[^\w\s]", " ", teks)
    teks = re.sub(r"\s+", " ", teks).strip()
    return teks


# ============================================================
# PEMUATAN MODEL SENTIMEN (INDOBERT — FINE-TUNED)
# ============================================================
@st.cache_resource(show_spinner="Memuat model sentimen IndoBERT...")
def muat_model_sentimen():
    '''
    Memuat model IndoBERT hasil fine-tuning dari folder models/sentiment_model/.
    Jika folder belum berisi model, menggunakan model dasar indobenchmark/indobert-base-p1
    sebagai fallback agar halaman tidak crash saat model belum disimpan.
    '''
    if os.path.exists(PATH_SENTIMENT_MODEL) and os.listdir(PATH_SENTIMENT_MODEL):
        sumber_model = PATH_SENTIMENT_MODEL
    else:
        sumber_model = MODEL_SENTIMENT_BASE_NAME
        st.warning(
            f"Model sentimen fine-tuned belum ditemukan di '{PATH_SENTIMENT_MODEL}'. "
            f"Menggunakan model dasar '{MODEL_SENTIMENT_BASE_NAME}' sebagai fallback. "
            "Untuk hasil akurat, simpan model hasil fine-tuning ke folder 'models/sentiment_model/'."
        )
    tokenizer = AutoTokenizer.from_pretrained(sumber_model)
    model = AutoModelForSequenceClassification.from_pretrained(sumber_model, num_labels=3)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# ============================================================
# PEMUATAN MODEL SARKASME (XLM-ROBERTA — PRETRAINED)
# ============================================================
@st.cache_resource(show_spinner="Memuat model sarkasme XLM-RoBERTa...")
def muat_model_sarkasme():
    '''Memuat model XLM-RoBERTa pretrained untuk deteksi sarkasme dari Hugging Face Hub.'''
    sumber_model = (
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sarcasm_model")
        if os.path.exists(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sarcasm_model")
        ) and os.listdir(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "sarcasm_model")
        )
        else MODEL_SARCASM_NAME
    )
    tokenizer = AutoTokenizer.from_pretrained(sumber_model)
    model = AutoModelForSequenceClassification.from_pretrained(sumber_model)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


# ============================================================
# FUNGSI INFERENSI SENTIMEN
# ============================================================
def prediksi_sentimen(teks_input, tokenizer_sentimen, model_sentimen):
    '''
    Memprediksi sentimen (Positif/Netral/Negatif) dari teks input menggunakan model IndoBERT.
    Mengembalikan label prediksi, confidence score, dan distribusi probabilitas seluruh kelas.
    '''
    teks_bersih = preprocessing_teks(teks_input)
    input_encoded = tokenizer_sentimen(
        teks_bersih, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH,
        padding="max_length", truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        keluaran = model_sentimen(**input_encoded)

    probabilitas = torch.softmax(keluaran.logits, dim=1).cpu().numpy()[0]
    indeks_prediksi = int(np.argmax(probabilitas))
    label_prediksi = LABEL_MAPPING_SENTIMENT.get(indeks_prediksi, "Tidak Diketahui")
    confidence = float(probabilitas[indeks_prediksi])

    distribusi_prob = {
        LABEL_MAPPING_SENTIMENT.get(i, f"Kelas {i}"): float(p)
        for i, p in enumerate(probabilitas)
    }
    return label_prediksi, confidence, distribusi_prob


# ============================================================
# FUNGSI INFERENSI SARKASME
# ============================================================
def prediksi_sarkasme(teks_input, tokenizer_sarkasme, model_sarkasme):
    '''
    Memprediksi sarkasme (Sarkasme/Non-Sarkasme) menggunakan model XLM-RoBERTa.
    Mengembalikan label prediksi, confidence score, dan distribusi probabilitas.
    '''
    teks_bersih = preprocessing_teks(teks_input)
    input_encoded = tokenizer_sarkasme(
        teks_bersih, return_tensors="pt", max_length=MAX_SEQUENCE_LENGTH,
        padding="max_length", truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        keluaran = model_sarkasme(**input_encoded)

    probabilitas = torch.softmax(keluaran.logits, dim=1).cpu().numpy()[0]
    label_id2label = model_sarkasme.config.id2label

    indeks_prediksi = int(np.argmax(probabilitas))
    raw_label = label_id2label.get(indeks_prediksi, "non-sarcastic")
    label_prediksi = "Sarkasme" if "sarcastic" in str(raw_label).lower() and "non" not in str(raw_label).lower() else "Non-Sarkasme"
    confidence = float(probabilitas[indeks_prediksi])

    distribusi_prob = {}
    for i, p in enumerate(probabilitas):
        raw = label_id2label.get(i, f"kelas_{i}")
        label_bersih = "Sarkasme" if "sarcastic" in str(raw).lower() and "non" not in str(raw).lower() else "Non-Sarkasme"
        distribusi_prob[label_bersih] = max(distribusi_prob.get(label_bersih, 0.0), float(p))

    return label_prediksi, confidence, distribusi_prob


# ============================================================
# PEMETAAN ISU DAN ANCAMAN (RULE-BASED)
# ============================================================
KATA_KUNCI_ISU = {
    "Gizi": ["gizi", "nutrisi", "bergizi", "makan", "protein", "kalori", "vitamin", "sehat", "menu"],
    "Sosial": ["sosial", "masyarakat", "warga", "anak", "siswa", "sekolah", "murid", "keluarga", "rakyat"],
    "Kualitas Makanan": ["kualitas", "basi", "busuk", "kotor", "tidak layak", "enak", "lezat", "higienis"],
    "Pendidikan": ["pendidikan", "belajar", "prestasi", "guru", "kurikulum", "akademik", "pelajaran"],
    "Distribusi": ["distribusi", "pengiriman", "merata", "terlambat", "logistik", "suplai", "kirim"],
    "Anggaran": ["anggaran", "dana", "biaya", "korupsi", "APBN", "uang", "efisiensi", "pemborosan"],
    "Politik": ["politik", "pemerintah", "kebijakan", "presiden", "menteri", "partai", "kampanye", "pemilu"],
}


def petakan_isu_strategis(teks_input):
    '''Memetakan teks ke kategori isu kebijakan paling relevan berdasarkan kata kunci.'''
    teks_lower = teks_input.lower()
    skor_per_isu = {}
    for isu, daftar_kata in KATA_KUNCI_ISU.items():
        skor_per_isu[isu] = sum(1 for kata in daftar_kata if kata in teks_lower)
    isu_terbaik = max(skor_per_isu, key=skor_per_isu.get)
    return isu_terbaik if skor_per_isu[isu_terbaik] > 0 else "Gizi"


def evaluasi_tingkat_ancaman(label_sentimen, label_sarkasme, isu):
    '''Menentukan tingkat ancaman berdasarkan kombinasi sentimen, sarkasme, dan isu.'''
    isu_sensitif = {"Anggaran", "Politik", "Kualitas Makanan"}
    isu_netral = {"Gizi", "Pendidikan", "Distribusi", "Sosial"}

    if label_sentimen == "Positif" and label_sarkasme == "Non-Sarkasme":
        return "No Threat (Dukungan Positif)"
    if label_sentimen == "Negatif" and label_sarkasme == "Sarkasme" and isu in isu_sensitif:
        return "High Threat (Kritis)"
    if label_sentimen == "Negatif" and (label_sarkasme == "Sarkasme" or isu in isu_sensitif):
        return "Medium Threat (Waspada)"
    if label_sentimen == "Negatif" and label_sarkasme == "Non-Sarkasme" and isu in isu_netral:
        return "Low Threat (Monitoring)"
    return "Low Threat (Monitoring)"
