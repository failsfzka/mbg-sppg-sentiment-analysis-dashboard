# ============================================================
# METADATA PENELITIAN — Dashboard Analitik MBG-SPPG
# Modul ini bersifat statis (tidak bergantung pada Streamlit)
# agar dapat diuji secara independen dan dipakai ulang oleh
# halaman About maupun halaman lain yang membutuhkan metadata riset.
# ============================================================

INFORMASI_PENELITIAN = {
    "judul": "Analisis Sentimen dan Deteksi Sarkasme pada Komentar YouTube Program Makan Bergizi Gratis (MBG-SPPG)",
    "deskripsi": (
        "Penelitian ini menganalisis opini publik terhadap Program Makan Bergizi Gratis (MBG) "
        "dan Satuan Pelayanan Pemenuhan Gizi (SPPG) melalui komentar YouTube, menggunakan "
        "pendekatan klasifikasi sentimen berbasis fine-tuning IndoBERT serta deteksi sarkasme "
        "lintas bahasa berbasis XLM-RoBERTa, dilengkapi pemetaan isu kebijakan dan tingkat ancaman."
    ),
    "sumber_data": "YouTube Data API v3",
    "objek_penelitian": "Komentar publik pada video-video YouTube terkait Program MBG/SPPG",
}

METODOLOGI_PIPELINE = [
    {"tahap": "1. Pengumpulan Data", "deskripsi": "Scraping komentar YouTube terkait Program MBG/SPPG menggunakan YouTube Data API v3."},
    {"tahap": "2. Preprocessing Teks", "deskripsi": "Case folding, normalisasi kata tidak baku/slang, dan stopword removal."},
    {"tahap": "3. Validasi & Sanitasi Data", "deskripsi": "Pengecekan missing value, string kosong, dan data duplikat."},
    {"tahap": "4. Pengambilan Sampel & Anotasi Manual", "deskripsi": "Sampling 3.000 komentar untuk anotasi manual 3 kelas sentimen (Positif/Netral/Negatif) sebagai data latih & uji model."},
    {"tahap": "5. Tokenisasi & Fine-Tuning IndoBERT", "deskripsi": "Tokenisasi dengan IndoBERT Tokenizer (panjang maksimum 64 token), fine-tuning model indobenchmark/indobert-base-p1 untuk klasifikasi sentimen 3 kelas."},
    {"tahap": "6. Evaluasi Model", "deskripsi": "Evaluasi performa model pada data uji menggunakan classification report (precision, recall, F1-score) dan confusion matrix."},
    {"tahap": "7. Inferensi Sentimen Massal", "deskripsi": "Penerapan model hasil fine-tuning ke seluruh populasi data komentar (bukan hanya sampel anotasi)."},
    {"tahap": "8. Deteksi Sarkasme", "deskripsi": "Klasifikasi sarkasme menggunakan model pretrained XLM-RoBERTa (w11wo/xlm-roberta-large-twitter-indonesia-sarcastic) tanpa pelatihan ulang."},
    {"tahap": "9. Konsolidasi Dataset Akhir", "deskripsi": "Penggabungan hasil prediksi sentimen, sarkasme, kategori isu, dan tingkat ancaman ke dalam satu dataset final berlabel."},
    {"tahap": "10. Analisis Pasca-Pelabelan", "deskripsi": "Pemetaan isu kebijakan berbasis kata kunci dan konstruksi matriks tingkat ancaman kebijakan."},
]

INFORMASI_MODEL_SENTIMEN = {
    "nama_model": "IndoBERT (indobenchmark/indobert-base-p1)",
    "tipe": "Fine-tuned Sequence Classification - 3 kelas (Positif / Netral / Negatif)",
    "jumlah_data_latih": 2400,
    "jumlah_data_uji": 600,
    "panjang_token_maksimum": 64,
    "learning_rate": "1e-5",
    "status_pelatihan": "Fine-tuned (dilatih ulang pada notebook Fine-Tuning IndoBERT)",
}

METRIK_EVALUASI_SENTIMEN = {
    "accuracy": 0.95,
    "macro_precision": 0.95,
    "macro_recall": 0.95,
    "macro_f1": 0.95,
    "weighted_f1": 0.95,
    "detail_per_kelas": [
        {"kelas": "Negatif", "precision": 0.93, "recall": 0.94, "f1_score": 0.93, "support": 196},
        {"kelas": "Netral", "precision": 0.96, "recall": 0.95, "f1_score": 0.95, "support": 256},
        {"kelas": "Positif", "precision": 0.95, "recall": 0.95, "f1_score": 0.95, "support": 148},
    ],
}

INFORMASI_MODEL_SARKASME = {
    "nama_model": "XLM-RoBERTa (w11wo/xlm-roberta-large-twitter-indonesia-sarcastic)",
    "tipe": "Pretrained Binary Classification - Sarkasme / Non-Sarkasme",
    "status_pelatihan": "Pretrained (digunakan langsung tanpa fine-tuning ulang)",
}

DAFTAR_REFERENSI = [
    "Wilie, B., Vincentio, K., Winata, G. I., et al. (2020). IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding. AACL-IJCNLP 2020.",
    "Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.",
    "Conneau, A., Khandelwal, K., Goyal, N., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale (XLM-RoBERTa). ACL 2020.",
    "w11wo (2022). xlm-roberta-large-twitter-indonesia-sarcastic. Hugging Face Model Hub.",
    "Google Developers. YouTube Data API v3 Documentation. https://developers.google.com/youtube/v3",
]


def ringkasan_metrik_dataframe():
    '''Mengembalikan detail metrik evaluasi per kelas dalam bentuk list of dict, siap dikonversi ke DataFrame oleh halaman About.'''
    return METRIK_EVALUASI_SENTIMEN["detail_per_kelas"]
