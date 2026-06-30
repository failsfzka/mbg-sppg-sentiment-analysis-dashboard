import os

# ============================================================
# PATH KONFIGURASI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STYLES_DIR = os.path.join(BASE_DIR, "styles")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

PATH_DATASET_EDA = os.path.join(DATA_DIR, "dataset_eda.csv")
PATH_DATASET_LABELING = os.path.join(DATA_DIR, "dataset_labeling.csv")
PATH_DATASET_INTEGRATED = os.path.join(DATA_DIR, "dataset_integrated.csv")

PATH_SENTIMENT_MODEL = os.path.join(MODEL_DIR, "sentiment_model")
PATH_SARCASM_MODEL = os.path.join(MODEL_DIR, "sarcasm_model")

PATH_LOGO = os.path.join(ASSETS_DIR, "logo.png")
PATH_BANNER = os.path.join(ASSETS_DIR, "banner.png")
PATH_CUSTOM_CSS = os.path.join(STYLES_DIR, "custom.css")

# ============================================================
# IDENTITAS APLIKASI
# ============================================================
APP_TITLE = "Dashboard Analitik MBG-SPPG"
APP_SUBTITLE = "Analisis Sentimen & Deteksi Sarkasme Komentar YouTube Program Makan Bergizi Gratis"
APP_ICON = "🍱"
APP_LAYOUT = "wide"

# ============================================================
# KONFIGURASI MODEL NLP
# ============================================================
MODEL_SENTIMENT_BASE_NAME = "indobenchmark/indobert-base-p1"
MODEL_SARCASM_NAME = "w11wo/xlm-roberta-large-twitter-indonesia-sarcastic"
MAX_SEQUENCE_LENGTH = 64
LABEL_MAPPING_SENTIMENT = {0: "Negatif", 1: "Netral", 2: "Positif"}

# ============================================================
# PALET WARNA TEMA DASHBOARD
# ============================================================
COLOR_PRIMARY = "#1B5E20"
COLOR_SECONDARY = "#FF8F00"
COLOR_BACKGROUND_CARD = "#FFFFFF"

WARNA_SENTIMEN = {
    "Positif": "#2ECC71",
    "Netral": "#F1C40F",
    "Negatif": "#E74C3C",
}

WARNA_SARKASME = {
    "Sarkasme": "#E67E22",
    "Non-Sarkasme": "#3498DB",
}

WARNA_THREAT = {
    "No Threat (Dukungan Positif)": "#2ECC71",
    "Low Threat (Monitoring)": "#F1C40F",
    "Medium Threat (Waspada)": "#E67E22",
    "High Threat (Kritis)": "#E74C3C",
}

DAFTAR_KATEGORI_ISU = [
    "Gizi", "Sosial", "Kualitas Makanan", "Pendidikan",
    "Distribusi", "Anggaran", "Politik",
]

# ============================================================
# DAFTAR HALAMAN DASHBOARD (untuk navigasi & referensi)
# ============================================================
DAFTAR_HALAMAN = [
    {"file": "pages/Home.py", "title": "Home", "icon": "🏠"},
    {"file": "pages/Dataset_Overview.py", "title": "Dataset Overview", "icon": "📊"},
    {"file": "pages/Text_Analytics.py", "title": "Text Analytics", "icon": "📝"},
    {"file": "pages/Model_Analytics.py", "title": "Model Analytics", "icon": "🤖"},
    {"file": "pages/Sentiment_Analytics.py", "title": "Sentiment Analytics", "icon": "😊"},
    {"file": "pages/Sarcasm_Analytics.py", "title": "Sarcasm Analytics", "icon": "😏"},
    {"file": "pages/Policy_Risk_Analytics.py", "title": "Policy Risk Analytics", "icon": "⚠️"},
    {"file": "pages/Dataset_Explorer.py", "title": "Dataset Explorer", "icon": "🔎"},
    {"file": "pages/Live_Prediction.py", "title": "Live Prediction", "icon": "🔮"},
    {"file": "pages/About.py", "title": "About", "icon": "ℹ️"},
]
