# ─────────────────────────────────────────────
# app/app.py — Aplikasi Web Data Mining
# Jalankan: streamlit run app/app.py
# ─────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import gdown
import os

if not os.path.exists("model/rf_model.pkl"):
    gdown.download(
        "https://drive.google.com/1poyk0eid_PBIOFjGIOpxnmWP66qYLDsj/view?usp=sharing",
        "model/rf_model.pkl",
        quiet=False
    )
# ── Konfigurasi halaman
st.set_page_config(
    page_title="Analisis Pembelajaran Matematika",
    page_icon="🎓",
    layout="wide"
)

# ── Load model
@st.cache_resource
def load_models():
    with open("model/kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("model/rf_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("model/label_encoders.pkl", "rb") as f:
        le_dict = pickle.load(f)
    with open("model/label_encoder_target.pkl", "rb") as f:
        le_target = pickle.load(f)
    with open("model/metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    return kmeans, rf, scaler, le_dict, le_target, meta

kmeans, rf_model, scaler, le_dict, le_target, meta = load_models()

# ── Navigasi sidebar
st.sidebar.title("🎓 Data Mining UAS")
st.sidebar.markdown("**Analisis Pembelajaran Matematika**")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Home", "📊 Dataset Overview", "🤖 Prediksi & Analisis", "📈 Visualisasi", "ℹ️ About"]
)

# ══════════════════════════════════
# HALAMAN 1: HOME
# ══════════════════════════════════
if page == "🏠 Home":
    st.title("🎓 Analisis Pembelajaran Matematika di Pendidikan Tinggi")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **Dataset**
UCI ML Repository
1044 records · 26 fitur")
    with col2:
        st.success(f"🤖 **Metode**
K-Means Clustering (k={meta['optimal_k']})
+ Random Forest")
    with col3:
        st.warning(f"🏆 **Performa Model**
Accuracy: {meta['accuracy']*100:.2f}%
Silhouette: {meta['silhouette_score']:.4f}")

    st.markdown("### Deskripsi Proyek")
    st.write("""
    Proyek ini bertujuan menganalisis pola pembelajaran matematika mahasiswa menggunakan
    dua metode Data Mining:
    - **K-Means Clustering** untuk menemukan kelompok mahasiswa berdasarkan perilaku belajar
    - **Random Forest Classification** untuk memprediksi nilai akhir mahasiswa

    Fitur `country` dihapus untuk menghindari bias geografis (*fairness-aware feature selection*).
    """)

    st.markdown("### Anggota Kelompok")
    st.write("- Nama 1 (NIM)
- Nama 2 (NIM)
- Nama 3 (NIM)")

# ══════════════════════════════════
# HALAMAN 2: DATASET OVERVIEW
# ══════════════════════════════════
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.markdown("**Sumber:** UCI Machine Learning Repository — Mathematics Learning in Higher Education")

    # Upload dataset
    uploaded = st.file_uploader("Upload dataset CSV (opsional)", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Dataset dimuat: {df.shape[0]} baris × {df.shape[1]} kolom")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", df.shape[0])
        col2.metric("Jumlah Fitur", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())
        col4.metric("Duplikat", df.duplicated().sum())

        st.subheader("Pratinjau Data")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe().round(2), use_container_width=True)

        # Visualisasi distribusi
        st.subheader("Distribusi Kolom")
        col_sel = st.selectbox("Pilih kolom:", df.columns.tolist())
        if df[col_sel].dtype in [np.float64, np.int64]:
            fig = px.histogram(df, x=col_sel, nbins=30, title=f"Distribusi {col_sel}")
        else:
            vc = df[col_sel].value_counts().reset_index()
            fig = px.bar(vc, x=col_sel, y="count", title=f"Distribusi {col_sel}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload file CSV dataset untuk melihat overview lengkap.")

# ══════════════════════════════════
# HALAMAN 3: PREDIKSI & ANALISIS
# ══════════════════════════════════
elif page == "🤖 Prediksi & Analisis":
    st.title("🤖 Prediksi & Analisis")
    st.markdown("Masukkan data mahasiswa untuk mendapatkan prediksi cluster dan grade.")

    st.subheader("Input Data Mahasiswa")
    st.info("Sesuaikan input di bawah dengan fitur aktual dataset Anda.")

    feature_cols = [c for c in meta['feature_cols'] if c != 'cluster_label']

    with st.form("prediction_form"):
        input_data = {}
        cols = st.columns(3)
        for i, feat in enumerate(feature_cols):
            with cols[i % 3]:
                if feat in meta['cat_features']:
                    options = list(le_dict[feat].classes_) if feat in le_dict else ['Ya', 'Tidak']
                    val = st.selectbox(feat, options)
                    input_data[feat] = le_dict[feat].transform([val])[0] if feat in le_dict else 0
                else:
                    input_data[feat] = st.number_input(feat, min_value=0.0, max_value=100.0, value=50.0)

        submitted = st.form_submit_button("🔍 Analisis", use_container_width=True)

    if submitted:
        # Preprocessing input
        input_df = pd.DataFrame([input_data])

        # Pastikan urutan kolom sesuai
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        input_scaled = scaler.transform(input_df)

        # Prediksi cluster
        cluster = kmeans.predict(input_scaled)[0]
        cluster_name = meta['cluster_names'].get(cluster, f'Cluster {cluster}')

        # Prediksi grade (tambah cluster label)
        input_clf = np.append(input_scaled, cluster).reshape(1, -1)
        grade_pred = rf_model.predict(input_clf)[0]
        grade_proba = rf_model.predict_proba(input_clf)[0]
        grade_label = le_target.inverse_transform([grade_pred])[0] if le_target else grade_pred

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Cluster:** {cluster} — {cluster_name}")
        with col2:
            st.info(f"**Prediksi Grade:** {grade_label}")

        # Probabilitas per kelas
        st.subheader("Probabilitas Grade")
        class_labels = le_target.classes_ if le_target else list(range(len(grade_proba)))
        prob_df = pd.DataFrame({'Grade': class_labels, 'Probabilitas': grade_proba})
        fig = px.bar(prob_df, x='Grade', y='Probabilitas',
                     color='Grade', title='Distribusi Probabilitas Prediksi Grade')
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════
# HALAMAN 4: VISUALISASI
# ══════════════════════════════════
elif page == "📈 Visualisasi":
    st.title("📈 Visualisasi Hasil Analisis")

    # Load gambar hasil visualisasi
    import os
    viz_files = [
        ('distribusi_target.png', 'Distribusi Target Grade'),
        ('elbow_silhouette.png', 'Elbow Method & Silhouette Score'),
        ('cluster_pca_2d.png', 'Visualisasi Cluster (PCA 2D)'),
        ('cluster_profile_heatmap.png', 'Profil Cluster'),
        ('confusion_matrix.png', 'Confusion Matrix'),
        ('feature_importance.png', 'Feature Importance'),
        ('perbandingan_model.png', 'Perbandingan Model'),
        ('heatmap_korelasi.png', 'Heatmap Korelasi'),
    ]

    for fname, title in viz_files:
        if os.path.exists(fname):
            st.subheader(title)
            st.image(fname, use_column_width=True)
        else:
            st.warning(f"{title} ({fname}) belum tersedia. Jalankan notebook terlebih dahulu.")

# ══════════════════════════════════
# HALAMAN 5: ABOUT
# ══════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About")

    st.subheader("Metode Data Mining")
    st.markdown("""
    **1. K-Means Clustering**
    Algoritma clustering berbasis partisi yang mengelompokkan data ke dalam k cluster
    berdasarkan jarak Euclidean ke centroid terdekat. Digunakan untuk menemukan
    profil pelajar yang berbeda tanpa label sebelumnya.

    **2. Random Forest Classification**
    Ensemble learning yang membangun banyak decision tree dan menggabungkan
    prediksinya melalui voting mayoritas. Robust terhadap overfitting dan
    memberikan feature importance.
    """)

    st.subheader("Dataset")
    st.markdown("""
    - **Nama:** Dataset for Assessing Mathematics Learning in Higher Education
    - **Sumber:** UCI Machine Learning Repository (ID: 1031)
    - **Jumlah Data:** 1044 records
    - **Fitur:** Perilaku belajar, motivasi, strategi belajar, hasil akademik
    - **Catatan:** Fitur `country` dihapus (bias geografis)
    """)

    st.subheader("Framework")
    st.info("Menggunakan framework CRISP-DM (Cross Industry Standard Process for Data Mining)")

    st.subheader("Informasi Proyek")
    st.markdown("""
    - **Mata Kuliah:** Data Mining
    - **Semester:** Genap
    - **Jenis:** UAS / Ujian Akhir Semester
    """)