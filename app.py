import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import gdown
import os

# ── Download rf_model.pkl dari Google Drive kalau belum ada
os.makedirs("model", exist_ok=True)
if not os.path.exists("model/rf_model.pkl"):
    with st.spinner("⏳ Mengunduh model dari Google Drive (±1 GB, mohon tunggu)..."):
        gdown.download(
            "https://drive.google.com/uc?id=1poyk0eid_PBIOFjGIOpxnmWP66qYLDsj",
            "model/rf_model.pkl",
            quiet=False
        )

# ── Konfigurasi halaman
st.set_page_config(
    page_title="Math Learning Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Inject CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F4EF;
    color: #1a1a1a;
}

.stApp {
    background-color: #F7F4EF;
}

/* Hero */
.hero-wrap {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 60%, #52B788 100%);
    border-radius: 24px;
    padding: 52px 48px 44px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: rgba(255,255,255,0.06);
}
.hero-wrap::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #ffffff;
    margin: 0 0 10px;
    line-height: 1.15;
}
.hero-title em {
    font-style: italic;
    color: #95D5B2;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* Stat cards */
.stat-row {
    display: flex;
    gap: 16px;
    margin-bottom: 36px;
    flex-wrap: wrap;
}
.stat-card {
    flex: 1;
    min-width: 150px;
    background: #ffffff;
    border-radius: 16px;
    padding: 24px 20px;
    border: 1px solid #E8E2D9;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #2D6A4F, #52B788);
    border-radius: 16px 16px 0 0;
}
.stat-icon { font-size: 1.5rem; margin-bottom: 8px; }
.stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    color: #1B4332;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-label {
    font-size: 0.78rem;
    color: #888;
    font-weight: 500;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Section header */
.section-head {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #1B4332;
    margin: 0 0 20px;
    padding-bottom: 10px;
    border-bottom: 2px solid #D8F3DC;
}

/* Input card */
.input-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 32px;
    border: 1px solid #E8E2D9;
    margin-bottom: 28px;
}

/* Result cards */
.result-cluster {
    background: linear-gradient(135deg, #1B4332, #2D6A4F);
    border-radius: 20px;
    padding: 28px 32px;
    color: white;
    margin-bottom: 16px;
}
.result-cluster .r-label {
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.6);
    margin-bottom: 6px;
}
.result-cluster .r-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.7rem;
    color: #95D5B2;
    margin-bottom: 4px;
}
.result-cluster .r-sub { font-size: 0.9rem; color: rgba(255,255,255,0.75); }

.result-kw {
    background: #ffffff;
    border-radius: 20px;
    padding: 28px 32px;
    border: 2px solid #52B788;
    margin-bottom: 16px;
}
.result-kw .r-label {
    font-size: 0.78rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 6px;
}
.result-kw .r-val {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #1B4332;
    margin-bottom: 4px;
}
.result-kw .conf-badge {
    display: inline-block;
    background: #D8F3DC;
    color: #1B4332;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 20px;
}

/* About cards */
.about-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 24px 28px;
    border: 1px solid #E8E2D9;
    margin-bottom: 16px;
}
.about-card h4 {
    font-family: 'DM Serif Display', serif;
    color: #1B4332;
    font-size: 1.15rem;
    margin: 0 0 10px;
}
.about-card p, .about-card li {
    font-size: 0.9rem;
    color: #555;
    line-height: 1.65;
}

/* Divider */
.sec-divider {
    border: none;
    border-top: 2px solid #E8E2D9;
    margin: 40px 0;
}

/* Streamlit overrides */
div[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.stSelectbox label, .stNumberInput label {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #444 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1B4332, #2D6A4F) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 14px 0 !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.stFileUploader {
    background: #ffffff !important;
    border-radius: 16px !important;
    border: 2px dashed #D8F3DC !important;
    padding: 12px !important;
}
</style>
""", unsafe_allow_html=True)

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

# ════════════════════════════════════════
# HERO
# ════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">Math Learning <em>Analyzer</em></div>
    <p class="hero-sub">Clustering perilaku belajar & prediksi keyword materi mahasiswa · K-Means + Random Forest · CRISP-DM</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# STAT CARDS
# ════════════════════════════════════════
acc = meta.get('accuracy', 0)
sil = meta.get('silhouette_score', 0)
k   = meta.get('optimal_k', '?')

st.markdown(f"""
<div class="stat-row">
    <div class="stat-card">
        <div class="stat-icon">📦</div>
        <div class="stat-val">9.546</div>
        <div class="stat-label">Total Records</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🧩</div>
        <div class="stat-val">{k}</div>
        <div class="stat-label">Optimal Clusters</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🎯</div>
        <div class="stat-val">{acc*100:.1f}%</div>
        <div class="stat-label">RF Accuracy</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">📐</div>
        <div class="stat-val">{sil:.3f}</div>
        <div class="stat-label">Silhouette Score</div>
    </div>
    <div class="stat-card">
        <div class="stat-icon">🌍</div>
        <div class="stat-val">Fairness</div>
        <div class="stat-label">Fitur country dihapus</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# SEKSI 1 — PREDIKSI
# ════════════════════════════════════════
st.markdown('<div class="section-head">🤖 Prediksi Data Mahasiswa</div>', unsafe_allow_html=True)
st.markdown('<div class="input-card">', unsafe_allow_html=True)

feature_cols = [c for c in meta['feature_cols'] if c != 'cluster_label']

with st.form("pred_form"):
    c1, c2, c3 = st.columns(3)
    input_data = {}

    col_map = {0: c1, 1: c2, 2: c3}
    for i, feat in enumerate(feature_cols):
        with col_map[i % 3]:
            if feat in meta.get('cat_features', []):
                options = list(le_dict[feat].classes_) if feat in le_dict else []
                val = st.selectbox(feat, options, key=f"sel_{feat}")
                input_data[feat] = le_dict[feat].transform([val])[0] if feat in le_dict else 0
            else:
                input_data[feat] = st.number_input(feat, min_value=0, max_value=99999, value=100, key=f"num_{feat}")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Analisis Sekarang")

st.markdown('</div>', unsafe_allow_html=True)

if submitted:
    input_df = pd.DataFrame([input_data])
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols].astype(float)

    input_scaled = scaler.transform(input_df)
    cluster = int(kmeans.predict(input_scaled)[0])
    cluster_name = meta.get('cluster_names', {}).get(cluster, f"Cluster {cluster}")

    input_clf = np.append(input_scaled, cluster).reshape(1, -1)
    grade_pred  = rf_model.predict(input_clf)[0]
    grade_proba = rf_model.predict_proba(input_clf)[0]
    grade_label = le_target.inverse_transform([grade_pred])[0]
    confidence  = grade_proba.max() * 100

    # Hasil
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(f"""
        <div class="result-cluster">
            <div class="r-label">Tipe Pelajar (Cluster)</div>
            <div class="r-val">Cluster {cluster}</div>
            <div class="r-sub">{cluster_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown(f"""
        <div class="result-kw">
            <div class="r-label">Prediksi Keyword Materi</div>
            <div class="r-val">{grade_label}</div>
            <span class="conf-badge">Confidence {confidence:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    # Top 5 chart
    top5_idx    = grade_proba.argsort()[-5:][::-1]
    top5_labels = le_target.inverse_transform(top5_idx)
    top5_probs  = grade_proba[top5_idx] * 100

    fig = go.Figure(go.Bar(
        x=top5_probs,
        y=top5_labels,
        orientation='h',
        marker=dict(
            color=top5_probs,
            colorscale=[[0, '#D8F3DC'], [1, '#1B4332']],
            showscale=False
        ),
        text=[f"{p:.1f}%" for p in top5_probs],
        textposition='outside',
    ))
    fig.update_layout(
        title="Top 5 Kemungkinan Keyword",
        xaxis_title="Probabilitas (%)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="DM Sans", size=13),
        margin=dict(l=20, r=60, t=50, b=20),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════
# SEKSI 2 — DATASET OVERVIEW
# ════════════════════════════════════════
st.markdown('<div class="section-head">📊 Dataset Overview</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload file CSV dataset kamu untuk analisis lebih lanjut", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"✅ Dataset dimuat: **{df.shape[0]:,} baris** × **{df.shape[1]} kolom**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records",    f"{df.shape[0]:,}")
    m2.metric("Jumlah Fitur",     df.shape[1])
    m3.metric("Missing Values",   df.isnull().sum().sum())
    m4.metric("Duplikat",         df.duplicated().sum())

    tab1, tab2, tab3 = st.tabs(["📋 Pratinjau Data", "📈 Statistik", "📊 Distribusi"])

    with tab1:
        st.dataframe(df.head(20), use_container_width=True)

    with tab2:
        st.dataframe(df.describe().round(2), use_container_width=True)

    with tab3:
        col_sel = st.selectbox("Pilih kolom:", df.columns.tolist())
        if df[col_sel].dtype in [np.float64, np.int64]:
            fig = px.histogram(df, x=col_sel, nbins=30,
                               title=f"Distribusi {col_sel}",
                               color_discrete_sequence=["#2D6A4F"])
        else:
            vc = df[col_sel].value_counts().reset_index()
            vc.columns = [col_sel, "count"]
            fig = px.bar(vc, x=col_sel, y="count",
                         title=f"Distribusi {col_sel}",
                         color_discrete_sequence=["#2D6A4F"])
        fig.update_layout(
            plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
            font=dict(family="DM Sans"), margin=dict(t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Upload file CSV untuk melihat analisis dataset secara interaktif.")

st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════
# SEKSI 3 — VISUALISASI HASIL
# ════════════════════════════════════════
st.markdown('<div class="section-head">📈 Visualisasi Hasil Analisis</div>', unsafe_allow_html=True)

viz_files = [
    ("analisis_cluster_grade.png", "Analisis Cluster vs Grade"),
    ("cluster_pca_2d.png",         "Visualisasi Cluster (PCA 2D)"),
    ("cluster_profile_heatmap.png","Profil Cluster Heatmap"),
    ("feature_importance.png",     "Feature Importance"),
    ("grade_per_cluster.png",      "Distribusi Grade per Cluster"),
    ("perbandingan_model.png",     "Perbandingan Model"),
]

available = [(f, t) for f, t in viz_files if os.path.exists(f)]
missing   = [(f, t) for f, t in viz_files if not os.path.exists(f)]

if available:
    for i in range(0, len(available), 2):
        cols = st.columns(2)
        for j, (fname, title) in enumerate(available[i:i+2]):
            with cols[j]:
                st.caption(title)
                st.image(fname, use_container_width=True)
else:
    for fname, title in missing:
        st.warning(f"**{title}** — file `{fname}` belum tersedia. Jalankan notebook terlebih dahulu.")

st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════
# SEKSI 4 — ABOUT
# ════════════════════════════════════════
st.markdown('<div class="section-head">ℹ️ Tentang Proyek</div>', unsafe_allow_html=True)

a1, a2 = st.columns(2)
with a1:
    st.markdown("""
    <div class="about-card">
        <h4>📦 Dataset</h4>
        <ul>
            <li><b>Nama:</b> Mathematics Learning in Higher Education</li>
            <li><b>Sumber:</b> UCI ML Repository (ID: 1031)</li>
            <li><b>Jumlah Data:</b> 9.546 records</li>
            <li><b>Fitur:</b> Student ID, Question ID, Type of Answer, Question Level, Topic, Subtopic</li>
            <li><b>Target:</b> Keywords</li>
            <li><b>Catatan:</b> Fitur <code>country</code> dihapus (fairness-aware)</li>
        </ul>
    </div>
    <div class="about-card">
        <h4>🔬 Metode</h4>
        <p><b>K-Means Clustering</b> — menemukan kelompok mahasiswa berdasarkan pola belajar tanpa label. Optimal k dipilih via Silhouette Score.</p>
        <p><b>Random Forest Classification</b> — memprediksi keyword materi menggunakan ensemble decision tree. Fitur cluster_label ditambahkan sebagai fitur tambahan.</p>
    </div>
    """, unsafe_allow_html=True)

with a2:
    st.markdown(f"""
    <div class="about-card">
        <h4>📊 Performa Model</h4>
        <ul>
            <li><b>Optimal K:</b> {meta.get('optimal_k', '?')}</li>
            <li><b>Silhouette Score:</b> {meta.get('silhouette_score', 0):.4f}</li>
            <li><b>RF Accuracy:</b> {meta.get('accuracy', 0)*100:.2f}%</li>
            <li><b>F1-Macro:</b> {meta.get('f1_macro', 0):.4f}</li>
        </ul>
    </div>
    <div class="about-card">
        <h4>🧭 Framework & Info</h4>
        <ul>
            <li><b>Framework:</b> CRISP-DM</li>
            <li><b>Mata Kuliah:</b> Data Mining</li>
            <li><b>Semester:</b> Genap</li>
            <li><b>Jenis:</b> UAS / Ujian Akhir Semester</li>
        </ul>
        <p style="margin-top:12px"><b>Anggota Kelompok:</b><br>
        · Nama 1 (NIM)<br>· Nama 2 (NIM)<br>· Nama 3 (NIM)</p>
    </div>
    """, unsafe_allow_html=True)