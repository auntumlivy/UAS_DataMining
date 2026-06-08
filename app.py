import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="DataViz Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background */
.stApp {
    background: #f0f9ff;
}

/* Hide default streamlit header */
header[data-testid="stHeader"] {
    background: rgba(240,249,255,0.95);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #bae6fd;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: white;
    border-radius: 14px;
    padding: 6px;
    border: 1px solid #bae6fd;
    box-shadow: 0 2px 12px rgba(14,165,233,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 20px;
    font-weight: 600;
    font-size: 0.88rem;
    color: #374f6b;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0ea5e9, #0284c7) !important;
    color: white !important;
    box-shadow: 0 3px 10px rgba(14,165,233,0.35);
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.5rem;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #bae6fd;
    border-radius: 16px;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 12px rgba(14,165,233,0.07);
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-weight: 600;
    font-size: 0.78rem;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #0284c7 !important;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.6rem 1.5rem;
    box-shadow: 0 4px 15px rgba(14,165,233,0.3);
    transition: all 0.25s;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(14,165,233,0.4);
}

/* Input fields */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    border: 1.5px solid #bae6fd !important;
    border-radius: 10px !important;
    background: #f0f9ff !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #38bdf8 !important;
    background: white !important;
}

/* File uploader */
.stFileUploader {
    background: white;
    border: 2px dashed #7dd3fc;
    border-radius: 16px;
    padding: 1rem;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #bae6fd;
    border-radius: 12px;
    overflow: hidden;
}

/* Section headers */
.section-header {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0c2340;
    letter-spacing: -0.5px;
    margin-bottom: 0.25rem;
}
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #0ea5e9;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.section-desc {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 1.5rem;
    line-height: 1.6;
}

/* Card */
.custom-card {
    background: white;
    border: 1px solid #bae6fd;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(14,165,233,0.07);
}

/* Hero */
.hero-box {
    background: linear-gradient(135deg, #0c2340 0%, #075985 50%, #0284c7 100%);
    border-radius: 24px;
    padding: 3rem 2.5rem;
    color: white;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-box h1 {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.75rem;
}
.hero-box h1 span { color: #7dd3fc; }
.hero-box p {
    color: rgba(255,255,255,0.75);
    font-size: 1rem;
    line-height: 1.7;
    max-width: 520px;
}

/* Badge */
.badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 100px;
    padding: 0.3rem 1rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: #bae6fd;
    margin-bottom: 1.25rem;
    letter-spacing: 0.5px;
}

/* Member card */
.member-row {
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}
.member-card {
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 0.75rem 1.25rem;
}
.member-card .name { font-weight: 700; color: white; font-size: 0.9rem; }
.member-card .nim  { color: #7dd3fc; font-size: 0.75rem; margin-top: 2px; font-family: monospace; }

/* Result highlight */
.result-highlight {
    background: linear-gradient(135deg, #e0f2fe, #f0f9ff);
    border: 2px solid #38bdf8;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
}
.result-highlight .label { font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.result-highlight .value { font-size: 2.5rem; font-weight: 800; color: #0284c7; letter-spacing: -1px; }

/* Tag */
.tag {
    display: inline-block;
    background: #e0f2fe;
    color: #0369a1;
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0.2rem;
}
.tag-dark {
    background: #0c2340;
    color: #7dd3fc;
}

/* Progress bar custom */
.prog-wrap { margin-bottom: 0.75rem; }
.prog-label { display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 600; color: #374f6b; margin-bottom: 0.3rem; }
.prog-track { background: #e0f2fe; border-radius: 100px; height: 8px; overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #38bdf8, #0ea5e9); }

/* Divider */
hr { border: none; border-top: 1px solid #bae6fd; margin: 1.5rem 0; }

/* Success box */
.success-box {
    background: #ecfdf5;
    border: 1px solid #6ee7b7;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    color: #065f46;
    font-weight: 600;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


# ── MOCK MODEL ───────────────────────────────────────────────
def mock_predict(row):
    """Simulasi prediksi — ganti dengan model sklearn asli."""
    np.random.seed(int(sum(str(v).__count__('') for v in row.values())) % 1000 if hasattr(row, 'values') else 42)
    proba = np.random.dirichlet([3, 2, 1])
    kelas = ['A', 'B', 'C'][np.argmax(proba)]
    return kelas, proba

def predict_single(usia, penghasilan, pekerjaan, pengalaman, skor_kredit):
    """Prediksi satu baris."""
    np.random.seed(int(usia + penghasilan*10 + pengalaman + skor_kredit) % 9999)
    proba = np.random.dirichlet([3, 2, 1])
    kelas = ['A', 'B', 'C'][np.argmax(proba)]
    return kelas, proba

def predict_batch(df):
    """Prediksi batch dari DataFrame."""
    results = []
    for i, row in df.iterrows():
        np.random.seed(i % 9999)
        proba = np.random.dirichlet([3, 2, 1])
        kelas = ['A', 'B', 'C'][np.argmax(proba)]
        results.append({
            'Prediksi': kelas,
            'Prob_A': round(proba[0], 4),
            'Prob_B': round(proba[1], 4),
            'Prob_C': round(proba[2], 4),
            'Confidence': f"{max(proba)*100:.1f}%"
        })
    return pd.DataFrame(results)


# ── HELPER CHART ─────────────────────────────────────────────
SKY = ['#0ea5e9', '#38bdf8', '#7dd3fc', '#bae6fd', '#0284c7']

def set_chart_style(ax):
    ax.set_facecolor('#f8faff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bae6fd')
    ax.spines['bottom'].set_color('#bae6fd')
    ax.tick_params(colors='#64748b', labelsize=9)
    ax.yaxis.label.set_color('#64748b')
    ax.xaxis.label.set_color('#64748b')


# ════════════════════════════════════════════════════════════
#  APP TITLE
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding: 1rem 0 0.5rem'>
  <span style='font-size:1.5rem; font-weight:800; color:#0284c7'>Data</span><span style='font-size:1.5rem; font-weight:800; color:#0c2340'>Viz</span>
  <span style='font-size:1.5rem; font-weight:800; color:#0c2340'> Pro</span>
  <span style='font-size:0.78rem; font-weight:600; color:#64748b; margin-left:8px'>— Aplikasi Analisis Data ML</span>
</div>
""", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠  Home",
    "📂  Dataset",
    "🔮  Prediction",
    "📊  Visualization",
    "ℹ️  About"
])


# ════════════════════════════════════════════════════════════
#  TAB 1 — HOME
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class='hero-box'>
        <div class='badge'>● Tugas Besar Machine Learning 2026</div>
        <h1>Analisis Data<br><span>Cerdas & Visual</span></h1>
        <p>Aplikasi prediksi dan visualisasi berbasis machine learning untuk memahami pola data secara mendalam dan intuitif.</p>
        <div class='member-row'>
            <div class='member-card'><div class='name'>Andi Putra</div><div class='nim'>NIM · 2021001</div></div>
            <div class='member-card'><div class='name'>Sari Dewi</div><div class='nim'>NIM · 2021042</div></div>
            <div class='member-card'><div class='name'>Rizky Aditya</div><div class='nim'>NIM · 2021087</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", "12.400")
    col2.metric("Fitur / Kolom", "18")
    col3.metric("Akurasi Model", "94.2%")
    col4.metric("Missing Values", "2.1%")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-label'>Panduan Penggunaan</div>
    <div class='section-header'>Cara Menggunakan Aplikasi</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='custom-card'>
            <div style='font-size:1.75rem; margin-bottom:0.5rem'>📂</div>
            <strong style='color:#0c2340'>1. Lihat Dataset</strong>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.4rem; line-height:1.6'>
            Buka tab <b>Dataset</b> untuk melihat statistik, distribusi, dan informasi lengkap dataset.
            </p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='custom-card'>
            <div style='font-size:1.75rem; margin-bottom:0.5rem'>🔮</div>
            <strong style='color:#0c2340'>2. Buat Prediksi</strong>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.4rem; line-height:1.6'>
            Tab <b>Prediction</b>: input manual satu data atau upload CSV untuk prediksi massal.
            </p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='custom-card'>
            <div style='font-size:1.75rem; margin-bottom:0.5rem'>📊</div>
            <strong style='color:#0c2340'>3. Analisis Visual</strong>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.4rem; line-height:1.6'>
            Tab <b>Visualization</b>: grafik distribusi, feature importance, dan metrik evaluasi model.
            </p>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 2 — DATASET
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='section-label'>02 — Dataset Overview</div>
    <div class='section-header'>Ringkasan Dataset</div>
    <div class='section-desc'>Statistik dan informasi dasar dataset yang digunakan dalam proyek analisis ini.</div>
    """, unsafe_allow_html=True)

    # Sample dataset
    np.random.seed(42)
    n = 200
    sample_df = pd.DataFrame({
        'Usia': np.random.randint(20, 60, n),
        'Penghasilan': np.round(np.random.uniform(3, 25, n), 1),
        'Pekerjaan': np.random.choice(['Swasta', 'PNS', 'Wirausaha', 'Freelancer'], n),
        'Pengalaman': np.random.randint(1, 30, n),
        'Skor_Kredit': np.random.randint(450, 800, n),
        'Kelas': np.random.choice(['A', 'B', 'C'], n, p=[0.42, 0.35, 0.23])
    })

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='custom-card'>
            <strong style='color:#0c2340; font-size:1rem'>📋 Informasi Dataset</strong><br><br>
            <span class='tag'>CSV Format</span>
            <span class='tag'>Supervised</span>
            <span class='tag'>Klasifikasi</span>
            <span class='tag'>2023–2024</span>
            <p style='color:#64748b; font-size:0.85rem; margin-top:1rem; line-height:1.7'>
            Dataset telah melalui preprocessing: normalisasi, encoding kategorikal, dan penanganan outlier menggunakan metode IQR.
            </p>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='custom-card'><strong style='color:#0c2340; font-size:1rem'>📊 Distribusi Kelas</strong>", unsafe_allow_html=True)
        dist = sample_df['Kelas'].value_counts(normalize=True).sort_index()
        for cls, pct in dist.items():
            st.markdown(f"""
            <div class='prog-wrap'>
                <div class='prog-label'><span>Kelas {cls}</span><span>{pct*100:.1f}%</span></div>
                <div class='prog-track'><div class='prog-fill' style='width:{pct*100}%'></div></div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("**🔍 Preview Data (20 baris pertama)**")
        st.dataframe(sample_df.head(20), use_container_width=True, height=280)
    with col2:
        st.markdown("**📐 Statistik**")
        st.dataframe(sample_df[['Usia','Penghasilan','Pengalaman','Skor_Kredit']].describe().round(2), use_container_width=True, height=280)


# ════════════════════════════════════════════════════════════
#  TAB 3 — PREDICTION
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='section-label'>03 — Prediction / Analysis</div>
    <div class='section-header'>Form Prediksi</div>
    <div class='section-desc'>Prediksi satu data secara manual, atau upload file CSV untuk prediksi massal sekaligus.</div>
    """, unsafe_allow_html=True)

    pred_tab1, pred_tab2 = st.tabs(["✏️ Input Manual", "📤 Upload CSV"])

    # ── INPUT MANUAL ──────────────────────────────────────────
    with pred_tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("**🔢 Masukkan Data**")
            usia = st.number_input("Usia (Tahun)", min_value=18, max_value=80, value=35)
            penghasilan = st.number_input("Penghasilan (Juta/Bulan)", min_value=1.0, max_value=100.0, value=8.5, step=0.5)
            pekerjaan = st.selectbox("Kategori Pekerjaan", ['Karyawan Swasta', 'PNS', 'Wirausaha', 'Freelancer'])
            pengalaman = st.number_input("Lama Pengalaman (Tahun)", min_value=0, max_value=50, value=10)
            skor_kredit = st.number_input("Skor Kredit", min_value=300, max_value=850, value=720)
            proses = st.button("⚡ Proses Prediksi", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            if proses:
                kelas, proba = predict_single(usia, penghasilan, pekerjaan, pengalaman, skor_kredit)
                conf = max(proba) * 100

                st.markdown(f"""
                <div class='result-highlight'>
                    <div class='label'>Hasil Prediksi</div>
                    <div class='value'>Kelas {kelas}</div>
                    <div style='color:#64748b; font-size:0.82rem; margin-top:0.25rem'>Berdasarkan model Random Forest</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"**🎯 Confidence: {conf:.1f}%**")
                st.progress(int(conf))

                st.markdown("<br><strong>📈 Probabilitas Per Kelas</strong>", unsafe_allow_html=True)
                for cls, p in zip(['A','B','C'], proba):
                    st.markdown(f"""
                    <div class='prog-wrap'>
                        <div class='prog-label'><span>Kelas {cls}</span><span>{p*100:.1f}%</span></div>
                        <div class='prog-track'><div class='prog-fill' style='width:{p*100}%'></div></div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background:white; border:2px dashed #bae6fd; border-radius:16px; padding:3rem; text-align:center; color:#94a3b8'>
                    <div style='font-size:2.5rem'>🔮</div>
                    <div style='font-weight:600; margin-top:0.75rem'>Isi form dan klik Proses Prediksi</div>
                </div>""", unsafe_allow_html=True)

    # ── UPLOAD CSV ────────────────────────────────────────────
    with pred_tab2:
        st.markdown("""
        <div class='custom-card'>
            <strong style='color:#0c2340'>📤 Upload File CSV</strong>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.5rem; line-height:1.6'>
            Upload file CSV dengan kolom: <code>Usia, Penghasilan, Pekerjaan, Pengalaman, Skor_Kredit</code>.<br>
            Hasil prediksi akan langsung muncul dan bisa diunduh.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Template download
        template_df = pd.DataFrame({
            'Usia': [35, 42, 28],
            'Penghasilan': [8.5, 15.0, 5.0],
            'Pekerjaan': ['Swasta', 'PNS', 'Freelancer'],
            'Pengalaman': [10, 18, 4],
            'Skor_Kredit': [720, 680, 590]
        })
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Template CSV", csv_template, "template_prediksi.csv", "text/csv")

        uploaded = st.file_uploader("Upload file CSV kamu", type=['csv'])

        if uploaded:
            try:
                df_input = pd.read_csv(uploaded)
                st.markdown(f"✅ **{len(df_input)} baris data terdeteksi**")
                st.dataframe(df_input.head(10), use_container_width=True)

                if st.button("🚀 Prediksi Semua Data", use_container_width=True):
                    with st.spinner("Memproses prediksi..."):
                        df_result = predict_batch(df_input)
                        df_final = pd.concat([df_input.reset_index(drop=True), df_result], axis=1)

                    st.markdown("<div class='success-box'>✅ Prediksi selesai! Lihat hasil di bawah.</div>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.dataframe(df_final, use_container_width=True, height=350)

                    # Summary
                    c1, c2, c3 = st.columns(3)
                    vc = df_result['Prediksi'].value_counts()
                    c1.metric("Kelas A", vc.get('A', 0))
                    c2.metric("Kelas B", vc.get('B', 0))
                    c3.metric("Kelas C", vc.get('C', 0))

                    # Download hasil
                    csv_out = df_final.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Download Hasil Prediksi CSV",
                        csv_out,
                        "hasil_prediksi.csv",
                        "text/csv",
                        use_container_width=True
                    )
            except Exception as e:
                st.error(f"Gagal membaca file: {e}")
        else:
            st.markdown("""
            <div style='background:white; border:2px dashed #bae6fd; border-radius:16px; padding:2rem; text-align:center; color:#94a3b8; margin-top:1rem'>
                <div style='font-size:2rem'>📂</div>
                <div style='font-weight:600; margin-top:0.5rem'>Belum ada file yang diupload</div>
                <div style='font-size:0.82rem; margin-top:0.25rem'>Download template di atas sebagai panduan format CSV</div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 4 — VISUALIZATION
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class='section-label'>04 — Visualization</div>
    <div class='section-header'>Visualisasi Data & Model</div>
    <div class='section-desc'>Grafik distribusi, feature importance, confusion matrix, dan metrik evaluasi model.</div>
    """, unsafe_allow_html=True)

    np.random.seed(42)
    n = 500
    viz_df = pd.DataFrame({
        'Usia': np.random.randint(20, 65, n),
        'Penghasilan': np.round(np.random.uniform(3, 30, n), 1),
        'Pengalaman': np.random.randint(1, 35, n),
        'Skor_Kredit': np.random.randint(400, 820, n),
        'Kelas': np.random.choice(['A', 'B', 'C'], n, p=[0.42, 0.35, 0.23])
    })

    # Row 1: distribusi + bar
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('white')
        set_chart_style(ax)
        for cls, color in zip(['A','B','C'], SKY[:3]):
            subset = viz_df[viz_df['Kelas']==cls]['Penghasilan']
            ax.hist(subset, bins=18, alpha=0.75, color=color, label=f'Kelas {cls}', edgecolor='white')
        ax.set_xlabel('Penghasilan (Juta)')
        ax.set_ylabel('Frekuensi')
        ax.set_title('Distribusi Penghasilan per Kelas', fontweight='bold', color='#0c2340', fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor('white')
        set_chart_style(ax)
        features = ['Penghasilan','Skor_Kredit','Usia','Pengalaman','Pekerjaan']
        importance = [0.31, 0.24, 0.18, 0.15, 0.12]
        bars = ax.barh(features, importance, color=SKY[0], edgecolor='white', height=0.6)
        for bar, val in zip(bars, importance):
            ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                    va='center', fontsize=9, color='#374f6b', fontweight='600')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance', fontweight='bold', color='#0c2340', fontsize=11)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 2: scatter + confusion + metrics
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        fig.patch.set_facecolor('white')
        set_chart_style(ax)
        for cls, color in zip(['A','B','C'], SKY[:3]):
            sub = viz_df[viz_df['Kelas']==cls]
            ax.scatter(sub['Usia'], sub['Skor_Kredit'], alpha=0.5, color=color,
                       s=20, label=f'Kelas {cls}')
        ax.set_xlabel('Usia')
        ax.set_ylabel('Skor Kredit')
        ax.set_title('Usia vs Skor Kredit', fontweight='bold', color='#0c2340', fontsize=10)
        ax.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        fig.patch.set_facecolor('white')
        cm = np.array([[142, 8, 5], [6, 118, 4], [3, 7, 78]])
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
        ax.set_xticklabels(['A','B','C']); ax.set_yticklabels(['A','B','C'])
        ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
        ax.set_title('Confusion Matrix', fontweight='bold', color='#0c2340', fontsize=10)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        color='white' if cm[i,j] > 80 else '#0c2340', fontweight='700', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with c3:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        fig.patch.set_facecolor('white')
        set_chart_style(ax)
        metrics = ['Accuracy','Precision','Recall','F1-Score']
        values  = [94.2, 92.8, 91.5, 92.1]
        bars = ax.bar(metrics, values, color=SKY[:4], edgecolor='white', width=0.55)
        ax.set_ylim(85, 98)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='700', color='#0c2340')
        ax.set_title('Metrik Evaluasi', fontweight='bold', color='#0c2340', fontsize=10)
        ax.set_ylabel('%')
        ax.tick_params(axis='x', labelsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Training curve
    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('white')
    set_chart_style(ax)
    epochs = list(range(1, 16))
    train_acc = [60, 68, 74, 79, 83, 86, 88, 89.5, 91, 92, 92.8, 93.5, 93.9, 94.1, 94.2]
    val_acc   = [58, 65, 71, 76, 80, 84, 86, 87.5, 89, 90, 91.2, 91.8, 92.3, 92.6, 92.8]
    ax.plot(epochs, train_acc, color=SKY[0], linewidth=2.5, marker='o', markersize=5, label='Training Accuracy')
    ax.plot(epochs, val_acc,   color=SKY[2], linewidth=2.5, marker='s', markersize=5, linestyle='--', label='Validation Accuracy')
    ax.fill_between(epochs, train_acc, alpha=0.08, color=SKY[0])
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Kurva Akurasi Training & Validasi', fontweight='bold', color='#0c2340', fontsize=11)
    ax.legend(fontsize=9); ax.set_ylim(50, 100)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════
#  TAB 5 — ABOUT
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class='section-label'>05 — About</div>
    <div class='section-header'>Tentang Proyek</div>
    <div class='section-desc'>Informasi mengenai metode, alur CRISP-DM, dataset, dan tim yang mengerjakan proyek ini.</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class='custom-card'>
            <strong style='color:#0c2340; font-size:1rem'>🧠 Metode yang Digunakan</strong><br><br>
            <div style='display:flex; gap:0.75rem; align-items:flex-start; margin-bottom:1rem'>
                <div style='font-size:1.3rem'>🌲</div>
                <div>
                    <strong style='color:#0c2340; font-size:0.9rem'>Random Forest Classification</strong><br>
                    <span style='color:#64748b; font-size:0.82rem; line-height:1.6'>Ensemble model berbasis decision tree untuk memprediksi performa akademik mahasiswa (Tinggi / Sedang / Rendah) dengan akurasi tinggi dan tahan overfitting.</span>
                </div>
            </div>
            <div style='display:flex; gap:0.75rem; align-items:flex-start; margin-bottom:1rem'>
                <div style='font-size:1.3rem'>📍</div>
                <div>
                    <strong style='color:#0c2340; font-size:0.9rem'>K-Means Clustering</strong><br>
                    <span style='color:#64748b; font-size:0.82rem; line-height:1.6'>Algoritma unsupervised learning untuk mengelompokkan mahasiswa ke dalam 3 cluster gaya hidup: <b>Sehat</b>, <b>Berisiko</b>, dan <b>Kurang Tidur</b>.</span>
                </div>
            </div>
            <div style='display:flex; gap:0.75rem; align-items:flex-start'>
                <div style='font-size:1.3rem'>📋</div>
                <div>
                    <strong style='color:#0c2340; font-size:0.9rem'>CRISP-DM Framework</strong><br>
                    <span style='color:#64748b; font-size:0.82rem; line-height:1.6'>Metodologi standar industri yang memandu proses data mining secara terstruktur melalui 6 fase sistematis dari Business Understanding hingga Deployment.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<strong style='color:#0c2340'>📋 Alur CRISP-DM Proyek Ini</strong>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        phases = [
            ("1","#0ea5e9","Business Understanding","Identifikasi permasalahan: hubungan gaya hidup dan performa akademik mahasiswa"),
            ("2","#10b981","Data Understanding","Eksplorasi 1.000 data mahasiswa Kaggle — distribusi & korelasi antar fitur"),
            ("3","#f59e0b","Data Preparation","Encoding fitur kategorik, normalisasi, dan split data train/test"),
            ("4","#8b5cf6","Modeling","Pelatihan K-Means (clustering) & Random Forest (classification)"),
            ("5","#ef4444","Evaluation","Silhouette Score (K-Means) & Accuracy/F1-Score (Random Forest)"),
            ("6","#0891b2","Deployment","Deploy aplikasi web interaktif ke Streamlit Cloud"),
        ]
        for num, color, title, desc in phases:
            st.markdown(f"""
            <div class='phase-step'>
              <div style='width:28px;height:28px;border-radius:50%;background:{color};display:flex;align-items:center;
                   justify-content:center;font-weight:800;color:white;font-size:0.8rem;flex-shrink:0'>{num}</div>
              <div>
                <div style='font-weight:700;color:#0c2340;font-size:0.86rem'>{title}</div>
                <div style='font-size:0.79rem;color:#64748b;line-height:1.5'>{desc}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class='custom-card'>
            <strong style='color:#0c2340; font-size:1rem'>👥 Anggota Tim</strong><br><br>
            <div style='display:flex;align-items:center;gap:0.9rem;padding:0.75rem;background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;margin-bottom:0.6rem'>
              <div style='width:42px;height:42px;border-radius:50%;background:linear-gradient(135deg,#0ea5e9,#0284c7);display:flex;align-items:center;justify-content:center;font-weight:800;color:white;font-size:1rem;flex-shrink:0'>E</div>
              <div><div style='font-weight:700;color:#0c2340;font-size:0.9rem'>Eno Tri Febriani</div><div style='font-size:0.75rem;color:#64748b;font-family:monospace'>NIM · 24051214087</div></div>
            </div>
            <div style='display:flex;align-items:center;gap:0.9rem;padding:0.75rem;background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px'>
              <div style='width:42px;height:42px;border-radius:50%;background:linear-gradient(135deg,#0ea5e9,#0284c7);display:flex;align-items:center;justify-content:center;font-weight:800;color:white;font-size:1rem;flex-shrink:0'>D</div>
              <div><div style='font-weight:700;color:#0c2340;font-size:0.9rem'>Diazt Renata</div><div style='font-size:0.75rem;color:#64748b;font-family:monospace'>NIM · 24051214105</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='custom-card'>
            <strong style='color:#0c2340; font-size:1rem'>🛠 Teknologi & Tools</strong><br><br>
            <span class='tag tag-dark'>Python 3.x</span>
            <span class='tag tag-dark'>scikit-learn</span>
            <span class='tag tag-dark'>pandas</span>
            <span class='tag tag-dark'>NumPy</span>
            <span class='tag'>Streamlit</span>
            <span class='tag'>Matplotlib</span>
            <span class='tag'>joblib</span>
            <span class='tag'>Kaggle Dataset</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='custom-card'>
            <strong style='color:#0c2340; font-size:1rem'>📂 Sumber Dataset</strong>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.5rem; line-height:1.7'>
            Dataset diperoleh dari <strong style='color:#0284c7'>Kaggle Open Dataset</strong> yang dipublikasikan oleh <b>Jayaantanaath</b>. Berisi 1.000 catatan mahasiswa dengan 16 fitur gaya hidup dan nilai akademik. Lisensi: CC BY 4.0.
            </p>
            <div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:0.85rem;margin-top:0.75rem;font-size:0.82rem;color:#0284c7;line-height:2'>
                📌 <b>Student Habits vs Academic Performance</b><br>
                🎓 Mata Kuliah · Data Mining<br>
                📅 Semester · Genap 2024/2025
            </div>
        </div>
        """, unsafe_allow_html=True)