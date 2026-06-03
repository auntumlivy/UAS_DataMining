import streamlit as st
import joblib
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Analisis Gaya Hidup Mahasiswa",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)


st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Background ── */
  .stApp { background: #e8f0fa; }

  /* ── Hero banner ── */
  .hero {
    background: linear-gradient(135deg, #c5d8f5 0%, #d6e8ff 50%, #c2d4f0 100%);
    border: 1px solid rgba(100,140,210,0.35);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(100,140,220,0.18) 0%, transparent 70%);
  }
  .hero::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(120,160,230,0.18) 0%, transparent 70%);
  }
  .hero h1 { font-size: 2rem; font-weight: 800; color: #1a3560; margin: 0; letter-spacing: -0.5px; }
  .hero p  { color: #3a5a9a; margin: 0.4rem 0 0; font-size: 0.95rem; }

  /* ── Section cards ── */
  .card {
    background: #f0f6ff;
    border: 1px solid rgba(100,140,210,0.22);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
  }
  .card-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #5577aa;
    margin-bottom: 1rem;
  }

  /* ── Sliders & number inputs ── */
  .stSlider > div > div { background: rgba(100,140,210,0.15) !important; }
  .stSlider [data-baseweb="slider"] [role="slider"] {
    background: #4a7fd4 !important;
    border: 2px solid #7aaae8 !important;
    width: 18px !important;
    height: 18px !important;
  }
  .stSlider [data-baseweb="slider-track-highlight"] { background: #4a7fd4 !important; }

  /* ── Select boxes ── */
  .stSelectbox [data-baseweb="select"] > div {
    background: #dceaf9 !important;
    border-color: rgba(100,140,210,0.3) !important;
    color: #1a3560 !important;
    border-radius: 10px !important;
  }

  /* ── Number input ── */
  .stNumberInput input {
    background: #dceaf9 !important;
    border-color: rgba(100,140,210,0.3) !important;
    color: #1a3560 !important;
    border-radius: 10px !important;
  }

  /* ── Predict button ── */
  div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #4a7fd4, #6a9fe8);
    color: white;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 16px rgba(74,127,212,0.3);
  }
  div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #3a6bbf, #5a8fd8);
    transform: translateY(-1px);
    box-shadow: 0 6px 22px rgba(74,127,212,0.4);
  }
  div[data-testid="stButton"] > button:active { transform: translateY(0px); }

  /* ── Result boxes ── */
  .result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 1.4rem;
  }
  .result-box {
    border-radius: 14px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    border: 1px solid rgba(100,140,210,0.2);
  }
  .result-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.65; margin-bottom: 0.35rem; }
  .result-value { font-size: 1.6rem; font-weight: 800; line-height: 1.1; }
  .result-sub   { font-size: 0.75rem; margin-top: 0.3rem; opacity: 0.65; }

  /* Colour themes per result */
  .box-sehat      { background: rgba(30,160,120,0.12); border-color: rgba(30,160,120,0.35); color: #0e6b50; }
  .box-berisiko   { background: rgba(210,140,30,0.12); border-color: rgba(210,140,30,0.35); color: #8a5a0a; }
  .box-kurangtidur{ background: rgba(210,60,60,0.10);  border-color: rgba(210,60,60,0.3);   color: #8a2020; }
  .box-high       { background: rgba(30,160,120,0.12); border-color: rgba(30,160,120,0.35); color: #0e6b50; }
  .box-medium     { background: rgba(210,140,30,0.12); border-color: rgba(210,140,30,0.35); color: #8a5a0a; }
  .box-low        { background: rgba(210,60,60,0.10);  border-color: rgba(210,60,60,0.3);   color: #8a2020; }

  /* ── Probability bar ── */
  .prob-row { margin-bottom: 0.55rem; }
  .prob-label-row {
    display: flex; justify-content: space-between;
    font-size: 0.78rem; margin-bottom: 0.22rem; color: #2a4575;
  }
  .prob-bar-bg {
    background: rgba(100,140,210,0.15);
    border-radius: 100px;
    height: 7px;
    overflow: hidden;
  }
  .prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 0.6s ease;
  }

  /* ── Tips ── */
  .tip-item {
    display: flex; align-items: flex-start; gap: 0.7rem;
    padding: 0.65rem 0.8rem;
    border-radius: 10px;
    background: rgba(100,140,210,0.07);
    border: 1px solid rgba(100,140,210,0.18);
    margin-bottom: 0.55rem;
    font-size: 0.85rem;
    color: #1e3a6a;
    line-height: 1.45;
  }
  .tip-icon { font-size: 1rem; flex-shrink: 0; }

  /* ── Metric pills (top stats) ── */
  .pill-row {
    display: flex; gap: 0.7rem; flex-wrap: wrap;
    justify-content: center;
    margin-bottom: 1.6rem;
  }
  .pill {
    background: rgba(100,140,210,0.12);
    border: 1px solid rgba(100,140,210,0.25);
    border-radius: 100px;
    padding: 0.35rem 0.9rem;
    font-size: 0.78rem;
    color: #3a5a9a;
    white-space: nowrap;
  }
  .pill b { color: #1a3560; }

  /* ── Section header ── */
  .sec-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: #1a3560;
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  /* ── Skala hint ── */
  .skala-hint {
    font-size: 0.76rem;
    color: #5577aa;
    margin-top: -0.6rem;
    margin-bottom: 0.8rem;
    padding: 0.4rem 0.7rem;
    background: rgba(100,140,210,0.07);
    border-left: 3px solid rgba(74,127,212,0.4);
    border-radius: 0 8px 8px 0;
    line-height: 1.5;
  }

  /* Hide default streamlit elements */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 760px; }

  /* Label text */
  .stSlider label, .stSelectbox label, .stNumberInput label {
    color: #2a4070 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
  }
  /* Value badge on slider */
  .stSlider [data-testid="stThumbValue"] {
    background: #4a7fd4 !important;
    color: white !important;
    border-radius: 6px !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
  }

  /* Divider */
  hr { border: none; border-top: 1px solid rgba(100,140,210,0.2); margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource(show_spinner=False)
def load_models():
    model_clf      = joblib.load("model/model_clf.pkl")
    model_cluster  = joblib.load("model/model_cluster.pkl")
    scaler_clf     = joblib.load("model/scaler_clf.pkl")
    scaler_cluster = joblib.load("model/scaler_cluster.pkl")
    le_performa    = joblib.load("model/le_performa.pkl")
    le_dict        = joblib.load("model/le_dict.pkl")
    with open("model/cluster_names.json")    as f: cluster_names    = json.load(f)
    with open("model/streamlit_features.json") as f: streamlit_features = json.load(f)
    with open("model/all_features.json")    as f: all_features     = json.load(f)
    return model_clf, model_cluster, scaler_clf, scaler_cluster, \
           le_performa, le_dict, cluster_names, streamlit_features, all_features

model_clf, model_cluster, scaler_clf, scaler_cluster, \
le_performa, le_dict, cluster_names, streamlit_features, all_features = load_models()



CLUSTER_CSS = {"Sehat": "sehat", "Berisiko": "berisiko", "Kurang Tidur": "kurangtidur"}
PERF_CSS    = {"High": "high", "Medium": "medium", "Low": "low"}
PERF_ID     = {"High": "🟢 Tinggi", "Medium": "🟡 Sedang", "Low": "🔴 Rendah"}
CLUSTER_ID  = {"Sehat": "🌿 Sehat", "Berisiko": "⚠️ Berisiko", "Kurang Tidur": "😴 Kurang Tidur"}

BAR_COLORS = {
    "High":   "linear-gradient(90deg,#1aaa80,#2ec99a)",
    "Medium": "linear-gradient(90deg,#d4900a,#f0b030)",
    "Low":    "linear-gradient(90deg,#d44040,#e87070)",
}

ENERGI_DESC = {
    1:  "😴 Selalu kelelahan, tidak bisa fokus sama sekali",
    2:  "😓 Sangat sering lelah & sulit berkonsentrasi",
    3:  "😔 Sering lelah & fokus mudah pecah",
    4:  "😕 Kadang lelah, fokus kurang stabil",
    5:  "😐 Cukup — energi & fokus naik turun",
    6:  "🙂 Lumayan baik, sesekali masih mudah terdistraksi",
    7:  "😊 Cukup segar & bisa fokus dengan baik",
    8:  "😄 Berenergi & fokus stabil sepanjang hari",
    9:  "🤩 Sangat segar & mudah berkonsentrasi",
    10: "⚡ Selalu segar, fokus maksimal & produktif",
}

def predict(inputs: dict):
    """Run clustering + classification given a flat dict of raw values."""
    # ── Cluster ──
    arr_clust = np.array([[inputs[f] for f in streamlit_features]])
    arr_clust_scaled = scaler_cluster.transform(arr_clust)
    clust_id  = str(model_cluster.predict(arr_clust_scaled)[0])
    clust_name = cluster_names.get(clust_id, f"Cluster {clust_id}")

    # ── Classification – build full feature vector ──
    arr_clf = np.zeros((1, len(all_features)))
    for i, feat in enumerate(all_features):
        if feat in inputs:
            arr_clf[0, i] = inputs[feat]
    arr_clf_scaled = scaler_clf.transform(arr_clf)
    clf_enc   = model_clf.predict(arr_clf_scaled)[0]
    clf_proba = model_clf.predict_proba(arr_clf_scaled)[0]
    clf_name  = le_performa.inverse_transform([clf_enc])[0]
    proba_map = {le_performa.classes_[i]: float(p) for i, p in enumerate(clf_proba)}
    return clust_name, clf_name, proba_map


def tips_for(cluster: str, perf: str):
    tips = {
        "Sehat": {
            "High":   ["Pertahankan rutinitas belajarmu! 🏆", "Waktu tidur sudah ideal — jaga konsistensinya 💤",
                       "Kurangi sedikit media sosial agar fokus makin tajam 📵"],
            "Medium": ["Coba tingkatkan jam belajar 30–60 menit per hari 📚",
                       "Rutinitas pagi yang konsisten bisa mendongkrak produktivitas ☀️",
                       "Pertahankan pola tidur sehatmu sebagai fondasi 💪"],
            "Low":    ["Gaya hidupmu sudah baik! Periksa strategi belajarmu 🔍",
                       "Coba teknik Pomodoro untuk sesi belajar yang lebih efektif ⏱️",
                       "Diskusikan kesulitan belajar dengan dosen atau teman 🤝"],
        },
        "Berisiko": {
            "High":   ["Performa bagus tapi gaya hidup berisiko — waspada burnout! ⚡",
                       "Tambah 1–2 jam tidur per malam untuk keberlangsungan jangka panjang 🌙",
                       "Kurangi screen time dan sisipkan istirahat aktif 🚶"],
            "Medium": ["Perbaiki pola tidur (target 7–8 jam) untuk meningkatkan konsentrasi 🛌",
                       "Batasi media sosial di jam belajar dengan mode fokus 📴",
                       "Olahraga ringan 20 menit/hari terbukti meningkatkan daya ingat 🏃"],
            "Low":    ["Prioritaskan perbaikan gaya hidup sebelum memacu belajar lebih keras ❤️",
                       "Konsultasi dengan konselor akademik atau tenaga kesehatan 🏥",
                       "Mulai dari target kecil: tidur lebih awal 30 menit saja 🎯"],
        },
        "Kurang Tidur": {
            "High":   ["Prestasi tinggi meski kurang tidur — tapi ini tidak berkelanjutan! ⏳",
                       "Kurang tidur kronis berdampak pada memori jangka panjang 🧠",
                       "Coba tidur lebih awal satu jam — performa bisa naik lagi 💡"],
            "Medium": ["Kurang tidur adalah hambatan utama fokusmu saat ini 😴",
                       "Target 7–8 jam tidur bisa meningkatkan nilai secara signifikan 📈",
                       "Hindari layar 1 jam sebelum tidur untuk kualitas tidur lebih baik 📱"],
            "Low":    ["Tidur yang cukup adalah langkah paling penting untukmu sekarang 🌟",
                       "Kurang tidur menurunkan kemampuan belajar hingga 40% — perbaiki dulu 💤",
                       "Buat jadwal tidur-bangun yang konsisten bahkan di akhir pekan 📅"],
        },
    }
    return tips.get(cluster, {}).get(perf, ["Terus semangat dan jaga kesehatanmu! 💪"])



st.markdown("""
<div class="hero">
  <h1>🎓 Analisis Gaya Hidup Mahasiswa</h1>
  <p>Analisis pola hidup & prediksi performa akademikmu dengan Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pill-row">
  <span class="pill">🤖 <b>Random Forest</b> Classifier</span>
  <span class="pill">📊 <b>K-Means</b> Pengelompokan (k=3)</span>
  <span class="pill">📦 Kerangka <b>CRISP-DM</b></span>
</div>
""", unsafe_allow_html=True)


with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📋 Data Profil Mahasiswa</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Usia (tahun)", min_value=17, max_value=35, value=20, step=1)
    with col2:
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan", "Lainnya"])
    with col3:
        part_time_job = st.selectbox("Kerja Paruh Waktu", ["Tidak", "Ya"])

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">⏰ Kebiasaan Harian</div>', unsafe_allow_html=True)

    sleep_hours        = st.slider("🛌 Jam Tidur per Hari", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
    study_hours        = st.slider("📚 Jam Belajar per Hari", min_value=0.0, max_value=12.0, value=4.0, step=0.5)
    social_media_hours = st.slider("📱 Jam Media Sosial per Hari", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
    netflix_hours      = st.slider("🎬 Jam Menonton Streaming per Hari", min_value=0.0, max_value=8.0, value=1.5, step=0.5)
    attendance_pct     = st.slider("📅 Persentase Kehadiran (%)", min_value=40, max_value=100, value=85, step=1)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="sec-header">🏃 Kesehatan & Lingkungan</div>', unsafe_allow_html=True)

    col4, col5 = st.columns(2)
    with col4:
        exercise_freq = st.slider("🏋️ Frekuensi Olahraga (hari/minggu)", 0, 7, 3)
        mental_health = st.slider("⚡ Energi & Fokus Harian (1–10)", 1, 10, 7)
        st.markdown(
            f'<div class="skala-hint">1 = sering lelah &amp; susah fokus &nbsp;·&nbsp; 10 = selalu segar &amp; fokus<br>'
            f'<b>Kamu:</b> {ENERGI_DESC[mental_health]}</div>',
            unsafe_allow_html=True
        )
    with col5:
        diet_quality  = st.selectbox("🥗 Kualitas Pola Makan",  ["Baik", "Cukup", "Kurang"])
        internet_qual = st.selectbox("🌐 Kualitas Internet",     ["Baik", "Sedang", "Buruk"])

    col6, col7 = st.columns(2)
    with col6:
        parent_edu      = st.selectbox("🎓 Pendidikan Orang Tua", ["S2/S3", "S1/D4", "SMA/Sederajat"])
    with col7:
        extracurricular = st.selectbox("🎭 Ekstrakurikuler",      ["Ya", "Tidak"])

    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("🔍 Analisis Sekarang", use_container_width=True)



if predict_btn:
    def safe_encode(le, val):
        classes = list(le.classes_)
        return classes.index(val) if val in classes else 0

    # Mapping pilihan Indonesia → nilai asli model
    gender_map       = {"Laki-laki": "Male", "Perempuan": "Female", "Lainnya": "Other"}
    job_map          = {"Tidak": "No", "Ya": "Yes"}
    diet_map         = {"Baik": "Good", "Cukup": "Fair", "Kurang": "Poor"}
    internet_map     = {"Baik": "Good", "Sedang": "Average", "Buruk": "Poor"}
    edu_map          = {"S2/S3": "Master", "S1/D4": "Bachelor", "SMA/Sederajat": "High School"}
    extra_map        = {"Ya": "Yes", "Tidak": "No"}

    inputs = {
        "age":                         float(age),
        "study_hours_per_day":         study_hours,
        "social_media_hours":          social_media_hours,
        "netflix_hours":               netflix_hours,
        "attendance_percentage":       float(attendance_pct),
        "sleep_hours":                 sleep_hours,
        "exercise_frequency":          float(exercise_freq),
        "mental_health_rating":        float(mental_health),
        "student_id_enc":              0,
        "gender_enc":                  safe_encode(le_dict["gender"],                       gender_map[gender]),
        "part_time_job_enc":           safe_encode(le_dict["part_time_job"],                job_map[part_time_job]),
        "diet_quality_enc":            safe_encode(le_dict["diet_quality"],                 diet_map[diet_quality]),
        "parental_education_level_enc":safe_encode(le_dict["parental_education_level"],     edu_map[parent_edu]),
        "internet_quality_enc":        safe_encode(le_dict["internet_quality"],             internet_map[internet_qual]),
        "extracurricular_participation_enc": safe_encode(le_dict["extracurricular_participation"], extra_map[extracurricular]),
    }

    cluster_name, perf_name, proba_map = predict(inputs)

    cluster_css = CLUSTER_CSS.get(cluster_name, "berisiko")
    perf_css    = PERF_CSS.get(perf_name, "medium")

    st.markdown("""<div style="height:1.2rem"></div>""", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎯 Hasil Analisis</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-grid">
      <div class="result-box box-{cluster_css}">
        <div class="result-label">Kelompok Gaya Hidup</div>
        <div class="result-value">{CLUSTER_ID.get(cluster_name, cluster_name)}</div>
        <div class="result-sub">Pengelompokan K-Means</div>
      </div>
      <div class="result-box box-{perf_css}">
        <div class="result-label">Prediksi Performa Akademik</div>
        <div class="result-value">{PERF_ID.get(perf_name, perf_name)}</div>
        <div class="result-sub">Pengklasifikasi Random Forest</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-header" style="margin-top:0.8rem">📊 Distribusi Probabilitas Performa</div>',
                unsafe_allow_html=True)
    label_map = {"High": "🟢 Tinggi", "Medium": "🟡 Sedang", "Low": "🔴 Rendah"}
    for label in ["High", "Medium", "Low"]:
        pct = proba_map.get(label, 0) * 100
        bar_color = BAR_COLORS[label]
        st.markdown(f"""
        <div class="prob-row">
          <div class="prob-label-row">
            <span>{label_map[label]}</span>
            <span style="font-weight:700">{pct:.1f}%</span>
          </div>
          <div class="prob-bar-bg">
            <div class="prob-bar-fill" style="width:{pct:.1f}%;background:{bar_color}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📌 Ringkasan Input</div>', unsafe_allow_html=True)
    summary_cols = st.columns(4)
    summary_data = [
        ("🛌 Tidur",        f"{sleep_hours} jam"),
        ("📚 Belajar",      f"{study_hours} jam"),
        ("📱 Sosmed",       f"{social_media_hours} jam"),
        ("📅 Kehadiran",    f"{attendance_pct}%"),
        ("🏋️ Olahraga",    f"{exercise_freq}×/minggu"),
        ("⚡ Energi",       f"{mental_health}/10"),
        ("🎬 Streaming",    f"{netflix_hours} jam"),
        ("🥗 Pola Makan",   diet_quality),
    ]
    for i, (label, val) in enumerate(summary_data):
        with summary_cols[i % 4]:
            st.markdown(f"""
            <div style="text-align:center;padding:0.6rem 0.3rem;background:rgba(100,140,210,0.1);
                 border-radius:10px;margin-bottom:0.5rem;border:1px solid rgba(100,140,210,0.2)">
              <div style="font-size:0.7rem;color:#5577aa;margin-bottom:0.15rem">{label}</div>
              <div style="font-size:0.95rem;font-weight:700;color:#1a3560">{val}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    tips = tips_for(cluster_name, perf_name)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💡 Rekomendasi Personal</div>', unsafe_allow_html=True)
    icons = ["✅", "💡", "🎯", "⚡", "🌟"]
    for i, tip in enumerate(tips):
        st.markdown(f"""
        <div class="tip-item">
          <span class="tip-icon">{icons[i % len(icons)]}</span>
          <span>{tip}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    cluster_desc = {
        "Sehat": ("Kamu memiliki pola hidup yang <b>seimbang</b> — tidur cukup, aktif belajar, "
                  "dan tidak terlalu banyak menghabiskan waktu di layar. Pertahankan!"),
        "Berisiko": ("Pola hidupmu menunjukkan beberapa <b>faktor risiko</b> — mungkin jam tidur kurang, "
                     "terlalu banyak screen time, atau kehadiran yang perlu ditingkatkan. Yuk diperbaiki!"),
        "Kurang Tidur": ("Jam tidurmu <b>di bawah rata-rata</b> yang sehat. Kurang tidur secara kronik "
                         "berdampak besar pada konsentrasi, memori, dan performa akademik."),
    }
    st.markdown(f"""
    <div style="background:rgba(74,127,212,0.08);border:1px solid rgba(74,127,212,0.25);
         border-radius:12px;padding:1rem 1.2rem;font-size:0.87rem;color:#1e3a6a;line-height:1.6">
      <b style="color:#2a5fa8">Tentang kelompok "{cluster_name}"</b><br>
      {cluster_desc.get(cluster_name, "")}
    </div>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 1rem;color:#8aaad8">
      <div style="font-size:3rem;margin-bottom:0.6rem">🔮</div>
      <div style="font-size:0.9rem">Isi form di atas lalu klik <b style="color:#4a7fd4">Analisis Sekarang</b></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:2rem 0 0.5rem;font-size:0.75rem;color:#8aaad8">
  Analisis Gaya Hidup Mahasiswa · Random Forest + K-Means · Dataset: Kaggle Student Habits
</div>
""", unsafe_allow_html=True)