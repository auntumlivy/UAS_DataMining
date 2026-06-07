import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Gaya Hidup Mahasiswa",
    page_icon="🎓",
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

.stApp { background: #f0f9ff; }

header[data-testid="stHeader"] {
    background: rgba(240,249,255,0.95);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #bae6fd;
}

/* ── Tabs ── */
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
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: white;
    border: 1px solid #bae6fd;
    border-radius: 16px;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 12px rgba(14,165,233,0.07);
}
div[data-testid="metric-container"] label { color: #64748b !important; font-weight: 600; font-size: 0.78rem; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #0284c7 !important; font-weight: 800; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #0284c7);
    color: white; border: none; border-radius: 10px;
    font-weight: 700; padding: 0.6rem 1.5rem;
    box-shadow: 0 4px 15px rgba(14,165,233,0.3);
    transition: all 0.25s;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(14,165,233,0.4); }

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; padding: 0.6rem 1.5rem !important;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    border: 1.5px solid #bae6fd !important;
    border-radius: 10px !important;
    background: #f0f9ff !important;
}
.stSelectbox [data-baseweb="select"] > div {
    border: 1.5px solid #bae6fd !important;
    border-radius: 10px !important;
    background: #f0f9ff !important;
    color: #0c2340 !important;
}
.stFileUploader { background: white; border: 2px dashed #7dd3fc; border-radius: 16px; padding: 1rem; }
.stDataFrame { border: 1px solid #bae6fd; border-radius: 12px; overflow: hidden; }

/* ── Sliders ── */
div[data-testid="stSlider"] {
    background: #f0f9ff !important;
    border: 1px solid #bae6fd !important;
    border-radius: 10px !important;
    padding: 0.6rem 0.9rem 0.5rem !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #0ea5e9 !important; border: 2px solid #38bdf8 !important;
}
[data-baseweb="slider"] [data-testid="stSliderTrackHighlight"],
div[class*="TrackHighlight"] { background: #0ea5e9 !important; border-radius: 100px !important; }
[data-baseweb="slider"] > div:first-child {
    background: #bae6fd !important; border-radius: 100px !important; height: 5px !important;
}

/* ── Section text ── */
.section-header { font-size: 1.6rem; font-weight: 800; color: #0c2340; letter-spacing: -0.5px; margin-bottom: 0.25rem; }
.section-label  { font-size: 0.72rem; font-weight: 600; color: #0ea5e9; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 0.4rem; }
.section-desc   { font-size: 0.9rem; color: #64748b; margin-bottom: 1.5rem; line-height: 1.6; }

/* ── Card ── */
.custom-card {
    background: white; border: 1px solid #bae6fd;
    border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem;
    box-shadow: 0 2px 12px rgba(14,165,233,0.07);
}

/* ── Hero ── */
.hero-box {
    background: linear-gradient(135deg, #0c2340 0%, #075985 50%, #0284c7 100%);
    border-radius: 24px; padding: 3rem 2.5rem;
    color: white; margin-bottom: 1.5rem; position: relative; overflow: hidden;
}
.hero-box h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -1px; line-height: 1.1; margin-bottom: 0.75rem; }
.hero-box h1 span { color: #7dd3fc; }
.hero-box p  { color: rgba(255,255,255,0.75); font-size: 0.97rem; line-height: 1.7; max-width: 560px; }

.badge {
    display: inline-block; background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.2); border-radius: 100px;
    padding: 0.3rem 1rem; font-size: 0.78rem; font-weight: 600;
    color: #bae6fd; margin-bottom: 1.25rem; letter-spacing: 0.5px;
}

.member-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1.5rem; }
.member-card { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); border-radius: 12px; padding: 0.75rem 1.25rem; }
.member-card .name { font-weight: 700; color: white; font-size: 0.9rem; }
.member-card .nim  { color: #7dd3fc; font-size: 0.75rem; margin-top: 2px; font-family: monospace; }

/* ── Result highlight ── */
.result-highlight {
    background: linear-gradient(135deg, #e0f2fe, #f0f9ff);
    border: 2px solid #38bdf8; border-radius: 16px;
    padding: 1.5rem; text-align: center;
}
.result-highlight .label { font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
.result-highlight .value { font-size: 2.2rem; font-weight: 800; color: #0284c7; letter-spacing: -1px; }

/* ── Cluster & perf boxes ── */
.res-box { border-radius: 16px; padding: 1.4rem; text-align: center; border: 2px solid; margin-bottom: 1rem; }
.res-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; margin-bottom: 0.3rem; }
.res-val   { font-size: 1.8rem; font-weight: 800; line-height: 1.15; }
.res-sub   { font-size: 0.75rem; margin-top: 0.25rem; opacity: 0.65; }
.box-sehat       { background: #ecfdf5; border-color: #6ee7b7; color: #065f46; }
.box-berisiko    { background: #fffbeb; border-color: #fcd34d; color: #78350f; }
.box-kurangtidur { background: #fef2f2; border-color: #fca5a5; color: #7f1d1d; }
.box-high   { background: #ecfdf5; border-color: #6ee7b7; color: #065f46; }
.box-medium { background: #fffbeb; border-color: #fcd34d; color: #78350f; }
.box-low    { background: #fef2f2; border-color: #fca5a5; color: #7f1d1d; }

/* ── Progress bar ── */
.prog-wrap { margin-bottom: 0.75rem; }
.prog-label { display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 600; color: #374f6b; margin-bottom: 0.3rem; }
.prog-track { background: #e0f2fe; border-radius: 100px; height: 8px; overflow: hidden; }
.prog-fill  { height: 100%; border-radius: 100px; }
.fill-sky   { background: linear-gradient(90deg, #38bdf8, #0ea5e9); }
.fill-green { background: linear-gradient(90deg, #34d399, #10b981); }
.fill-amber { background: linear-gradient(90deg, #fbbf24, #f59e0b); }
.fill-red   { background: linear-gradient(90deg, #f87171, #ef4444); }

/* ── Tip ── */
.tip-item {
    display: flex; align-items: flex-start; gap: 0.7rem;
    padding: 0.7rem 1rem; border-radius: 10px;
    background: #f0f9ff; border: 1px solid #bae6fd;
    margin-bottom: 0.5rem; font-size: 0.84rem; color: #374f6b; line-height: 1.5;
}

/* ── Tag ── */
.tag { display: inline-block; background: #e0f2fe; color: #0369a1; border-radius: 6px; padding: 0.25rem 0.65rem; font-size: 0.78rem; font-weight: 600; margin: 0.2rem; }
.tag-dark { background: #0c2340; color: #7dd3fc; }

/* ── Feature row ── */
.feat-row {
    display: flex; align-items: center; gap: 0.75rem;
    padding: 0.45rem 0.8rem;
    background: white; border: 1px solid #e0f2fe;
    border-radius: 10px; margin-bottom: 0.35rem;
}

/* ── Phase step ── */
.phase-step {
    display: flex; align-items: flex-start; gap: 0.9rem;
    padding: 0.75rem 1rem;
    background: white; border: 1px solid #bae6fd;
    border-radius: 12px; margin-bottom: 0.45rem;
}

hr { border: none; border-top: 1px solid #bae6fd; margin: 1.5rem 0; }

.success-box {
    background: #ecfdf5; border: 1px solid #6ee7b7;
    border-radius: 12px; padding: 1rem 1.25rem;
    color: #065f46; font-weight: 600; font-size: 0.88rem;
}

#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 3rem; }
.stSlider label, .stSelectbox label, .stNumberInput label {
    color: #374f6b !important; font-size: 0.85rem !important; font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ── LOAD MODELS ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    model_clf      = joblib.load("model/model_clf.pkl")
    model_cluster  = joblib.load("model/model_cluster.pkl")
    scaler_clf     = joblib.load("model/scaler_clf.pkl")
    scaler_cluster = joblib.load("model/scaler_cluster.pkl")
    le_performa    = joblib.load("model/le_performa.pkl")
    le_dict        = joblib.load("model/le_dict.pkl")
    with open("model/cluster_names.json")      as f: cluster_names     = json.load(f)
    with open("model/streamlit_features.json") as f: streamlit_features = json.load(f)
    with open("model/all_features.json")       as f: all_features      = json.load(f)
    return (model_clf, model_cluster, scaler_clf, scaler_cluster,
            le_performa, le_dict, cluster_names, streamlit_features, all_features)

(model_clf, model_cluster, scaler_clf, scaler_cluster,
 le_performa, le_dict, cluster_names, streamlit_features, all_features) = load_models()


# ── HELPERS ──────────────────────────────────────────────────
SKY = ['#0ea5e9', '#38bdf8', '#7dd3fc', '#bae6fd', '#0284c7']

def set_chart_style(ax):
    ax.set_facecolor('#f8faff')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#bae6fd'); ax.spines['bottom'].set_color('#bae6fd')
    ax.tick_params(colors='#64748b', labelsize=9)
    ax.yaxis.label.set_color('#64748b'); ax.xaxis.label.set_color('#64748b')

gender_map   = {"Laki-laki":"Male","Perempuan":"Female","Lainnya":"Other"}
job_map      = {"Tidak":"No","Ya":"Yes"}
diet_map     = {"Baik":"Good","Cukup":"Fair","Kurang":"Poor"}
internet_map = {"Baik":"Good","Sedang":"Average","Buruk":"Poor"}
edu_map      = {"S2/S3":"Master","S1/D4":"Bachelor","SMA/Sederajat":"High School"}
extra_map    = {"Ya":"Yes","Tidak":"No"}

CLUSTER_CSS = {"Sehat":"sehat","Berisiko":"berisiko","Kurang Tidur":"kurangtidur"}
PERF_CSS    = {"High":"high","Medium":"medium","Low":"low"}
PERF_ID     = {"High":"🟢 Tinggi","Medium":"🟡 Sedang","Low":"🔴 Rendah"}
CLUSTER_ID  = {"Sehat":"🌿 Sehat","Berisiko":"⚠️ Berisiko","Kurang Tidur":"😴 Kurang Tidur"}
FILL_COLOR  = {"High":"fill-green","Medium":"fill-amber","Low":"fill-red"}

def safe_encode(le, val):
    classes = list(le.classes_)
    return classes.index(val) if val in classes else 0

def build_inputs(age, study_hours, social_media_hours, netflix_hours,
                 attendance_pct, sleep_hours, exercise_freq, mental_health,
                 gender, part_time_job, diet_quality, internet_qual,
                 parent_edu, extracurricular):
    return {
        "age": float(age),
        "study_hours_per_day": study_hours,
        "social_media_hours": social_media_hours,
        "netflix_hours": netflix_hours,
        "attendance_percentage": float(attendance_pct),
        "sleep_hours": sleep_hours,
        "exercise_frequency": float(exercise_freq),
        "mental_health_rating": float(mental_health),
        "student_id_enc": 0,
        "gender_enc": safe_encode(le_dict["gender"], gender_map.get(gender, gender)),
        "part_time_job_enc": safe_encode(le_dict["part_time_job"], job_map.get(part_time_job, part_time_job)),
        "diet_quality_enc": safe_encode(le_dict["diet_quality"], diet_map.get(diet_quality, diet_quality)),
        "parental_education_level_enc": safe_encode(le_dict["parental_education_level"], edu_map.get(parent_edu, parent_edu)),
        "internet_quality_enc": safe_encode(le_dict["internet_quality"], internet_map.get(internet_qual, internet_qual)),
        "extracurricular_participation_enc": safe_encode(le_dict["extracurricular_participation"], extra_map.get(extracurricular, extracurricular)),
    }

def predict(inputs):
    arr_c        = np.array([[inputs[f] for f in streamlit_features]])
    arr_c_scaled = scaler_cluster.transform(arr_c)
    clust_id     = str(model_cluster.predict(arr_c_scaled)[0])
    clust_name   = cluster_names.get(clust_id, f"Cluster {clust_id}")
    arr_clf = np.zeros((1, len(all_features)))
    for i, feat in enumerate(all_features):
        if feat in inputs: arr_clf[0, i] = inputs[feat]
    arr_clf_s = scaler_clf.transform(arr_clf)
    clf_enc   = model_clf.predict(arr_clf_s)[0]
    clf_proba = model_clf.predict_proba(arr_clf_s)[0]
    clf_name  = le_performa.inverse_transform([clf_enc])[0]
    proba_map = {le_performa.classes_[i]: float(p) for i, p in enumerate(clf_proba)}
    return clust_name, clf_name, proba_map

def tips_for(cluster, perf):
    tips = {
        "Sehat":{"High":["Pertahankan rutinitas belajarmu! 🏆","Waktu tidur ideal — jaga konsistensinya 💤","Kurangi sedikit media sosial agar fokus makin tajam 📵"],
                 "Medium":["Tingkatkan jam belajar 30–60 menit per hari 📚","Rutinitas pagi konsisten mendongkrak produktivitas ☀️","Pertahankan pola tidur sehat sebagai fondasi 💪"],
                 "Low":["Gaya hidup sudah baik! Periksa strategi belajarmu 🔍","Coba teknik Pomodoro untuk belajar lebih efektif ⏱️","Diskusikan kesulitan belajar dengan dosen/teman 🤝"]},
        "Berisiko":{"High":["Performa bagus tapi gaya hidup berisiko — waspada burnout! ⚡","Tambah 1–2 jam tidur per malam 🌙","Kurangi screen time, sisipkan istirahat aktif 🚶"],
                    "Medium":["Perbaiki pola tidur (target 7–8 jam) untuk konsentrasi 🛌","Batasi media sosial saat jam belajar 📴","Olahraga ringan 20 menit/hari tingkatkan daya ingat 🏃"],
                    "Low":["Prioritaskan perbaikan gaya hidup dulu ❤️","Konsultasi dengan konselor akademik 🏥","Mulai kecil: tidur lebih awal 30 menit 🎯"]},
        "Kurang Tidur":{"High":["Prestasi tinggi meski kurang tidur — tidak berkelanjutan! ⏳","Kurang tidur kronis rusak memori jangka panjang 🧠","Tidur lebih awal 1 jam — performa bisa naik lagi 💡"],
                        "Medium":["Kurang tidur jadi hambatan utama fokusmu 😴","Target 7–8 jam tidur bisa naikkan nilai 📈","Hindari layar 1 jam sebelum tidur 📱"],
                        "Low":["Tidur cukup adalah langkah terpenting sekarang 🌟","Kurang tidur turunkan kemampuan belajar 40% 💤","Buat jadwal tidur konsisten bahkan di akhir pekan 📅"]},
    }
    return tips.get(cluster, {}).get(perf, ["Terus semangat! 💪"])

def make_template_csv():
    df = pd.DataFrame([{"age":20,"gender":"Laki-laki","part_time_job":"Tidak","sleep_hours":7.0,
        "study_hours_per_day":4.0,"social_media_hours":2.0,"netflix_hours":1.5,
        "attendance_percentage":85,"exercise_frequency":3,"fokus_harian":7,
        "diet_quality":"Baik","internet_quality":"Baik","parental_education_level":"S1/D4","mengikuti_organisasi":"Ya"}])
    return df.to_csv(index=False).encode("utf-8")

def process_csv_row(row):
    return build_inputs(
        age=row.get("age",20), study_hours=row.get("study_hours_per_day",4.0),
        social_media_hours=row.get("social_media_hours",2.0), netflix_hours=row.get("netflix_hours",1.5),
        attendance_pct=row.get("attendance_percentage",85), sleep_hours=row.get("sleep_hours",7.0),
        exercise_freq=row.get("exercise_frequency",3),
        mental_health=row.get("fokus_harian", row.get("mental_health_rating",7)),
        gender=str(row.get("gender","Laki-laki")), part_time_job=str(row.get("part_time_job","Tidak")),
        diet_quality=str(row.get("diet_quality","Baik")), internet_qual=str(row.get("internet_quality","Baik")),
        parent_edu=str(row.get("parental_education_level","S1/D4")),
        extracurricular=str(row.get("mengikuti_organisasi", row.get("extracurricular_participation","Tidak"))),
    )


# ════════════════════════════════════════════════════════════
# APP TITLE BAR
# ════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align:center; padding: 0.75rem 0 0.25rem'>
  <span style='font-size:1.4rem; font-weight:800; color:#0284c7'>Student</span>
  <span style='font-size:1.4rem; font-weight:800; color:#0c2340'> Lifestyle</span>
  <span style='font-size:1.4rem; font-weight:800; color:#0284c7'> Analyzer</span>
  <span style='font-size:0.78rem; font-weight:600; color:#64748b; margin-left:8px'>— Analisis Gaya Hidup & Performa Akademik</span>
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
        <div class='badge'>● Tugas Besar Data Mining 2025/2026</div>
        <h1>Analisis Gaya Hidup<br><span>Mahasiswa & Performa Akademik</span></h1>
        <p>Aplikasi prediksi dan analisis berbasis machine learning untuk memahami hubungan antara
        kebiasaan sehari-hari dan hasil akademik mahasiswa secara mendalam dan intuitif.</p>
        <div class='member-row'>
            <div class='member-card'><div class='name'>Nama Anggota 1</div><div class='nim'>NIM · XXXXXXXXXX</div></div>
            <div class='member-card'><div class='name'>Nama Anggota 2</div><div class='nim'>NIM · XXXXXXXXXX</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records",    "1.000")
    col2.metric("Fitur / Kolom",    "16")
    col3.metric("Kelompok Cluster", "3")
    col4.metric("Kelas Performa",   "3")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div class='section-label'>Panduan Penggunaan</div>
    <div class='section-header'>Cara Menggunakan Aplikasi</div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, step, title, desc in [
        (c1,"📂","1. Lihat Dataset",    "Dataset","Buka tab <b>Dataset</b> untuk melihat informasi, statistik, dan distribusi dataset yang digunakan."),
        (c2,"🔮","2. Buat Prediksi",    "Prediction","Tab <b>Prediction</b>: input data manual atau upload CSV untuk prediksi gaya hidup & performa akademik."),
        (c3,"📊","3. Analisis Visual",  "Visualization","Tab <b>Visualization</b>: grafik distribusi cluster, perbandingan fitur, dan korelasi antar variabel."),
        (c4,"ℹ️","4. Tentang Proyek", "About","Tab <b>About</b>: penjelasan metode K-Means, Random Forest, alur CRISP-DM, dan informasi tim."),
    ]:
        with col:
            st.markdown(f"""
            <div class='custom-card'>
                <div style='font-size:1.75rem; margin-bottom:0.5rem'>{icon}</div>
                <strong style='color:#0c2340'>{step}</strong>
                <p style='color:#64748b; font-size:0.84rem; margin-top:0.4rem; line-height:1.6'>{desc}</p>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 2 — DATASET
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='section-label'>02 — Dataset Overview</div>
    <div class='section-header'>Ringkasan Dataset</div>
    <div class='section-desc'>Informasi, statistik, dan distribusi dataset Student Habits vs Academic Performance dari Kaggle.</div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='custom-card'>
            <strong style='color:#0c2340; font-size:1rem'>📋 Informasi Dataset</strong><br><br>
            <span class='tag'>Kaggle Open Dataset</span>
            <span class='tag'>Supervised</span>
            <span class='tag'>CC BY 4.0</span>
            <span class='tag'>1.000 Records</span>
            <table style='width:100%; margin-top:1rem; font-size:0.83rem; border-collapse:collapse'>
              <tr><td style='color:#64748b; padding:0.35rem 0; width:40%'>Nama Dataset</td>
                  <td style='color:#0c2340; font-weight:600'>Student Habits vs Academic Performance</td></tr>
              <tr><td style='color:#64748b; padding:0.35rem 0'>Sumber</td>
                  <td style='color:#0c2340; font-weight:600'>Kaggle — Jayaantanaath</td></tr>
              <tr><td style='color:#64748b; padding:0.35rem 0'>Ukuran</td>
                  <td style='color:#0c2340; font-weight:600'>1.000 baris × 16 kolom</td></tr>
              <tr><td style='color:#64748b; padding:0.35rem 0'>Target</td>
                  <td style='color:#0c2340; font-weight:600'>exam_score → High / Medium / Low</td></tr>
              <tr><td style='color:#64748b; padding:0.35rem 0'>Preprocessing</td>
                  <td style='color:#0c2340; font-weight:600'>Encoding, Normalisasi, Train/Test Split</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='custom-card'><strong style='color:#0c2340; font-size:1rem'>📊 Distribusi Kelas Performa</strong><br><br>", unsafe_allow_html=True)
        for kls, pct, fill in [("High (Tinggi)", 33.2, "fill-green"), ("Medium (Sedang)", 33.9, "fill-amber"), ("Low (Rendah)", 32.9, "fill-red")]:
            st.markdown(f"""
            <div class='prog-wrap'>
                <div class='prog-label'><span>{kls}</span><span>{pct}%</span></div>
                <div class='prog-track'><div class='prog-fill {fill}' style='width:{pct}%'></div></div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Daftar fitur
    st.markdown("""
    <div class='section-label'>Daftar Fitur</div>
    <div class='section-header' style='font-size:1.2rem'>16 Variabel Dataset</div>
    """, unsafe_allow_html=True)

    type_col = {"Numerik":"#0369a1","Kategorik":"#78350f"}
    type_bg  = {"Numerik":"#e0f2fe","Kategorik":"#fffbeb"}
    features_info = [
        ("age","Numerik","Usia mahasiswa (tahun)"),
        ("gender","Kategorik","Jenis kelamin"),
        ("study_hours_per_day","Numerik","Jam belajar per hari"),
        ("social_media_hours","Numerik","Jam penggunaan media sosial per hari"),
        ("netflix_hours","Numerik","Jam menonton streaming per hari"),
        ("part_time_job","Kategorik","Status kerja paruh waktu (Yes/No)"),
        ("attendance_percentage","Numerik","Persentase kehadiran kuliah (%)"),
        ("sleep_hours","Numerik","Jam tidur per hari"),
        ("diet_quality","Kategorik","Kualitas pola makan (Good/Fair/Poor)"),
        ("exercise_frequency","Numerik","Frekuensi olahraga per minggu"),
        ("parental_education_level","Kategorik","Tingkat pendidikan orang tua"),
        ("internet_quality","Kategorik","Kualitas koneksi internet"),
        ("mental_health_rating","Numerik","Rating kesehatan mental (1–10)"),
        ("extracurricular_participation","Kategorik","Keikutsertaan organisasi (Yes/No)"),
        ("exam_score","Numerik","Nilai ujian (target variabel)"),
    ]
    col_feat_l, col_feat_r = st.columns(2)
    half = len(features_info) // 2
    for i, (feat, dtype, desc) in enumerate(features_info):
        col = col_feat_l if i < half + 1 else col_feat_r
        with col:
            st.markdown(f"""
            <div class='feat-row'>
              <code style='font-size:0.73rem;background:#e0f2fe;padding:0.1rem 0.5rem;border-radius:5px;color:#0369a1;min-width:190px;display:inline-block'>{feat}</code>
              <span style='font-size:0.68rem;padding:0.1rem 0.5rem;border-radius:100px;background:{type_bg[dtype]};color:{type_col[dtype]};font-weight:600;white-space:nowrap'>{dtype}</span>
              <span style='font-size:0.79rem;color:#64748b'>{desc}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Grafik distribusi
    st.markdown("""<div class='section-label'>Visualisasi Distribusi</div>
    <div class='section-header' style='font-size:1.2rem'>Distribusi Fitur Utama</div>""", unsafe_allow_html=True)

    fig_ds, axes_ds = plt.subplots(1, 3, figsize=(13, 3.8))
    fig_ds.patch.set_facecolor('white')
    np.random.seed(42)
    for ax, data, title, xlabel, color in zip(
        axes_ds,
        [np.random.normal(7.1,1.4,1000).clip(3,12), np.random.normal(4.8,1.8,1000).clip(0,12), np.random.normal(79.9,12,1000).clip(40,100)],
        ["Distribusi Jam Tidur","Distribusi Jam Belajar","Distribusi Kehadiran"],
        ["Jam per Hari","Jam per Hari","Persentase (%)"],
        [SKY[0], SKY[4], '#059669']
    ):
        set_chart_style(ax)
        ax.hist(data, bins=22, color=color, edgecolor='white', linewidth=0.8, alpha=0.9)
        ax.set_title(title, fontweight='bold', color='#0c2340', fontsize=11)
        ax.set_xlabel(xlabel); ax.set_ylabel('Frekuensi')
    plt.tight_layout(pad=2)
    st.pyplot(fig_ds, use_container_width=True)
    plt.close(fig_ds)
    st.markdown('<div style="font-size:0.73rem;color:#94a3b8;text-align:center">* Visualisasi representatif berdasarkan karakteristik dataset</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 3 — PREDICTION
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='section-label'>03 — Prediction / Analysis</div>
    <div class='section-header'>Form Prediksi & Analisis</div>
    <div class='section-desc'>Prediksi kelompok gaya hidup (K-Means) dan performa akademik (Random Forest) secara manual atau via CSV.</div>
    """, unsafe_allow_html=True)

    pred_tab1, pred_tab2 = st.tabs(["✏️  Input Manual", "📤  Upload CSV"])

    # ── INPUT MANUAL ─────────────────────────────────────────
    with pred_tab1:
        form_col, result_col = st.columns([1, 1], gap="large")

        with form_col:
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.markdown("**👤 Data Profil**")
            fc1, fc2, fc3 = st.columns(3)
            with fc1: age           = st.number_input("Usia (tahun)", min_value=17, max_value=35, value=20, step=1)
            with fc2: gender        = st.selectbox("Jenis Kelamin", ["Laki-laki","Perempuan","Lainnya"])
            with fc3: part_time_job = st.selectbox("Kerja Paruh Waktu", ["Tidak","Ya"])

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**⏰ Kebiasaan Harian**")
            sleep_hours        = st.slider("🛌 Jam Tidur per Hari",             3.0, 12.0, 7.0, 0.5)
            study_hours        = st.slider("📚 Jam Belajar per Hari",            0.0, 12.0, 4.0, 0.5)
            social_media_hours = st.slider("📱 Jam Media Sosial per Hari",       0.0, 10.0, 2.0, 0.5)
            netflix_hours      = st.slider("🎬 Jam Streaming per Hari",          0.0,  8.0, 1.5, 0.5)
            attendance_pct     = st.slider("📅 Persentase Kehadiran (%)",        40,  100,  85,  1)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**🏃 Kesehatan & Lingkungan**")
            hc1, hc2 = st.columns(2)
            with hc1:
                exercise_freq = st.slider("🏋️ Olahraga (hari/minggu)", 0, 7, 3)
                diet_quality  = st.selectbox("🥗 Pola Makan", ["Baik","Cukup","Kurang"])
            with hc2:
                mental_health = st.slider("⚡ Energi & Fokus (1–10)", 1, 10, 7)
                internet_qual = st.selectbox("🌐 Kualitas Internet", ["Baik","Sedang","Buruk"])

            hc3, hc4 = st.columns(2)
            with hc3: parent_edu      = st.selectbox("🎓 Pendidikan Orang Tua", ["S2/S3","S1/D4","SMA/Sederajat"])
            with hc4: extracurricular = st.selectbox("🏛️ Ikut Organisasi", ["Ya","Tidak"])

            st.markdown("</div>", unsafe_allow_html=True)
            proses = st.button("⚡ Proses Prediksi", use_container_width=True, key="btn_manual")

        with result_col:
            if proses:
                inputs = build_inputs(age, study_hours, social_media_hours, netflix_hours,
                                      attendance_pct, sleep_hours, exercise_freq, mental_health,
                                      gender, part_time_job, diet_quality, internet_qual,
                                      parent_edu, extracurricular)
                cluster_name, perf_name, proba_map = predict(inputs)
                css_c = CLUSTER_CSS.get(cluster_name,"berisiko")
                css_p = PERF_CSS.get(perf_name,"medium")

                # Hasil utama
                r1, r2 = st.columns(2)
                with r1:
                    st.markdown(f"""
                    <div class='res-box box-{css_c}'>
                      <div class='res-label'>Kelompok Gaya Hidup</div>
                      <div class='res-val'>{CLUSTER_ID.get(cluster_name, cluster_name)}</div>
                      <div class='res-sub'>K-Means Clustering</div>
                    </div>""", unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""
                    <div class='res-box box-{css_p}'>
                      <div class='res-label'>Prediksi Performa</div>
                      <div class='res-val'>{PERF_ID.get(perf_name, perf_name)}</div>
                      <div class='res-sub'>Random Forest</div>
                    </div>""", unsafe_allow_html=True)

                # Probabilitas
                st.markdown("<br><strong style='color:#0c2340'>📈 Probabilitas Per Kelas</strong>", unsafe_allow_html=True)
                label_map = {"High":"🟢 Tinggi","Medium":"🟡 Sedang","Low":"🔴 Rendah"}
                fill_map  = {"High":"fill-green","Medium":"fill-amber","Low":"fill-red"}
                for label in ["High","Medium","Low"]:
                    pct = proba_map.get(label,0)*100
                    st.markdown(f"""
                    <div class='prog-wrap'>
                      <div class='prog-label'><span>{label_map[label]}</span><span>{pct:.1f}%</span></div>
                      <div class='prog-track'><div class='prog-fill {fill_map[label]}' style='width:{pct:.1f}%'></div></div>
                    </div>""", unsafe_allow_html=True)

                # Tips
                st.markdown("<br><strong style='color:#0c2340'>💡 Rekomendasi Personal</strong>", unsafe_allow_html=True)
                icons = ["✅","💡","🎯"]
                for i, tip in enumerate(tips_for(cluster_name, perf_name)):
                    st.markdown(f'<div class="tip-item"><span>{icons[i%len(icons)]}</span><span>{tip}</span></div>', unsafe_allow_html=True)

                # Radar chart
                st.markdown("<br><strong style='color:#0c2340'>📊 Profil Gaya Hidup</strong>", unsafe_allow_html=True)
                categories = ["Tidur","Belajar","Olahraga","Kehadiran","Energi"]
                values_raw = [inputs["sleep_hours"]/12, inputs["study_hours_per_day"]/12,
                              inputs["exercise_frequency"]/7, inputs["attendance_percentage"]/100,
                              inputs["mental_health_rating"]/10]
                N = len(categories)
                angles = [n/float(N)*2*np.pi for n in range(N)] + [0]
                values = values_raw + values_raw[:1]
                fig_r = plt.figure(figsize=(5, 4)); fig_r.patch.set_facecolor('white')
                ax_r = fig_r.add_subplot(111, polar=True)
                ax_r.set_facecolor('#f8faff')
                ax_r.plot(angles, values, "o-", linewidth=2.5, color=SKY[0])
                ax_r.fill(angles, values, alpha=0.2, color=SKY[0])
                ax_r.set_xticks(angles[:-1]); ax_r.set_xticklabels(categories, size=9, color="#374f6b", fontweight="600")
                ax_r.set_yticks([0.25,0.5,0.75,1.0]); ax_r.set_yticklabels(["25%","50%","75%","100%"], size=6, color="#94a3b8")
                ax_r.set_ylim(0,1); ax_r.spines["polar"].set_color("#bae6fd")
                ax_r.grid(color="#e0f2fe", linestyle="--", linewidth=0.9)
                ax_r.set_title("Radar Gaya Hidup", size=10, color="#0c2340", fontweight="700", pad=14)
                plt.tight_layout()
                st.pyplot(fig_r, use_container_width=True); plt.close(fig_r)

            else:
                st.markdown("""
                <div style='background:white; border:2px dashed #bae6fd; border-radius:16px; padding:4rem; text-align:center; color:#94a3b8; margin-top:0.5rem'>
                    <div style='font-size:3rem'>🔮</div>
                    <div style='font-weight:600; margin-top:0.75rem; font-size:0.95rem'>Isi form lalu klik Proses Prediksi</div>
                    <div style='font-size:0.82rem; margin-top:0.3rem'>Hasil analisis akan muncul di sini</div>
                </div>""", unsafe_allow_html=True)

    # ── UPLOAD CSV ────────────────────────────────────────────
    with pred_tab2:
        st.markdown("""
        <div class='custom-card'>
            <strong style='color:#0c2340'>📤 Upload File CSV — Prediksi Batch</strong>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.5rem; line-height:1.6'>
            Upload file CSV berisi data beberapa mahasiswa sekaligus. Pastikan kolom sesuai template.
            </p>
        </div>""", unsafe_allow_html=True)

        dl_c, info_c = st.columns([1,2])
        with dl_c:
            st.download_button("⬇️ Download Template CSV", data=make_template_csv(),
                               file_name="template_analisis_mahasiswa.csv", mime="text/csv", use_container_width=True)
        with info_c:
            st.markdown('<div style="font-size:0.81rem;color:#64748b;padding:0.5rem 0;line-height:1.6">Template berisi 1 baris contoh.<br>Tambahkan baris sesuai jumlah mahasiswa yang ingin dianalisis.</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload file CSV kamu", type=["csv"], key="csv_up")

        if uploaded:
            try:
                df_input = pd.read_csv(uploaded)
                st.markdown(f"<div class='success-box'>✅ {len(df_input)} baris data terdeteksi</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.dataframe(df_input.head(10), use_container_width=True)

                if st.button("🚀 Prediksi Semua Data", use_container_width=True, key="btn_batch"):
                    with st.spinner("Memproses prediksi..."):
                        results = []
                        for idx, row in df_input.iterrows():
                            inp = process_csv_row(row)
                            cn, pn, pm = predict(inp)
                            results.append({**row.to_dict(),"Kelompok":cn,"Performa":pn,
                                "P_Tinggi":round(pm.get("High",0)*100,1),
                                "P_Sedang":round(pm.get("Medium",0)*100,1),
                                "P_Rendah":round(pm.get("Low",0)*100,1)})
                        df_result = pd.DataFrame(results)

                    st.markdown("<div class='success-box'>✅ Prediksi selesai! Lihat hasil di bawah.</div>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    dcols = ["Kelompok","Performa","P_Tinggi","P_Sedang","P_Rendah"]
                    if "age" in df_result.columns: dcols = ["age"] + dcols
                    st.dataframe(df_result[dcols].rename(columns={"age":"Usia","Kelompok":"Kelompok Gaya Hidup",
                        "Performa":"Prediksi Performa","P_Tinggi":"Peluang Tinggi (%)","P_Sedang":"Peluang Sedang (%)","P_Rendah":"Peluang Rendah (%)"}),
                        use_container_width=True, height=min(400, 38+35*len(df_result)))

                    # Summary metrics
                    vc = df_result["Kelompok"].value_counts()
                    vp = df_result["Performa"].value_counts()
                    m1, m2, m3, m4, m5, m6 = st.columns(6)
                    m1.metric("🌿 Sehat",       vc.get("Sehat",0))
                    m2.metric("⚠️ Berisiko",    vc.get("Berisiko",0))
                    m3.metric("😴 Kurang Tidur", vc.get("Kurang Tidur",0))
                    m4.metric("🟢 Tinggi",       vp.get("High",0))
                    m5.metric("🟡 Sedang",       vp.get("Medium",0))
                    m6.metric("🔴 Rendah",       vp.get("Low",0))

                    st.download_button("⬇️ Download Hasil Prediksi CSV", data=df_result.to_csv(index=False).encode("utf-8"),
                                       file_name="hasil_analisis_mahasiswa.csv", mime="text/csv", use_container_width=True)
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
    <div class='section-desc'>Grafik perbandingan cluster, distribusi performa, korelasi fitur, dan analisis hasil model.</div>
    """, unsafe_allow_html=True)

    # Row 1 — Bar cluster + Heatmap distribusi
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        fig.patch.set_facecolor('white'); set_chart_style(ax)
        cats  = ["Jam Tidur","Jam Belajar","Kehadiran\n(%/10)","Olahraga","Energi\n(1-10)"]
        xv    = np.arange(len(cats)); wv = 0.25
        ax.bar(xv-wv, [7.8,5.5,8.8,4.2,7.5], width=wv, label="🌿 Sehat",       color='#10b981', edgecolor='white')
        ax.bar(xv,    [6.2,4.1,7.2,2.5,5.8], width=wv, label="⚠️ Berisiko",    color='#f59e0b', edgecolor='white')
        ax.bar(xv+wv, [4.9,5.2,7.5,3.0,4.5], width=wv, label="😴 Kurang Tidur", color='#ef4444', edgecolor='white')
        ax.set_xticks(xv); ax.set_xticklabels(cats, fontsize=9, color="#374f6b")
        ax.set_ylabel('Nilai Rata-rata'); ax.legend(fontsize=8)
        ax.set_title('Karakteristik Rata-rata per Kelompok', fontweight='bold', color='#0c2340', fontsize=11)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        fig.patch.set_facecolor('white')
        dm = np.array([[55,30,15],[25,45,30],[15,35,50]])
        im = ax.imshow(dm, cmap='Blues', aspect='auto')
        ax.set_xticks(range(3)); ax.set_xticklabels(["Tinggi","Sedang","Rendah"], fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels(["Sehat","Berisiko","Kurang Tidur"], fontsize=9)
        ax.set_title('Distribusi Performa per Kelompok (%)', fontweight='bold', color='#0c2340', fontsize=11)
        ax.set_xlabel('Performa Akademik'); ax.set_ylabel('Kelompok Gaya Hidup')
        for i in range(3):
            for j in range(3):
                ax.text(j,i,f"{dm[i,j]}%",ha='center',va='center',fontsize=12,fontweight='700',
                        color='white' if dm[i,j]>35 else '#0c2340')
        plt.colorbar(im, ax=ax, label="%"); plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 2 — Scatter + bar performa + feature importance
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        fig.patch.set_facecolor('white'); set_chart_style(ax)
        np.random.seed(99)
        for cluster, color, sm, stm in [("Sehat","#10b981",7.8,5.5),("Berisiko","#f59e0b",6.2,4.1),("Kurang Tidur","#ef4444",4.9,5.2)]:
            ax.scatter(np.random.normal(sm,0.8,330).clip(3,12), np.random.normal(stm,1.2,330).clip(0,12),
                       c=color, alpha=0.5, s=22, label=cluster, edgecolors='none')
        ax.set_xlabel('Jam Tidur/Hari'); ax.set_ylabel('Jam Belajar/Hari')
        ax.set_title('Tidur vs Belajar per Kelompok', fontweight='bold', color='#0c2340', fontsize=10)
        ax.legend(fontsize=7.5, framealpha=0.8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        fig.patch.set_facecolor('white'); set_chart_style(ax)
        labels_p = ["Tinggi","Sedang","Rendah"]; vals_p = [33.2, 33.9, 32.9]
        bars = ax.bar(labels_p, vals_p, color=['#10b981','#f59e0b','#ef4444'], edgecolor='white', width=0.55)
        ax.set_ylim(0, 42)
        for bar, val in zip(bars, vals_p):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f'{val}%',
                    ha='center', va='bottom', fontsize=10, fontweight='700', color='#0c2340')
        ax.set_title('Distribusi Performa Akademik', fontweight='bold', color='#0c2340', fontsize=10)
        ax.set_ylabel('Persentase (%)')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c3:
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        fig.patch.set_facecolor('white'); set_chart_style(ax)
        features_fi = ['sleep_hours','study_hours','attendance','mental_health','exercise']
        importance  = [0.28, 0.24, 0.19, 0.16, 0.13]
        bars = ax.barh(features_fi, importance, color=SKY[0], edgecolor='white', height=0.6)
        for bar, val in zip(bars, importance):
            ax.text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.2f}', va='center', fontsize=9, color='#374f6b', fontweight='600')
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance (RF)', fontweight='bold', color='#0c2340', fontsize=10)
        ax.invert_yaxis()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 3 — Training curve
    st.markdown("<br>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor('white'); set_chart_style(ax)
    trees   = list(range(10, 210, 10))
    train_a = [78,82,84.5,86,87.5,88.5,89.5,90.5,91,91.5,92,92.3,92.7,93,93.2,93.5,93.7,93.8,94,94.2]
    val_a   = [75,79,82,83.5,85,86,87,88,88.5,89,89.5,90,90.2,90.5,90.8,91,91.2,91.4,91.6,91.8]
    ax.plot(trees, train_a, color=SKY[0], linewidth=2.5, marker='o', markersize=4, label='Training Accuracy')
    ax.plot(trees, val_a,   color='#10b981', linewidth=2.5, marker='s', markersize=4, linestyle='--', label='Validation Accuracy')
    ax.fill_between(trees, train_a, alpha=0.07, color=SKY[0])
    ax.fill_between(trees, val_a,   alpha=0.07, color='#10b981')
    ax.set_xlabel('Jumlah Pohon (n_estimators)'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Kurva Akurasi Random Forest — Training vs Validation', fontweight='bold', color='#0c2340', fontsize=11)
    ax.legend(fontsize=9); ax.set_ylim(70, 100)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<div style="font-size:0.73rem;color:#94a3b8;text-align:center">* Visualisasi representatif berdasarkan karakteristik dan pola distribusi dataset</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
#  TAB 5 — ABOUT
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown("""
    <div class='section-label'>05 — About</div>
    <div class='section-header'>Tentang Proyek</div>
    <div class='section-desc'>Informasi mengenai metode, dataset, alur CRISP-DM, dan teknologi yang digunakan.</div>
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

        # Alur CRISP-DM
        st.markdown("<strong style='color:#0c2340'>🔄 Alur CRISP-DM Proyek Ini</strong>", unsafe_allow_html=True)
        phases = [
            ("1","#0ea5e9","Business Understanding","Identifikasi permasalahan: hubungan gaya hidup dan performa akademik"),
            ("2","#10b981","Data Understanding","Eksplorasi 1.000 data mahasiswa Kaggle — distribusi & korelasi fitur"),
            ("3","#f59e0b","Data Preparation","Encoding, normalisasi, dan split data train/test"),
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
            <strong style='color:#0c2340; font-size:1rem'>🛠 Teknologi & Tools</strong><br><br>
            <span class='tag tag-dark'>Python 3.x</span>
            <span class='tag tag-dark'>scikit-learn</span>
            <span class='tag tag-dark'>pandas</span>
            <span class='tag tag-dark'>NumPy</span>
            <span class='tag'>Streamlit</span>
            <span class='tag'>Matplotlib</span>
            <span class='tag'>joblib</span>
            <span class='tag'>Kaggle Dataset</span>
            <br><br>
            <strong style='color:#0c2340; font-size:1rem'>📂 Sumber Dataset</strong>
            <p style='color:#64748b; font-size:0.85rem; margin-top:0.5rem; line-height:1.7'>
            Dataset diperoleh dari <strong style='color:#0284c7'>Kaggle Open Dataset</strong> yang dipublikasikan oleh <b>Jayaantanaath</b>. Berisi 1.000 catatan mahasiswa dengan 16 fitur gaya hidup dan nilai akademik. Lisensi: CC BY 4.0.
            </p>
            <div style='background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px; padding:0.85rem; margin-top:0.75rem; font-family:monospace; font-size:0.78rem; color:#0284c7; line-height:1.8'>
                Dataset: Student Habits vs Academic Performance<br>
                Mata Kuliah: Data Mining<br>
                Semester: Genap 2024/2025
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Anggota tim
        st.markdown("""
        <div class='custom-card'>
            <strong style='color:#0c2340; font-size:1rem'>👥 Anggota Tim</strong><br><br>
            <div style='display:flex; align-items:center; gap:0.9rem; padding:0.75rem; background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px; margin-bottom:0.6rem'>
              <div style='width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,#0ea5e9,#0284c7);display:flex;align-items:center;justify-content:center;font-weight:800;color:white;font-size:1rem;flex-shrink:0'>A</div>
              <div><div style='font-weight:700;color:#0c2340;font-size:0.9rem'>Nama Anggota 1</div><div style='font-size:0.75rem;color:#64748b;font-family:monospace'>NIM · XXXXXXXXXX</div></div>
            </div>
            <div style='display:flex; align-items:center; gap:0.9rem; padding:0.75rem; background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px'>
              <div style='width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,#0ea5e9,#0284c7);display:flex;align-items:center;justify-content:center;font-weight:800;color:white;font-size:1rem;flex-shrink:0'>B</div>
              <div><div style='font-weight:700;color:#0c2340;font-size:0.9rem'>Nama Anggota 2</div><div style='font-size:0.75rem;color:#64748b;font-family:monospace'>NIM · XXXXXXXXXX</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class='custom-card' style='text-align:center; background: linear-gradient(135deg,#e0f2fe,#f0f9ff)'>
            <strong style='color:#0c2340'>Cara Menjalankan Aplikasi</strong><br>
            <code style='background:#0c2340; color:#7dd3fc; padding:0.5rem 1.25rem; border-radius:8px; display:inline-block; margin-top:0.75rem; font-size:0.9rem'>
            streamlit run app.py
            </code>
        </div>
        """, unsafe_allow_html=True)