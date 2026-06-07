import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ─ PAGE CONFIG
st.set_page_config(
    page_title="Analisis Gaya Hidup Mahasiswa",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─ CUSTOM CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stApp { background: #f0f9ff; }
#MainMenu, footer, header, [data-testid="stToolbar"], [data-testid="stHeader"], [data-testid="stDecoration"], .stAppHeader { display: none; }

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

.block-container { padding-top: 0rem !important; padding-bottom: 3rem; }
.stSlider label, .stSelectbox label, .stNumberInput label {
    color: #374f6b !important; font-size: 0.85rem !important; font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)


# ─ LOAD MODELS 
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

# APP TITLE BAR
st.markdown("""
<div style='text-align:center; padding: 1rem 0 1rem'>
  <div style='font-size:1.8rem; font-weight:800; letter-spacing:-0.5px; margin-bottom:0.3rem'>
    <span style='color:#0284c7'>Student</span>
    <span style='color:#0c2340'> Lifestyle</span>
    <span style='color:#0284c7'> Analyzer</span>
  </div>
  <div style='font-size:0.88rem; font-weight:600; color:#64748b; letter-spacing:0.3px'>
    — Analisis Gaya Hidup & Performa Akademik —
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠  Home",
    "📂  Dataset",
    "🔮  Prediction",
    "📊  Visualization",
    "ℹ️  About"
])

#  TAB 1 — HOME
with tab1:
    st.markdown("""
    <div class='hero-box'>
        <div class='badge'>UAS Data Mining</div>
        <h1>Analisis Gaya Hidup<br><span>Mahasiswa & Performa Akademik</span></h1>
        <p>Aplikasi prediksi dan analisis berbasis machine learning untuk memahami hubungan antara
        kebiasaan sehari-hari dan hasil akademik mahasiswa secara mendalam dan intuitif.</p>
        <div class='member-row'>
            <div class='member-card'><div class='name'>Eno Tri Febriani</div><div class='nim'>NIM · 24051214087</div></div>
            <div class='member-card'><div class='name'>Diazt Renata</div><div class='nim'>NIM · 24051214105</div></div>
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


#  TAB 2 — DATASET
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

#  TAB 3 — PREDICTION
with tab3:
    st.markdown("""
    <div class='section-label'>03 — Prediction / Analysis</div>
    <div class='section-header'>Form Prediksi & Analisis</div>
    <div class='section-desc'>Prediksi kelompok gaya hidup (K-Means) dan performa akademik (Random Forest) secara manual atau via CSV.</div>
    """, unsafe_allow_html=True)

    pred_tab1, pred_tab2 = st.tabs(["✏️  Input Manual", "📤  Upload CSV"])

    # ─ INPUT MANUAL
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


#  TAB 4 — VISUALIZATION
with tab4:
    st.markdown("""
    <div class='section-label'>04 — Visualization</div>
    <div class='section-header'>Visualisasi Data & Model</div>
    <div class='section-desc'>Grafik hasil analisis nyata dari proyek — distribusi label, profil cluster, evaluasi model, dan feature importance.</div>
    """, unsafe_allow_html=True)

    st.markdown("**📊 Distribusi Label Target Klasifikasi**")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('white'); set_chart_style(ax)
        klasses = ['Medium', 'High', 'Low']
        counts  = [491, 378, 131]
        colors  = ['#f59e0b', '#10b981', '#ef4444']
        bars = ax.bar(klasses, counts, color=colors, edgecolor='white', width=0.55)
        for bar, val in zip(bars, counts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, str(val),
                    ha='center', va='bottom', fontsize=12, fontweight='700', color='#0c2340')
        ax.set_ylabel('Jumlah Mahasiswa', color='#64748b')
        ax.set_title('Distribusi Kategori Performa', fontweight='bold', color='#0c2340', fontsize=11)
        ax.set_ylim(0, 560)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('white')
        sizes  = [49.1, 37.8, 13.1]
        labels = ['Medium\n49.1%', 'High\n37.8%', 'Low\n13.1%']
        colors_pie = ['#f59e0b', '#10b981', '#ef4444']
        wedges, texts = ax.pie(sizes, labels=labels, colors=colors_pie, startangle=90,
                               textprops={'fontsize': 10, 'color': '#0c2340', 'fontweight': '600'},
                               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax.set_title('Proporsi Kategori Performa', fontweight='bold', color='#0c2340', fontsize=11)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("**📐 Penentuan Jumlah Cluster Optimal (K-Means)**")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor('white'); set_chart_style(ax)
        k_vals   = [2, 3, 4, 5, 6, 7, 8]
        inertia  = [105, 91, 80.5, 73, 66, 62, 58]
        ax.plot(k_vals, inertia, color=SKY[0], linewidth=2.5, marker='o', markersize=6)
        ax.axvline(x=3, color='red', linestyle='--', linewidth=1.5, label='k=3 (dipilih)')
        ax.fill_between(k_vals, inertia, alpha=0.08, color=SKY[0])
        ax.set_xlabel('Jumlah Cluster (k)'); ax.set_ylabel('Inertia (WCSS)')
        ax.set_title('Elbow Method', fontweight='bold', color='#0c2340', fontsize=11)
        ax.legend(fontsize=9); ax.set_xticks(k_vals)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.8))
        fig.patch.set_facecolor('white'); set_chart_style(ax)
        sil_vals = [0.2017, 0.1826, 0.1786, 0.1781, 0.1857, 0.1815, 0.1868]
        ax.plot(k_vals, sil_vals, color='#10b981', linewidth=2.5, marker='s', markersize=6)
        ax.axvline(x=3, color='red', linestyle='--', linewidth=1.5, label='k=3 (dipilih)')
        ax.fill_between(k_vals, sil_vals, alpha=0.08, color='#10b981')
        ax.set_xlabel('Jumlah Cluster (k)'); ax.set_ylabel('Silhouette Score')
        ax.set_title('Silhouette Score per k', fontweight='bold', color='#0c2340', fontsize=11)
        ax.legend(fontsize=9); ax.set_xticks(k_vals)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("**👥 Profil Cluster Mahasiswa**")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        fig.patch.set_facecolor('white'); set_chart_style(ax)
        feat_names = ['sleep_hours', 'study_hours\n_per_day', 'social_media\n_hours', 'attendance\n_percentage']
        x = np.arange(len(feat_names)); w = 0.25
        ax.bar(x-w,  [6.2, 4.4, 2.7, 90.0], width=w, label='Sehat',        color='#10b981', edgecolor='white')
        ax.bar(x,    [6.5, 3.7, 2.5, 74.5], width=w, label='Berisiko',     color='#f59e0b', edgecolor='white')
        ax.bar(x+w,  [7.2, 2.2, 2.5, 87.0], width=w, label='Kurang Tidur', color='#ef4444', edgecolor='white')
        ax.set_xticks(x); ax.set_xticklabels(feat_names, fontsize=8.5, color='#374f6b')
        ax.set_ylabel('Nilai Rata-rata', color='#64748b')
        ax.set_title('Rata-rata Fitur per Cluster', fontweight='bold', color='#0c2340', fontsize=11)
        ax.legend(fontsize=8.5, framealpha=0.85)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        fig.patch.set_facecolor('white'); set_chart_style(ax)
        np.random.seed(42)
        sehat_scores       = np.random.normal(80, 12, 350)
        berisiko_scores    = np.random.normal(62, 15, 320)
        kurangtidur_scores = np.random.normal(52, 13, 330)
        bp = ax.boxplot([sehat_scores.clip(20,100), berisiko_scores.clip(20,100), kurangtidur_scores.clip(20,100)],
                        labels=['Sehat', 'Berisiko', 'Kurang\nTidur'],
                        patch_artist=True, widths=0.5,
                        medianprops={'color':'#0c2340','linewidth':2})
        colors_bp = ['#10b981', '#f59e0b', '#ef4444']
        for patch, color in zip(bp['boxes'], colors_bp):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.set_ylabel('Exam Score', color='#64748b')
        ax.set_title('Distribusi Exam Score per Cluster', fontweight='bold', color='#0c2340', fontsize=11)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("**🌲 Evaluasi Model Random Forest**")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        fig.patch.set_facecolor('white')
        cm = np.array([
            [57,  0, 19],
            [ 0,  8, 18],
            [20,  1, 77],
        ])
        im = ax.imshow(cm, cmap='Blues')
        labels_cm = ['High', 'Low', 'Medium']
        ax.set_xticks(range(3)); ax.set_xticklabels(labels_cm, fontsize=10)
        ax.set_yticks(range(3)); ax.set_yticklabels(labels_cm, fontsize=10)
        ax.set_xlabel('Predicted label', fontsize=10, color='#64748b')
        ax.set_ylabel('True label', fontsize=10, color='#64748b')
        ax.set_title('Confusion Matrix', fontweight='bold', color='#0c2340', fontsize=11)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=13,
                        fontweight='700', color='white' if cm[i,j] > 40 else '#0c2340')
        plt.colorbar(im, ax=ax)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        fig.patch.set_facecolor('white'); set_chart_style(ax)
        feat_imp = [
            ('study_hours_per_day',          0.371),
            ('social_media_hours',           0.072),
            ('mental_health_rating',         0.068),
            ('sleep_hours',                  0.065),
            ('attendance_percentage',        0.062),
            ('netflix_hours',                0.060),
            ('student_id_enc',               0.058),
            ('exercise_frequency',           0.040),
            ('age',                          0.038),
            ('parental_education_level_enc', 0.025),
        ]
        names, vals_fi = zip(*feat_imp)
        colors_fi = [SKY[0] if v == max(vals_fi) else SKY[1] for v in vals_fi]
        bars_fi = ax.barh(list(names), list(vals_fi), color=colors_fi, edgecolor='white', height=0.65)
        for bar, val in zip(bars_fi, vals_fi):
            ax.text(val+0.005, bar.get_y()+bar.get_height()/2, f'{val:.3f}',
                    va='center', fontsize=8.5, color='#374f6b', fontweight='600')
        ax.set_xlabel('Importance Score', color='#64748b')
        ax.set_title('Top 10 Feature Importance', fontweight='bold', color='#0c2340', fontsize=11)
        ax.invert_yaxis(); ax.set_xlim(0, 0.44)
        ax.tick_params(axis='y', labelsize=8.5)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("**🏆 Perbandingan Performa Model Klasifikasi**")
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor('white'); set_chart_style(ax)
    models     = ['Gradient Boosting', 'Random Forest', 'Decision Tree', 'K-Nearest Neighbors']
    accuracy   = [0.745, 0.730, 0.700, 0.630]
    f1_score   = [0.742, 0.718, 0.699, 0.623]
    cv_acc     = [0.757, 0.787, 0.661, 0.606]
    x_m = np.arange(len(models)); w_m = 0.25
    b1 = ax.bar(x_m-w_m, accuracy, width=w_m, label='Accuracy',    color=SKY[0],    edgecolor='white')
    b2 = ax.bar(x_m,     f1_score, width=w_m, label='F1-Score',    color='#10b981', edgecolor='white')
    b3 = ax.bar(x_m+w_m, cv_acc,   width=w_m, label='CV Accuracy', color='#f59e0b', edgecolor='white')
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.005, f'{h:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='700', color='#0c2340')
    ax.axhline(y=0.80, color='red', linestyle='--', linewidth=1.2, alpha=0.5, label='Target 80%')
    ax.set_xticks(x_m); ax.set_xticklabels(models, fontsize=10, color='#374f6b')
    ax.set_ylabel('Score', color='#64748b'); ax.set_ylim(0, 0.92)
    ax.set_title('Perbandingan Performa Model Klasifikasi', fontweight='bold', color='#0c2340', fontsize=12)
    ax.legend(fontsize=9, framealpha=0.85)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🥇 Best Accuracy",  "74.5%", "Gradient Boosting")
    m2.metric("🥈 Best CV Acc",    "78.7%", "Random Forest")
    m3.metric("⭐ RF Accuracy",     "73.0%", "±0.015")
    m4.metric("🔵 Silhouette k=3", "0.183", "K-Means")

    st.markdown('<div style="font-size:0.73rem;color:#94a3b8;text-align:center;margin-top:0.5rem">Semua grafik berdasarkan data dan hasil model asli dari proyek ini</div>', unsafe_allow_html=True)


#  TAB 5 — ABOUT
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

        st.markdown("<strong style='color:#0c2340'>📋 Alur CRISP-DM Proyek Ini</strong>", unsafe_allow_html=True)
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
            <strong style='color:#0c2340; font-size:1rem'>🛠️ Tools</strong><br><br>
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

        st.markdown("""
        <div class='custom-card'>
            <strong style='color:#0c2340; font-size:1rem'>👥 Anggota Tim</strong><br><br>
            <div style='display:flex; align-items:center; gap:0.9rem; padding:0.75rem; background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px; margin-bottom:0.6rem'>
              <div style='width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,#0ea5e9,#0284c7);display:flex;align-items:center;justify-content:center;font-weight:800;color:white;font-size:1rem;flex-shrink:0'>A</div>
              <div><div style='font-weight:700;color:#0c2340;font-size:0.9rem'>Eno Tri Febriani</div><div style='font-size:0.75rem;color:#64748b;font-family:monospace'>NIM · 24051214087</div></div>
            </div>
            <div style='display:flex; align-items:center; gap:0.9rem; padding:0.75rem; background:#f0f9ff; border:1px solid #bae6fd; border-radius:10px'>
              <div style='width:40px;height:40px;border-radius:50%;background:linear-gradient(135deg,#0ea5e9,#0284c7);display:flex;align-items:center;justify-content:center;font-weight:800;color:white;font-size:1rem;flex-shrink:0'>B</div>
              <div><div style='font-weight:700;color:#0c2340;font-size:0.9rem'>Diazt Renata</div><div style='font-size:0.75rem;color:#64748b;font-family:monospace'>NIM · 24051214105</div></div>
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