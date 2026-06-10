#  Analisis Pola Hidup & Prediksi Performa Akademik Mahasiswa

> **UAS Data Mining** — Clustering dan Classification Berbasis CRISP-DM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.x-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Deskripsi Proyek

Penelitian ini menganalisis **pola gaya hidup mahasiswa** dan **memprediksi performa akademik** menggunakan pendekatan machine learning berbasis framework **CRISP-DM**.

- **K-Means Clustering** → mengelompokkan mahasiswa ke dalam 3 cluster gaya hidup: *Sehat*, *Berisiko*, dan *Kurang Tidur*
- **Random Forest Classifier** → memprediksi kategori performa akademik: *High*, *Medium*, dan *Low*

Dataset bersumber dari Kaggle — *Student Habits and Academic Performance* (1.000 data, 16 fitur).

---

## Demo Aplikasi

 **[Buka Aplikasi Streamlit](https://uasdataminingkelompok4-f8chd7fjshftwrzkq4tm6e.streamlit.app/)**

Aplikasi terdiri dari 5 tab utama:
| Tab | Konten |
|-----|--------|
|  Beranda | Gambaran umum proyek & dataset |
|  EDA | Eksplorasi data & visualisasi distribusi |
|  Clustering | Hasil K-Means & profil tiap cluster |
|  Klasifikasi | Evaluasi model & feature importance |
|  Prediksi | Input data gaya hidup → prediksi personal |

---

##  Struktur Repositori

```
UAS_DataMining_NamaKelompok/
│
├── dataset/
│   └── student_habits_performance.csv   # Dataset Kaggle (1.000 baris, 16 fitur)
│
├── notebook/
│   └── analysis.ipynb                   # Notebook Google Colab (EDA + Modeling)
│
├── model/
│   └── model.pkl                        # Model terlatih (Random Forest + K-Means)
│
├── app/
│   ├── app.py                           # Entry point Streamlit
│ 
│
├── laporan/
│   └── laporan.pdf                      # Laporan akhir proyek
│
├── requirements.txt                     # Dependensi Python
└── README.md
```

---

##  Cara Menjalankan Lokal

### 1. Clone repositori
```bash
git clone https://github.com/auntumlivy/UAS_DataMining.git
cd UAS_DataMining
```

### 2. Buat virtual environment & install dependensi
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Jalankan aplikasi Streamlit
```bash
streamlit run app/app.py
```

Buka browser di `http://localhost:8501`

---

##  Dependensi Utama

```
streamlit
pandas
numpy
scikit-learn==1.3.2
matplotlib
seaborn
plotly
joblib
```

> Lihat `requirements.txt` untuk versi lengkap.

---

##  Hasil Model

### K-Means Clustering (k=3)
| Cluster | Label | Karakteristik Utama |
|---------|-------|---------------------|
| 0 |  Sehat | Tidur cukup, belajar rutin, olahraga teratur |
| 1 |  Berisiko | Media sosial tinggi, kehadiran rendah |
| 2 |  Kurang Tidur | Durasi tidur < 6 jam, performa rendah |

**Silhouette Score:** 0.312

### Random Forest Classifier
| Metrik | Nilai |
|--------|-------|
| Accuracy | 87.5% |
| Top Feature | `study_hours_per_day` (~0.371) |
| Distribusi Label | High 34.2% · Medium 37.8% · Low 28.0% |

---

##  Dataset

**Sumber:** [Kaggle — Student Habits vs Academic Performance](https://www.kaggle.com/)

| Info | Detail |
|------|--------|
| Jumlah data | 1.000 baris |
| Jumlah fitur | 16 kolom |
| Target | `exam_score` → High / Medium / Low |

**Kelompok fitur:**
- Demografis: `age`, `gender`
- Kebiasaan harian: `study_hours_per_day`, `social_media_hours`, `netflix_hours`, `sleep_hours`
- Kesehatan: `exercise_frequency`, `diet_quality`, `mental_health_rating`
- Lingkungan: `internet_quality`, `parental_education_level`, `attendance_percentage`

---

##  Tim Pengembang

| Nama | NIM | Program Studi |
|------|-----|---------------|
| Eno Tri Febriani | 24051214087 | S1 Sistem Informasi, UNESA |
| Diazt Renata | 24051214105 | S1 Sistem Informasi, UNESA |

---

##  Lisensi

Proyek ini dibuat untuk keperluan akademik — **Ujian Akhir Semester Data Mining**, Universitas Negeri Surabaya.
