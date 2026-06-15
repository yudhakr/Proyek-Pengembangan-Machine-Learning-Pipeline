# Fake News Classification with TFX Pipeline

Proyek ini membangun pipeline machine learning untuk klasifikasi berita palsu (Fake News) menggunakan TensorFlow Extended (TFX) dan dataset **Fake and Real News Dataset** dari Kaggle.

---

## Dataset

- **Sumber**: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) oleh Clment Bisaillon
- **Jumlah data**: ~44.000 artikel berita
- **Fitur**:
  - `title` — judul artikel
  - `text` — isi artikel
  - `subject` — kategori topik
  - `date` — tanggal publikasi
- **Label**:
  - `FAKE` — berita palsar
  - `REAL` — berita asli

---

## Masalah

Penyebaran berita palsu di media digital semakin marak dan sulit dibedakan secara manual. Diperlukan sistem otomatis yang dapat mengklasifikasikan apakah sebuah artikel berita tergolong palsu atau asli dengan cepat dan akurat.

---

## Solusi Machine Learning

Membangun pipeline TFX end-to-end yang mencakup seluruh siklus hidup ML:

1. **Data ingestion** — membaca dataset CSV
2. **Data validation** — memeriksa kualitas data
3. **Data transformation** — membersihkan teks dan encoding label
4. **Model training** — melatih model neural network untuk klasifikasi teks
5. **Model evaluation** — mengevaluasi performa model
6. **Model deployment** — menyimpan model siap serving

Pipeline dijalankan menggunakan `InteractiveContext` untuk pengembangan interaktif di Jupyter Notebook.

---

## Metode Pengolahan

### Text Preprocessing
- Menggabungkan kolom `title` dan `text`
- Lowercasing
- Menghapus karakter non-alfanumerik
- Menghapus spasi berlebih

### Label Encoding
- `REAL` → 1
- `FAKE` → 0

### TFX Transform
Menggunakan `tensorflow-transform` (TFX Transform) untuk preprocessing yang konsisten antara training dan serving.

---

## Arsitektur Model

```
Input (string teks)
       ↓
TextVectorization (max_tokens=15000, output_sequence_length=200)
       ↓
Embedding (vocab_size=15000, embedding_dim=128)
       ↓
GlobalAveragePooling1D
       ↓
Dense (64, ReLU)
       ↓
Dropout (0.5)
       ↓
Dense (32, ReLU)
       ↓
Dropout (0.3)
       ↓
Dense (1, Sigmoid)
```

Model menggunakan pendekatan **bag-of-words dengan representasi dense** (embedding) yang cocok untuk klasifikasi teks.

---

## Metrik Evaluasi

| Metrik       | Keterangan                                  |
|--------------|---------------------------------------------|
| Accuracy     | Persentase prediksi benar                   |
| Precision    | Proporsi prediksi positif yang benar        |
| Recall       | Proporsi data positif yang terdeteksi       |
| AUC          | Area Under the ROC Curve                    |

---

## Performa Model

_Hasil performa akan muncul setelah notebook dijalankan. Model diharapkan mencapai:_

- **Accuracy**: >95%
- **Precision**: >95%
- **Recall**: >95%
- **AUC**: >0.99

---

## Struktur Proyek

```
├── notebook.ipynb      # Notebook utama pipeline TFX
├── transform.py         # Modul preprocessing TFX Transform
├── trainer.py           # Modul training TFX Trainer
├── requirements.txt     # Dependensi Python
├── README.md            # Dokumentasi proyek
├── pipeline/            # Folder pipeline
└── serving_model/       # Output model siap serving
```

## Cara Menjalankan

1. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

2. Buka notebook:
   ```bash
   jupyter notebook notebook.ipynb
   ```

3. Jalankan seluruh sel secara berurutan.

> **Catatan**: Pastikan koneksi internet aktif untuk mengunduh dataset dari Kaggle.
# Proyek-Pengembangan-Machine-Learning-Pipeline
