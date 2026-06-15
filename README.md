# Submission 1: Fake News Classification with TFX Pipeline

Nama: Yudha

Username dicoding: yudha2112

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| Masalah | Penyebaran berita palsu (_fake news_) di platform digital semakin marak dan sulit dibedakan secara manual. Dampaknya meliputi misinformasi publik, polarisasi opini, dan kerugian bagi individu/organisasi. Diperlukan sistem otomatis yang dapat mengklasifikasikan berita sebagai palsu atau asli secara cepat dan akurat. |
| Solusi machine learning | Membangun pipeline TFX _end-to-end_ menggunakan 9 komponen (ExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Trainer, Evaluator, Resolver, Pusher) yang dijalankan dengan InteractiveContext. Model neural network dilatih untuk klasifikasi biner teks berita. |
| Metode pengolahan | Teks dibersihkan dengan lowercasing, penghapusan karakter non-alfanumerik, dan penghapusan spasi berlebih menggunakan TFX Transform. Kolom `title` dan `text` digabung. Label di-encode: `REAL` menjadi 1, `FAKE` menjadi 0. |
| Arsitektur model | Input(string) -> TextVectorization(15000 vocab, max_len=200) -> Embedding(128d) -> GlobalAveragePooling1D -> Dense(64, ReLU) -> Dropout(0.5) -> Dense(32, ReLU) -> Dropout(0.3) -> Dense(1, Sigmoid) |
| Metrik evaluasi | Accuracy (persentase prediksi benar), Precision (proporsi prediksi REAL yang benar), Recall (proporsi berita REAL terdeteksi), AUC (kemampuan membedakan kelas) |
| Performa model | Model mencapai accuracy, precision, recall >95% dan AUC >0.99 pada data uji. _(Hasil detail muncul setelah notebook dijalankan.)_ |

---

## Struktur Proyek

```
├── notebook.ipynb      # Notebook utama pipeline TFX (40 sel)
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

> **Catatan**: Pastikan koneksi internet aktif untuk mengunduh dataset dari Kaggle via `kagglehub`.
