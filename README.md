# Tugas Besar Data Mining: Market Basket Analysis — Time Laundry

Repositori ini berisi implementasi **Market Basket Analysis** untuk mengidentifikasi pola hubungan antar layanan pada Time Laundry menggunakan **Algoritma Apriori**.

## Judul Penelitian

> Penerapan Market Basket Analysis Menggunakan Algoritma Apriori untuk Menentukan Strategi Bundling Layanan pada Time Laundry.

## Anggota Kelompok & NIM

| No  | Nama                   | NIM       |
| --- | ---------------------- | --------- |
| 1   | Afifah Naufal Rahmani  | 714230026 |
| 2   | Nurfanis Rosman        | 714230022 |
| 3   | Efendi Sugiantoro      | 714230018 |
| 4   | Hadzik Mochamad Sofyan | 714230040 |

## Deskripsi Proyek

Penelitian ini bertujuan untuk membantu pengelola Time Laundry dalam memahami pola perilaku konsumen melalui data transaksi. Dengan menggunakan metode Market Basket Analysis dan algoritma Apriori, penelitian ini mengidentifikasi keterkaitan antar layanan untuk merumuskan strategi penggabungan layanan (bundling) yang efektif guna meningkatkan volume transaksi.

**Dataset:** 273 catatan transaksi pelanggan dengan 19 varian layanan.

**Metodologi:** Knowledge Discovery in Database (KDD)

- Data Selection & Cleaning
- Data Transformation (One-Hot Encoding)
- Data Mining (Algoritma Apriori)
- Evaluation & Visualization

**Parameter Apriori:**

- Minimum Support: 4%
- Minimum Confidence: 30%

## Struktur Folder

```
market-basket-analysis/
|-- data/                          # Dataset (CSV)
|   |-- mba-laundry-transaction-20260704.csv
|   |-- mba-laundry-transaction-preprocessing-20260704.csv
|   |-- mba-laundry-pricelist-20260704.csv
|
|-- notebook/                      # Jupyter Notebook (per tahapan)
|   |-- preprocessing_template.ipynb   # Tahap 1 & 2: Loading & Cleaning
|   |-- eda_template.ipynb             # Tahap 3: Exploratory Data Analysis
|   |-- modeling_template.ipynb        # Tahap 4, 5, 6: Transformation, Mining, Visualization
|
|-- src/                           # Source Code Python (Modular)
|   |-- data_loader.py             # Fungsi pemuatan & pembersihan data
|   |-- model.py                   # Algoritma Apriori & Association Rules
|   |-- utils.py                   # Visualisasi (Heatmap, Bar Chart, Network Graph)
|   |-- main.py                    # Pipeline utama (end-to-end)
|   |-- main_notebook.ipynb        # Versi interaktif dari main.py
|
|-- report/                        # Laporan (PPT, Jurnal)
|-- requirements.txt               # Daftar dependensi Python
|-- run.sh                         # Script untuk menjalankan pipeline (Linux/Mac)
|-- run.bat                        # Script untuk menjalankan pipeline (Windows)
|-- README.md                      # Dokumentasi ini
```

## Cara Menjalankan

### 1. Install Dependensi

```bash
pip install -r requirements.txt
```

### 2. Jalankan Pipeline via Terminal

```bash
python src/main.py
```

### 3. Jalankan via Jupyter Notebook

Buka salah satu notebook di folder `notebook/` atau `src/main_notebook.ipynb` menggunakan Jupyter:

```bash
jupyter notebook
```

## Hasil Utama

- Ditemukan **9 frequent itemset** yang signifikan.
- Aturan **Cuci Kering -> Setrika** memiliki confidence tertinggi (100%).
- Aturan **Setrika, Sprei -> Cuci Kering** memiliki Lift sebesar 3.38 (asosiasi sangat kuat).
- **Rekomendasi:** Terapkan promo "Bed Room Bundle" (Sprei + Bed Cover) berdasarkan Lift tertinggi.

## Teknologi yang Digunakan

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- mlxtend (Algoritma Apriori)
- networkx (Network Graph)
- Jupyter Notebook
