# src/main.py

"""
Main Pipeline: Market Basket Analysis (Algoritma Apriori)
=========================================================
Pipeline end-to-end untuk menjalankan seluruh proses Data Mining:
  1. Load data (transaction + pricelist + preprocessing)
  2. Cleaning (hapus noise, fix tipe data)
  3. EDA (tabel & grafik pendapatan)
  4. Transformation (matriks biner)
  5. Mining (Apriori + Association Rules)
  6. Visualization (heatmap, bar chart, network graph)
  7. Kesimpulan
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no pop-up windows)
import warnings
warnings.filterwarnings('ignore')
import os

from data_loader import load_transaction_raw, load_preprocessing, load_pricelist
from model import create_binary_matrix, run_apriori, generate_rules
from utils import (
    plot_revenue_combo, plot_revenue_item,
    plot_heatmap, plot_top_rules, plot_network,
    print_summary, format_rupiah
)


def main():
    # Buat folder output untuk grafik
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'report', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(" MARKET BASKET ANALYSIS -- TIME LAUNDRY")
    print(" Algoritma Apriori | Min Support: 4% | Min Confidence: 30%")
    print("=" * 60)

    # ── TAHAP 1: Load Data ────────────────────────────────────
    print("\n[1/6] Memuat dataset...")
    df_raw = load_transaction_raw()
    df_pre, transactions = load_preprocessing()
    df_price = load_pricelist()
    print(f"  [OK] Data transaksi mentah  : {df_raw.shape[0]} baris, {df_raw.shape[1]} kolom")
    print(f"  [OK] Data preprocessing     : {df_pre.shape[0]} transaksi")
    print(f"  [OK] Data pricelist         : {df_price.shape[0]} jenis layanan")

    # ── TAHAP 2: Data Cleaning ────────────────────────────────
    print("\n[2/6] Data cleaning selesai (kolom noise dihapus, tipe data Kg diperbaiki).")
    print("  Preview data bersih:")
    print(df_raw.head().to_string(index=False))

    # ── TAHAP 3: EDA & Business Value ─────────────────────────
    print("\n[3/6] Analisis Pendapatan...")
    print("\n  --- Pendapatan Level Paket Transaksi (Top 10) ---")
    combo_data = df_raw.groupby('Variasi')['Total harga'].sum().sort_values(ascending=False).head(10).reset_index()
    combo_data['Total harga (Rp)'] = combo_data['Total harga'].apply(format_rupiah)
    print(combo_data[['Variasi', 'Total harga (Rp)']].to_string(index=False))

    plot_revenue_combo(df_raw, save_path=os.path.join(output_dir, 'revenue_combo.png'))
    print(f"  [OK] Grafik tersimpan: report/figures/revenue_combo.png")

    print("\n  --- Pendapatan Level Satuan Barang (Top 10) ---")
    df_desc = plot_revenue_item(df_raw, df_price, save_path=os.path.join(output_dir, 'revenue_item.png'))
    print(f"  [OK] Grafik tersimpan: report/figures/revenue_item.png")
    df_desc_display = df_desc.copy()
    df_desc_display['Total_Pendapatan (Rp)'] = df_desc_display['Total_Pendapatan'].apply(format_rupiah)
    print(df_desc_display[['Layanan', 'Frekuensi', 'Total_Pendapatan (Rp)']].to_string(index=False))

    # ── TAHAP 4: Data Transformation ──────────────────────────
    print("\n[4/6] Transformasi ke Matriks Biner (One-Hot Encoding)...")
    df_encoded = create_binary_matrix(transactions)
    print(f"  [OK] Matriks biner: {df_encoded.shape[0]} transaksi x {df_encoded.shape[1]} layanan")
    print("\n  Cuplikan Matriks Biner (5 baris pertama):")
    print(df_encoded.astype(int).head().to_string())

    # ── TAHAP 5: Data Mining (Apriori) ────────────────────────
    print("\n[5/6] Menjalankan Algoritma Apriori...")
    frequent_itemsets = run_apriori(df_encoded, min_support=0.04)
    print(f"  [OK] Frequent itemsets ditemukan: {len(frequent_itemsets)}")

    rules = generate_rules(frequent_itemsets, min_confidence=0.3)
    print(f"  [OK] Association rules ditemukan: {len(rules)}")

    if len(rules) > 0:
        print("\n  --- Top 10 Association Rules (Diurutkan berdasarkan Lift) ---")
        display_rules = rules.head(10).copy()
        display_rules['antecedent_str'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        display_rules['consequent_str'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
        print(display_rules[['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift']].to_string(index=False))

    # ── TAHAP 6: Visualisasi ──────────────────────────────────
    print("\n[6/6] Membuat visualisasi...")

    if len(rules) > 0:
        plot_heatmap(rules, save_path=os.path.join(output_dir, 'heatmap.png'))
        print(f"  [OK] Grafik tersimpan: report/figures/heatmap.png")
        plot_top_rules(rules, save_path=os.path.join(output_dir, 'top_rules.png'))
        print(f"  [OK] Grafik tersimpan: report/figures/top_rules.png")
        plot_network(rules, save_path=os.path.join(output_dir, 'network_graph.png'))
        print(f"  [OK] Grafik tersimpan: report/figures/network_graph.png")

    # ── RANGKUMAN ─────────────────────────────────────────────
    print_summary(rules)

    print("\n[DONE] Pipeline selesai! Semua tahapan KDD berhasil dieksekusi.")


if __name__ == "__main__":
    main()
