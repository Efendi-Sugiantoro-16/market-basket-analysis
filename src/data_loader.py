# src/data_loader.py

"""
data_loader.py

Modul untuk memuat dan membersihkan dataset Time Laundry.
Mendukung tiga jenis file:
  - Transaction (data mentah lengkap)
  - Preprocessing (data siap encoding: ID + Variasi)
  - Pricelist (referensi harga per layanan)
"""

import pandas as pd
import os

# Path relatif dari root repository
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')

TRANSACTION_FILE = 'mba-laundry-transaction-20260704.csv'
PREPROCESSING_FILE = 'mba-laundry-transaction-preprocessing-20260704.csv'
PRICELIST_FILE = 'mba-laundry-pricelist-20260704.csv'


def load_transaction_raw():
    """
    Memuat data transaksi mentah dan melakukan pembersihan dasar:
      - Menghapus kolom 'Unnamed: 6' jika ada (noise dari format CSV).
      - Mengubah kolom 'Kg' dari format koma ke titik, lalu cast ke float.

    Returns:
        pd.DataFrame: DataFrame transaksi yang sudah bersih.
    """
    filepath = os.path.join(DATA_PATH, TRANSACTION_FILE)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    df = pd.read_csv(filepath)

    # Hapus kolom noise
    if 'Unnamed: 6' in df.columns:
        df = df.drop(columns=['Unnamed: 6'])

    # Perbaiki tipe data Kg (koma -> titik -> float)
    if 'Kg' in df.columns and df['Kg'].dtype == 'object':
        df['Kg'] = df['Kg'].str.replace(',', '.').astype(float)

    return df


def load_preprocessing():
    """
    Memuat data preprocessing (ID, Variasi) dan mem-parse kolom 'Variasi'
    menjadi list item per transaksi.

    Returns:
        tuple: (df, transactions)
            - df (pd.DataFrame): DataFrame asli dengan kolom ID dan Variasi.
            - transactions (list of list): Daftar keranjang belanja per transaksi.
              Contoh: [['Cuci Kering', 'Setrika'], ['Karpet'], ...]
    """
    filepath = os.path.join(DATA_PATH, PREPROCESSING_FILE)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    df = pd.read_csv(filepath)

    # Parse kolom Variasi menjadi list item
    transactions = df['Variasi'].apply(
        lambda x: [item.strip() for item in str(x).split(',')]
    ).tolist()

    return df, transactions


def load_pricelist():
    """
    Memuat data referensi harga layanan (pricelist).

    Returns:
        pd.DataFrame: DataFrame dengan kolom Variasi dan Harga.
    """
    filepath = os.path.join(DATA_PATH, PRICELIST_FILE)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")

    return pd.read_csv(filepath)


# ── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("TEST: Load Transaction Raw")
    print("=" * 50)
    df_raw = load_transaction_raw()
    print(f"Shape: {df_raw.shape}")
    print(df_raw.head())

    print("\n" + "=" * 50)
    print("TEST: Load Preprocessing")
    print("=" * 50)
    df_pre, trx = load_preprocessing()
    print(f"Shape: {df_pre.shape}")
    print(f"Contoh transaksi: {trx[:5]}")

    print("\n" + "=" * 50)
    print("TEST: Load Pricelist")
    print("=" * 50)
    df_price = load_pricelist()
    print(df_price)
