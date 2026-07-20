# src/model.py

"""
model.py

Modul inti untuk Market Basket Analysis menggunakan Algoritma Apriori.
Berisi fungsi-fungsi:
  - Transformasi data transaksi ke matriks biner (One-Hot Encoding).
  - Pencarian Frequent Itemsets menggunakan Apriori.
  - Pembangkitan Association Rules dengan metrik Support, Confidence, dan Lift.
"""

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd


def create_binary_matrix(transactions):
    """
    Mengubah daftar transaksi (list of list) menjadi matriks biner
    menggunakan TransactionEncoder dari mlxtend.

    Parameters:
        transactions (list of list): Keranjang belanja per transaksi.
            Contoh: [['Cuci Kering', 'Setrika'], ['Karpet'], ...]

    Returns:
        pd.DataFrame: Matriks biner (0/1) dengan kolom = nama layanan.
    """
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    return df_encoded


def run_apriori(df_encoded, min_support=0.04):
    """
    Menjalankan algoritma Apriori untuk menemukan frequent itemsets
    dari matriks biner.

    Rumus Support:
        Support(A) = (Jumlah transaksi mengandung A) / (Total transaksi)

    Parameters:
        df_encoded (pd.DataFrame): Matriks biner hasil dari create_binary_matrix().
        min_support (float): Ambang batas minimum support (default: 4%).

    Returns:
        pd.DataFrame: Tabel frequent itemsets dengan kolom 'support' dan 'itemsets'.
    """
    frequent_itemsets = apriori(
        df_encoded,
        min_support=min_support,
        use_colnames=True
    )
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
    return frequent_itemsets


def generate_rules(frequent_itemsets, min_confidence=0.3):
    """
    Membangkitkan Association Rules dari frequent itemsets.

    Rumus Confidence:
        Confidence(A → B) = Support(A ∪ B) / Support(A)

    Rumus Lift:
        Lift(A → B) = Confidence(A → B) / Support(B)
        - Lift > 1: Asosiasi positif (saling menarik).
        - Lift = 1: Independen (tidak ada hubungan).
        - Lift < 1: Asosiasi negatif (saling menggantikan/substitusi).

    Parameters:
        frequent_itemsets (pd.DataFrame): Hasil dari run_apriori().
        min_confidence (float): Ambang batas minimum confidence (default: 30%).

    Returns:
        pd.DataFrame: Tabel aturan asosiasi diurutkan berdasarkan Lift tertinggi.
    """
    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    # Urutkan berdasarkan Lift Ratio tertinggi
    rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)

    return rules


# ── Quick Test ──────────────────────────────────────────────
if __name__ == "__main__":
    # Contoh sederhana
    sample_transactions = [
        ['Cuci Kering', 'Setrika'],
        ['Karpet'],
        ['Bed Cover', 'Sprei'],
        ['Cuci Kering', 'Setrika', 'Sprei'],
        ['Setrika'],
    ]

    print("=" * 50)
    print("TEST: Binary Matrix")
    print("=" * 50)
    df_bin = create_binary_matrix(sample_transactions)
    print(df_bin)

    print("\n" + "=" * 50)
    print("TEST: Apriori (min_support=0.2)")
    print("=" * 50)
    freq = run_apriori(df_bin, min_support=0.2)
    print(freq)

    print("\n" + "=" * 50)
    print("TEST: Association Rules")
    print("=" * 50)
    if len(freq) > 0:
        rules = generate_rules(freq, min_confidence=0.3)
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    else:
        print("Tidak ada frequent itemset ditemukan.")
