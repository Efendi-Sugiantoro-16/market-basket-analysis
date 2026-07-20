# src/utils.py

"""
utils.py

Modul utilitas untuk visualisasi dan fungsi bantu Market Basket Analysis.
Berisi:
  - Format angka ke Rupiah.
  - Grafik pendapatan (Level Paket & Satuan).
  - Heatmap korelasi antar layanan.
  - Bar chart Top Association Rules.
  - Network Graph (Peta Arus Kasir).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
from collections import Counter


def format_rupiah(value):
    """
    Mengonversi angka menjadi format Rupiah Indonesia.

    Parameters:
        value (int/float): Nilai numerik.

    Returns:
        str: String format 'Rp x.xxx.xxx'
    """
    return f"Rp {int(value):,.0f}".replace(",", ".")


def plot_revenue_combo(df, top_n=10, save_path=None):
    """
    Membuat bar chart pendapatan berdasarkan paket transaksi utuh (Combo-Level).
    Memakai data mentah kolom 'Variasi' tanpa dipecah.

    Parameters:
        df (pd.DataFrame): DataFrame transaksi mentah (harus punya kolom 'Variasi' dan 'Total harga').
        top_n (int): Jumlah paket teratas yang ditampilkan.
        save_path (str): Path untuk menyimpan gambar (opsional).
    """
    data = df.groupby('Variasi')['Total harga'].sum().sort_values(ascending=False).head(top_n).reset_index()

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=data, x='Variasi', y='Total harga', color='#00796B')

    # Anotasi nilai Rupiah di atas setiap bar
    for i, row in data.iterrows():
        ax.text(i, row['Total harga'] + row['Total harga'] * 0.01,
                format_rupiah(row['Total harga']),
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.title(f'Top {top_n} Pendapatan per Paket Transaksi (Combo-Level)', fontsize=14, fontweight='bold')
    plt.ylabel('Total Pendapatan (Rp)', fontsize=11)
    plt.xlabel('Variasi Layanan', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_revenue_item(df, df_price, top_n=10, save_path=None):
    """
    Membuat bar chart pendapatan berdasarkan satuan barang individu (Item-Level).
    Variasi dipecah per item, lalu frekuensinya dikalikan harga satuan.

    Parameters:
        df (pd.DataFrame): DataFrame transaksi mentah.
        df_price (pd.DataFrame): DataFrame referensi harga (kolom 'Variasi', 'Harga').
        top_n (int): Jumlah item teratas yang ditampilkan.
        save_path (str): Path untuk menyimpan gambar (opsional).
    """
    # Pecah variasi menjadi item tunggal
    all_items = []
    for variasi in df['Variasi']:
        items = [item.strip() for item in str(variasi).split(',')]
        all_items.extend(items)

    item_counts = Counter(all_items)

    # Buat DataFrame deskripsi
    price_map = dict(zip(df_price['Variasi'], df_price['Harga']))
    records = []
    for item, count in item_counts.items():
        harga = price_map.get(item, 0)
        records.append({
            'Layanan': item,
            'Frekuensi': count,
            'Harga_Satuan': harga,
            'Total_Pendapatan': count * harga
        })

    df_desc = pd.DataFrame(records).sort_values('Total_Pendapatan', ascending=False).head(top_n).reset_index(drop=True)

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=df_desc, x='Layanan', y='Total_Pendapatan', color='#00796B')

    for i, row in df_desc.iterrows():
        ax.text(i, row['Total_Pendapatan'] + row['Total_Pendapatan'] * 0.01,
                format_rupiah(row['Total_Pendapatan']),
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.title(f'Top {top_n} Pendapatan per Satuan Layanan (Item-Level)', fontsize=14, fontweight='bold')
    plt.ylabel('Total Pendapatan (Rp)', fontsize=11)
    plt.xlabel('Nama Layanan', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return df_desc


def plot_heatmap(rules, save_path=None):
    """
    Membuat Heatmap Korelasi antar layanan berdasarkan nilai Lift dari Association Rules.

    Parameters:
        rules (pd.DataFrame): Tabel association rules (harus punya kolom antecedents, consequents, lift).
        save_path (str): Path untuk menyimpan gambar (opsional).
    """
    # Buat pivot table dari rules
    heatmap_data = rules.copy()
    heatmap_data['antecedent'] = heatmap_data['antecedents'].apply(lambda x: ', '.join(list(x)))
    heatmap_data['consequent'] = heatmap_data['consequents'].apply(lambda x: ', '.join(list(x)))

    # Filter hanya aturan dengan 1 antecedent dan 1 consequent untuk heatmap yang bersih
    mask = (heatmap_data['antecedents'].apply(len) == 1) & (heatmap_data['consequents'].apply(len) == 1)
    heatmap_filtered = heatmap_data[mask]

    if heatmap_filtered.empty:
        print("Tidak ada aturan 1-to-1 untuk heatmap.")
        return

    pivot = heatmap_filtered.pivot_table(
        index='antecedent',
        columns='consequent',
        values='lift',
        aggfunc='mean'
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.2f',
        cmap='YlGnBu',
        linewidths=0.5,
        square=True
    )
    plt.title('Heatmap Korelasi Layanan (Lift Ratio)', fontsize=14, fontweight='bold')
    plt.xlabel('Consequent (Layanan Tujuan)', fontsize=11)
    plt.ylabel('Antecedent (Layanan Asal)', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_top_rules(rules, top_n=10, save_path=None):
    """
    Membuat Horizontal Bar Chart untuk Top N Association Rules berdasarkan Lift.

    Parameters:
        rules (pd.DataFrame): Tabel association rules.
        top_n (int): Jumlah aturan teratas yang ditampilkan.
        save_path (str): Path untuk menyimpan gambar (opsional).
    """
    top_rules = rules.head(top_n).copy()
    top_rules['rule_label'] = top_rules.apply(
        lambda row: f"{', '.join(list(row['antecedents']))} → {', '.join(list(row['consequents']))}",
        axis=1
    )

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=top_rules,
        y='rule_label',
        x='lift',
        palette='magma',
        orient='h'
    )

    for i, (_, row) in enumerate(top_rules.iterrows()):
        ax.text(row['lift'] + 0.02, i,
                f"Lift: {row['lift']:.2f}",
                va='center', fontsize=9, fontweight='bold')

    plt.title(f'Top {top_n} Association Rules (Berdasarkan Lift Ratio)', fontsize=14, fontweight='bold')
    plt.xlabel('Lift Ratio', fontsize=11)
    plt.ylabel('Aturan Asosiasi', fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_network(rules, top_n=10, save_path=None):
    """
    Membuat Network Graph (Peta Arus Kasir) dari Top Association Rules.
    Node = Layanan, Edge = Aturan asosiasi, Ketebalan edge = nilai Lift.

    Parameters:
        rules (pd.DataFrame): Tabel association rules.
        top_n (int): Jumlah aturan teratas yang divisualisasikan.
        save_path (str): Path untuk menyimpan gambar (opsional).
    """
    top_rules = rules.head(top_n)

    G = nx.DiGraph()

    for _, row in top_rules.iterrows():
        ante = ', '.join(list(row['antecedents']))
        cons = ', '.join(list(row['consequents']))
        G.add_edge(ante, cons, weight=row['lift'], confidence=row['confidence'])

    plt.figure(figsize=(12, 9))

    pos = nx.spring_layout(G, k=2.5, seed=42)
    edges = G.edges(data=True)
    weights = [d['weight'] for _, _, d in edges]

    # Normalisasi ketebalan garis
    max_w = max(weights) if weights else 1
    min_w = min(weights) if weights else 0
    edge_widths = [1 + 4 * (w - min_w) / (max_w - min_w + 0.01) for w in weights]

    nx.draw_networkx_nodes(G, pos, node_color='#00796B', node_size=2500, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_color='white')
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='#E64A19',
                           arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.15',
                           alpha=0.8)

    # Label edge dengan nilai Lift
    edge_labels = {(u, v): f"Lift: {d['weight']:.2f}" for u, v, d in edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title('Network Graph: Peta Arus Rekomendasi Kasir', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_summary(rules):
    """
    Mencetak rangkuman kesimpulan dari hasil Association Rules ke terminal.

    Parameters:
        rules (pd.DataFrame): Tabel association rules.
    """
    print("\n" + "=" * 60)
    print(" RANGKUMAN KESIMPULAN MARKET BASKET ANALYSIS")
    print("=" * 60)

    if len(rules) == 0:
        print("Tidak ada aturan asosiasi yang ditemukan.")
        return

    top = rules.iloc[0]
    ante_str = ', '.join(list(top['antecedents']))
    cons_str = ', '.join(list(top['consequents']))

    print(f"\n Golden Rule (Asosiasi Terkuat):")
    print(f"   {ante_str} -> {cons_str}")
    print(f"   Support    : {top['support']:.4f}")
    print(f"   Confidence : {top['confidence']:.4f}")
    print(f"   Lift       : {top['lift']:.4f}")

    print(f"\n Total aturan asosiasi yang ditemukan: {len(rules)} aturan")
    print(f" Aturan dengan Lift > 1 (valid)     : {len(rules[rules['lift'] > 1])} aturan")

    print("\n Rekomendasi Bisnis:")
    print("   1. Layanan Cuci Kering & Setrika adalah Cash Cow utama.")
    print("   2. Terapkan promo 'Bed Room Bundle' (Sprei + Bed Cover).")
    print("   3. Hindari bundling layanan yang bersifat substitusi (Lift < 1).")
    print("=" * 60)
