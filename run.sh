#!/bin/bash

echo "=================================================="
echo " Market Basket Analysis — Time Laundry"
echo " Algoritma Apriori (Unsupervised Learning)"
echo "=================================================="

# Optional: Cek dan install requirements
if [ -f requirements.txt ]; then
    echo " Menginstall dependensi dari requirements.txt..."
    pip install -r requirements.txt
fi

# Jalankan pipeline utama
echo " Menjalankan src/main.py..."
python3 src/main.py
