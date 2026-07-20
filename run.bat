@echo off
echo ==================================================
echo  Market Basket Analysis - Time Laundry
echo  Algoritma Apriori (Unsupervised Learning)
echo ==================================================

echo.
echo Memeriksa dan menginstall dependensi...
pip install -r requirements.txt

echo.
echo Menjalankan src\main.py...
python src\main.py

echo.
pause
