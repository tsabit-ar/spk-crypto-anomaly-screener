# SPK Crypto Anomaly Screener (Binance Futures USDT-M)

Sistem Pendukung Keputusan (SPK) untuk mendeteksi anomali pada pasar Binance Futures USDT-M berbasis Python dengan arsitektur **$0 Cost Infrastructure** (menggunakan endpoint publik Binance tanpa memerlukan API Key).

---

## 📁 Struktur Direktori

```text
spk-crypto-anomaly-screener/
├── src/
│   ├── __init__.py
│   └── ingestion.py      # Modul ingestion data Binance Futures USDT-M
├── requirements.txt      # Dependensi proyek
├── test_phase1.py        # Skrip pengujian validasi koneksi & parsing data Phase 1
└── README.md             # Dokumentasi proyek
```

---

## 🚀 Setup & Instalasi

### 1. Prasyarat
- Python 3.10+
- Pip / virtual environment

### 2. Instalasi Dependensi
```powershell
pip install -r requirements.txt
```

---

## 🧪 Menjalankan Pengujian (Phase 1)

Untuk memvalidasi koneksi ke seluruh 5 endpoint publik Binance Futures dan memastikan parsing data berjalan dengan benar:

```powershell
python test_phase1.py
```

### Endpoint yang Diuji:
1. `GET /fapi/v1/ticker/24hr` - Bulk 24h Ticker data seluruh koin USDT-M.
2. `GET /fapi/v1/premiumIndex` - Mark Price, Index Price, dan Funding Rate.
3. `GET /futures/data/openInterestHist` - Riwayat Open Interest (OI) per jam untuk menghitung 4H Delta OI.
4. `GET /fapi/v1/klines` - Data candlestick OHLCV 1-jam (20 candle terakhir).
5. `GET /fapi/v1/depth` - Orderbook Depth limit 100 untuk mengukur likuiditas ±2%.
