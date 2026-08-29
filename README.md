# SPK Crypto Anomaly Screener (Binance Futures USDT-M)

Sistem Pendukung Keputusan (SPK) untuk mendeteksi anomali pada pasar Binance Futures USDT-M berbasis Python dengan metode **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** dan arsitektur **$0 Cost Infrastructure** (menggunakan endpoint publik Binance tanpa memerlukan API Key).

---

## 📁 Struktur Direktori Proyek

```text
spk-crypto-anomaly-screener/
├── src/
│   ├── __init__.py           # Package namespace
│   ├── ingestion.py          # Modul Ingestion 5 endpoint Binance Futures + Resilient DoH
│   ├── prefilter.py          # Pre-filter Universe (Log-Volume, Volatilitas, Hard Filters)
│   ├── features.py           # Ekstraksi 5 kriteria fitur anomali (C1 - C5)
│   ├── topsis.py             # TOPSIS Multi-Criteria Decision Engine
│   ├── database.py           # SQLite State Management & Alert Cooldown Machine
│   └── telegram_bot.py       # Telegram Alert Message Formatter & HTTP Dispatcher
├── main.py                   # Main Pipeline Orchestrator & CLI Entrypoint
├── requirements.txt          # Dependensi (requests, pandas, numpy, python-dotenv)
├── test_phase1.py            # Test Ingestion Phase 1
├── test_phase2.py            # Test Feature & TOPSIS Phase 2
├── test_phase3.py            # Test State Machine & Telegram Phase 3
├── .env.example              # Template konfigurasi environment
├── .gitignore                # Standar Python gitignore
└── README.md                 # Dokumentasi proyek
```

---

## 📐 Kriteria SPK & Bobot TOPSIS

| Kode | Kriteria | Tipe | Bobot | Deskripsi |
| :--- | :--- | :---: | :---: | :--- |
| **C1** | **Funding Rate (%)** | *Cost* | 0.25 | Tingkat suku bunga pendanaan Binance Futures. Suku bunga negatif/rendah lebih disukai (potensi short squeeze). |
| **C2** | **4H Delta Open Interest (%)** | *Benefit* | 0.25 | Pertumbuhan jumlah posisi terbuka (OI) dalam jendela waktu 4 jam terakhir. |
| **C3** | **1H Bollinger Band Width (%)** | *Cost* | 0.20 | Rasio lebar Bollinger Band 1 jam $(4\sigma / \text{SMA})$. Nilai rendah mengindikasikan kompresi volatilitas sebelum ekspansi. |
| **C4** | **Depth Imbalance Ratio ($\pm 2\%$)** | *Benefit* | 0.15 | Rasio kedalaman likuiditas Bid terhadap Ask dalam rentang $\pm 2\%$ dari harga tengah orderbook. |
| **C5** | **Volume / OI Velocity Ratio** | *Benefit* | 0.15 | Rasio volume quote 1 jam terhadap total nilai Open Interest (mengukur perputaran likuiditas). |

---

## 🚀 Setup & Instalasi

### 1. Prasyarat
- Python 3.10+
- Pip / virtual environment

### 2. Instalasi Dependensi
```powershell
pip install -r requirements.txt
```

### 3. Konfigurasi Environment (`.env`)
Salin file template `.env.example` menjadi `.env`:
```env
TELEGRAM_BOT_TOKEN=8525028603:AAFhu1NMnqo897Z02dRBJZCQQBD95eMt7CQ
TELEGRAM_CHAT_ID=6936822675
MIN_CI_ALERT_THRESHOLD=0.65
COOLDOWN_HOURS=4
DELTA_BYPASS=0.15
SCAN_LIMIT=25
```

---

## 🧪 Menjalankan Pengujian per Fase

```powershell
# Phase 1: Validasi Koneksi & 5 Endpoint Ingestion Binance Futures
python test_phase1.py

# Phase 2: Unit Test TOPSIS & Live Integration Screening Top Universe
python test_phase2.py

# Phase 3: Unit Test SQLite State Machine, Cooldown, dan Dispatcher Telegram
python test_phase3.py
```

---

## ⚡ Menjalankan Pipeline Orchestrator

```powershell
# 1. Menjalankan Pemindaian Live (Otomatis mengirim alert Telegram jika ada anomali)
python main.py

# 2. Menjalankan dalam Mode Simulasi (Dry Run tanpa mengirim notifikasi Telegram)
python main.py --dry-run

# 3. Kustomisasi Parameter Eksekusi via CLI
python main.py --limit 30 --threshold 0.70 --cooldown 6.0 --bypass 0.20
```
