# 🧊 Antarctic AI Navigation & Iceberg Trajectory Prediction System
### Smart India Hackathon (SIH) Prototype

An end-to-end AI-powered maritime navigation system leveraging the **BYU/NIC Antarctic Iceberg Tracking Database**, **PyTorch GRU deep learning** for trajectory forecasting, a **Gaussian collision risk engine**, and **A\* pathfinding** for safe polar route optimization — served via a **FastAPI + Leaflet interactive geospatial dashboard**.

---

## 🚀 Quick Start

### 1. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

### 2. Run the Data Preprocessing Pipeline
```bash
PYTHONPATH=. ./venv/bin/python src/preprocessing.py \
  --raw_dir stats_database_v7.1 \
  --output data/processed/iceberg_tracks_clean.csv \
  --top_n 75
```
**Output:** `data/processed/iceberg_tracks_clean.csv` (243,433 trajectory points from 75 high-fidelity tracks)

### 3. Train the PyTorch GRU Model
```bash
PYTHONPATH=. ./venv/bin/python src/train_gru.py \
  --epochs 10 \
  --batch_size 128
```
**Output:** `models/gru_iceberg.pt` checkpoint + `models/feature_scaler.pkl`  
**Hardware:** Automatically utilizes Apple Silicon MPS GPU acceleration.

### 4. Evaluate Models & Generate Benchmark Plots
```bash
PYTHONPATH=. ./venv/bin/python src/evaluate.py
```
**Output:** `models/eval_metrics.json` + `artifacts/trajectory_evaluation.png`

### 5. Test Route Optimizer (Self-Contained Demo)
```bash
PYTHONPATH=. ./venv/bin/python src/route_optimizer.py --test
```

### 6. Start the Interactive Web Dashboard
```bash
PYTHONPATH=. ./venv/bin/uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```
Open **http://127.0.0.1:8000** in your browser.

---

## 📊 Benchmark Results (on Test Set — 12 Icebergs, 46,576 Windows)

| Horizon | Model | Mean Haversine Error | Median Error |
|---------|-------|---------------------|--------------|
| +24h | Constant Velocity Baseline | 2.86 km (1.55 NM) | 0.32 km |
| +24h | **PyTorch GRU** | **2.25 km (1.21 NM)** | **0.17 km** |
| +48h | Constant Velocity Baseline | 5.16 km (2.78 NM) | 0.95 km |
| +48h | **PyTorch GRU** | **3.80 km (2.05 NM)** | **0.47 km** |

**GRU improves over Constant Velocity baseline by: +21.3% (24h) / +26.4% (48h)**

---

## 📁 Project Structure

```
ml dataset/
├── data/
│   ├── raw/                           # Downloaded BYU/NIC CSV archives
│   └── processed/
│       ├── iceberg_tracks_clean.csv   # Unified clean 75-track dataset
│       └── metadata_summary.json      # Dataset stats & iceberg catalog
├── models/
│   ├── gru_iceberg.pt                 # PyTorch GRU checkpoint (best epoch)
│   ├── feature_scaler.pkl             # StandardScaler for input features
│   ├── train_history.json             # Per-epoch loss curves
│   └── eval_metrics.json              # Full benchmark comparison
├── src/
│   ├── preprocessing.py               # Data pipeline: download → clean → features
│   ├── features.py                    # Sequence windows, scaling, Datasets
│   ├── baseline.py                    # Constant Velocity dead-reckoning model
│   ├── train_gru.py                   # PyTorch GRU architecture & training loop
│   ├── evaluate.py                    # Multi-horizon metrics & trajectory plots
│   └── route_optimizer.py             # Risk grid & A* maritime pathfinding
├── app/
│   ├── backend/
│   │   └── main.py                    # FastAPI REST server
│   └── frontend/
│       ├── index.html                 # Polar Leaflet map dashboard
│       ├── style.css                  # Dark mode glassmorphism UI
│       └── app.js                     # API connectors & map interactions
├── artifacts/
│   └── trajectory_evaluation.png      # Benchmark plot (actual vs pred)
├── requirements.txt
└── README.md
```

---

## 🌊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Server health & model loading status |
| `GET` | `/api/stations` | Antarctic research stations & ports |
| `GET` | `/api/icebergs` | Full iceberg catalog with latest positions |
| `GET` | `/api/icebergs/{id}/track` | Historical trajectory for a specific iceberg |
| `POST` | `/api/predict` | GRU vs Baseline 24h/48h forecast |
| `POST` | `/api/route/optimize` | A* optimal collision-free route |
| `POST` | `/api/route/replan` | Dynamic rerouting around encroaching icebergs |
| `GET` | `/api/benchmarks` | Full model evaluation metrics JSON |
| `GET` | `/artifacts/trajectory_evaluation.png` | Benchmark trajectory plot |

---

## 🧠 Model Architecture

**PyTorch GRU Sequence Model:**
- **Input:** 13-dimensional daily feature vectors (lat, lon, vx, vy, speed, sin/cos heading, sin/cos DOY, wind_u, wind_v, curr_u, curr_v)
- **Sequence Length:** 14 days sliding window
- **Architecture:** Input FC layer → 2-layer GRU (hidden=128) → Dual pooling head → Multi-horizon regressor
- **Output:** Displacement deltas `[Δlat₁, Δlon₁, Δlat₂, Δlon₂]` for +24h and +48h
- **Training:** AdamW optimizer, SmoothL1 (Huber) loss, Cosine Annealing LR, gradient clipping, MPS acceleration

**Constant Velocity Baseline:**
- Spherical dead-reckoning using last observed speed and heading via great-circle projection

---

## 🗺️ Route Optimizer

**Risk Grid:**
- 0.5° × 0.5° spatial grid covering [-78°, -50°S] × [-180°, 180°E]
- Gaussian kernel hazard density per iceberg with expanding σ over forecast horizon

**A\* Pathfinder:**
- 8-connected polar navigation graph
- Combined cost: `dist × (1 + λ_fuel × sea_state) + λ_risk × Risk²`
- Antarctic landmask avoidance, antimeridian wrap-around
- Dynamic replanning when icebergs drift within 25 km safety perimeter

---

## 🇮🇳 Indian Antarctic Stations

| Station | Coordinates | Region |
|---------|-------------|--------|
| **Maitri** | 70.767°S, 11.733°E | Queen Maud Land |
| **Bharati** | 69.407°S, 76.187°E | Larsemann Hills |

---

## 📦 Dependencies

```
torch>=2.0      # PyTorch GRU model + MPS acceleration
numpy, pandas   # Data processing
scikit-learn    # StandardScaler
matplotlib      # Trajectory visualization plots
fastapi         # REST API server
uvicorn         # ASGI server
scipy           # Spatial computations
requests        # BYU dataset auto-download
```

---

## 🔮 Future Enhancements

- Integrate real ERA5 reanalysis wind/ocean current grids
- LSTM attention mechanism for long-range seasonal dependencies
- Ensemble model (GRU + physics-based ocean model)
- 3D iceberg keel depth estimation for draft-based collision risk
- Automatic AIS vessel data integration for live traffic awareness
