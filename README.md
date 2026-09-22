# Real Estate ML Tree & KD-Tree Scraper (Explainable PropTech)

This project is an end-to-end Explainable PropTech system for real-estate search and price prediction, delivered with a modern web UI.

## Features
- **Spatial Search:** Fast radius search with a custom KD-Tree (`O(√N + k)` under typical distribution assumptions).
- **Explainable Price Prediction (XAI):** Hybrid prediction pipeline (global Random Forest + local Gaussian RBF weighting) with confidence score and comparable properties.
- **Live Data Ingestion:** Async Craigslist scraper with custom FIFO queue + LIFO undo stack and batch insertion flow.
- **Modern UI:** Interactive Leaflet-based frontend with light/dark mode.

## Quick Start

### 1) Configure environment variables
Create your local `.env` from `.env.example` and fill in your own Supabase credentials:

```bash
cp .env.example .env
```

### 2) Start the app (Docker)

#### Windows
```cmd
start.bat
```

#### macOS / Linux
```bash
./start.sh
```

### 3) Access
- **Frontend UI:** [http://localhost:8080/](http://localhost:8080/)
- **Swagger docs:** [http://localhost:8080/docs](http://localhost:8080/docs)
- **Health check:** [http://localhost:8080/health](http://localhost:8080/health)

> On first startup, the app loads the dataset from Supabase, builds the in-memory KD-Tree, and trains the model. This can take roughly 30-90 seconds.

## Documentation
For architecture, algorithm, and system-level details, see [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md).
