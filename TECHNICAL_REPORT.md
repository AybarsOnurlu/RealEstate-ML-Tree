# Explainable PropTech — Technical Architecture Report

**Project:** Real Estate Search & Price Prediction API  
**Version:** 2.0.0  
**Date:** 25 May 2026  
**Author:** Aybars Onurlu  
**Dataset:** King County House Sales (~21,613 records)  
**Runtime:** Docker (linux/amd64), AWS EC2 t3.large, Uvicorn/FastAPI

---

## 1. Executive Summary

This system was built as an end-to-end Explainable PropTech platform that combines:

1. Fast geospatial property lookup using a custom 2D KD-Tree.
2. Explainable price prediction using a hybrid ML approach.
3. Live listing ingestion with queue-based processing and undo support.

The backend runs on FastAPI/Uvicorn and serves both API endpoints and static frontend assets. Supabase (PostgreSQL) is the only external persistence dependency.

---

## 2. System Architecture

- **Frontend:** `frontend/index.html`, `frontend/app.js`, `frontend/style.css`
- **API Layer:** `app.py` (FastAPI)
- **Spatial Index:** `kdtree.py` (custom KD-Tree)
- **Custom Data Structures:** `structures.py` (linked-list queue/stack)
- **Storage:** Supabase `properties` table
- **Live Source:** Craigslist Seattle scraping flow

Main API endpoints:
- `/health`, `/search`, `/knn`, `/predict`, `/metrics`, `/sync`, `/benchmark`
- `/scrape/trigger`, `/scrape/process`, `/scrape/undo`

---

## 3. Data Structures

### KD-Tree
- Custom 2D KD-Tree storing property latitude/longitude.
- Supports balanced build, dynamic insert, radius search, and k-NN search.
- Distance metric: Haversine (great-circle distance).

### PropertyQueue (FIFO)
- Linked-list queue used in scraper ingestion before DB insertion.
- Operations are O(1) for enqueue/dequeue.

### ScraperUndoStack (LIFO)
- Linked-list stack storing inserted batch IDs.
- Enables reverting the latest processed scrape batch.

---

## 4. ML & Explainability

- **Global model:** Random Forest Regressor trained on structured house features.
- **Local weighting:** Gaussian RBF similarity around input point.
- **Final estimate:** Blend of global and local signals with confidence weighting.
- **Explainability output:** Comparable properties + confidence indicator (`rbf_weight`) in `/predict` response.

---

## 5. Performance Notes

- KD-Tree search significantly reduces average lookup cost compared to brute-force scans.
- Batch insertion in `/scrape/process` minimizes DB round-trips.
- Incremental KD-Tree updates reduce full rebuild pressure during live ingestion.

---

## 6. Frontend Behavior

- Leaflet map interaction for click-based search and prediction.
- Radius slider for spatial filtering.
- Prediction panel with confidence visualization and comparable listings.
- Live scraper admin controls for trigger/process/undo flows.

---

## 7. API Surface (High Level)

- **Search:** Radius and nearest-neighbor queries.
- **Prediction:** Feature-based price estimate + XAI output.
- **System:** Health, benchmark, sync, model metrics.
- **Ingestion:** Trigger scrape, process queue, undo last batch.

---

## 8. Data Quality and Public Release Notes

Before a public release:

1. Keep `.env` out of version control.
2. Validate Supabase rows for outliers (invalid coordinates, impossible prices, null critical fields).
3. Re-run the benchmark after major ingestion/model changes.
4. Confirm production keys are rotated if they were ever committed.

---

## 9. Future Improvements

- Add automated data quality checks in the ingestion pipeline.
- Add periodic model retraining schedules.
- Extend explainability output with feature-attribution summaries.
- Add CI-based smoke tests for scraper and API health.
