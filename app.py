"""
app.py — Real Estate Search & Price Prediction API
====================================================
Stack:
  - FastAPI       : REST endpoints
  - KDTree        : custom in-memory spatial index (kdtree.py) — NO scipy/rtree
  - scikit-learn  : RandomForestRegressor for price prediction
  - supabase-py   : PostgreSQL (Supabase) data source

Schema (King County House Sales dataset):
  price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, yr_built, lat, long

Endpoints:
  GET  /health     → liveness check + tree/model status
  GET  /search     → radius search via KD-Tree (Haversine)
  GET  /knn        → k-nearest-neighbour search via KD-Tree
  POST /predict    → price prediction via Random Forest
  GET  /metrics    → ML model evaluation metrics (R², RMSE, MAE, feature importance)
  POST /sync       → reload data from Supabase & rebuild tree + retrain model
  GET  /benchmark  → KD-Tree vs Brute-Force speed comparison
"""

import os
import time
import logging
import random
import re
import math
import pandas as pd
import kagglehub
import httpx
from bs4 import BeautifulSoup
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from supabase import create_client, Client

from kdtree import KDTree, Property
from structures import PropertyQueue, ScraperUndoStack

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("realestate_api")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

# ---------------------------------------------------------------------------
# Global in-memory state
# ---------------------------------------------------------------------------

kd_tree: KDTree = KDTree()
rf_model: Optional[RandomForestRegressor] = None
model_metrics: Dict[str, Any] = {}          # populated after training
feature_scales: Dict[str, float] = {}       # populated after training for IDW
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Scraper Data Structures
scrape_queue = PropertyQueue()
undo_stack = ScraperUndoStack()

# Load Real Dataset for Scraper Simulation
scraper_df = None
try:
    _path = kagglehub.dataset_download("harlfoxem/housesalesprediction")
    _csv_path = os.path.join(_path, "kc_house_data.csv")
    scraper_df = pd.read_csv(_csv_path)
except Exception as e:
    logger.error("Failed to load Kaggle dataset for scraper: %s", e)

# ---------------------------------------------------------------------------
# ML Feature configuration
# ---------------------------------------------------------------------------

# Features used for RandomForest training & prediction
# NOTE: these must exactly match the column names in the Property dataclass
FEATURE_NAMES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "yr_built",
    "lat",
    "long",
]
TARGET = "price"

# ---------------------------------------------------------------------------
# Supabase helpers
# ---------------------------------------------------------------------------

def fetch_all_properties() -> List[Property]:
    """
    Pull every row from the `properties` table in Supabase.
    Malformed rows are skipped with a warning.
    """
    logger.info("Fetching all properties from Supabase …")

    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        response = (
            supabase.table("properties")
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = response.data
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    properties: List[Property] = []
    skipped = 0

    for row in all_rows:
        try:
            prop = Property(
                id=str(row["id"]),
                price=float(row["price"]),
                bedrooms=float(row["bedrooms"]),
                bathrooms=float(row["bathrooms"]),
                sqft_living=float(row["sqft_living"]),
                sqft_lot=float(row["sqft_lot"]),
                floors=float(row["floors"]),
                yr_built=float(row["yr_built"]),
                lat=float(row["lat"]),
                long=float(row["long"]),
            )
            properties.append(prop)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping malformed row id=%s  reason=%s", row.get("id"), exc)
            skipped += 1

    logger.info("Fetched %d valid properties (%d skipped).", len(properties), skipped)
    return properties


# ---------------------------------------------------------------------------
# Tree + Model management
# ---------------------------------------------------------------------------

def rebuild_tree_and_retrain(properties: List[Property]) -> None:
    """
    1. Build a balanced KD-Tree from the property list.
    2. Train a RandomForestRegressor and compute evaluation metrics.
    3. Compute feature standard deviations for hybrid predictions.
    Both operations mutate module-level globals.
    """
    global kd_tree, rf_model, model_metrics, feature_scales

    # ── KD-Tree ──────────────────────────────────────────────────────────────
    kd_tree = KDTree()
    kd_tree.build(properties)
    logger.info("KD-Tree built: %s", kd_tree)

    # ── Random Forest ────────────────────────────────────────────────────────
    MIN_SAMPLES = 50
    if len(properties) < MIN_SAMPLES:
        logger.warning(
            "Only %d records — skipping ML training (need ≥ %d).",
            len(properties), MIN_SAMPLES,
        )
        return

    X = np.array(
        [[getattr(p, f) for f in FEATURE_NAMES] for p in properties],
        dtype=np.float64,
    )
    y = np.array([getattr(p, TARGET) for p in properties], dtype=np.float64)

    # Train / test split for unbiased metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    # Evaluate on held-out test set
    y_pred = rf_model.predict(X_test)
    rmse   = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae    = float(mean_absolute_error(y_test, y_pred))
    r2     = float(r2_score(y_test, y_pred))

    model_metrics = {
        "r2_score":          round(r2,   4),
        "rmse":              round(rmse, 2),
        "mae":               round(mae,  2),
        "oob_score":         round(float(rf_model.oob_score_), 4),
        "train_samples":     len(X_train),
        "test_samples":      len(X_test),
        "feature_importance": {
            name: round(float(imp), 6)
            for name, imp in zip(FEATURE_NAMES, rf_model.feature_importances_)
        },
    }

    # ── Feature Standard Deviations ──────────────────────────────────────────
    for f in ["bedrooms", "bathrooms", "sqft_living", "yr_built"]:
        std_val = float(np.std([getattr(p, f) for p in properties]))
        feature_scales[f] = std_val if std_val > 0.0 else 1.0

    logger.info(
        "RandomForest trained — R²=%.4f  RMSE=%.0f  MAE=%.0f  OOB=%.4f",
        r2, rmse, mae, rf_model.oob_score_,
    )


# ---------------------------------------------------------------------------
# FastAPI Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== API STARTUP ===")
    try:
        props = fetch_all_properties()
        rebuild_tree_and_retrain(props)
        logger.info("Startup complete — %d properties indexed.", len(props))
    except Exception as exc:
        logger.error("Startup data load failed: %s", exc)
    yield
    logger.info("=== API SHUTDOWN ===")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Real Estate Search & Price Prediction API",
    description=(
        "Spatial property search powered by a **custom KD-Tree** (Haversine distance, "
        "built from scratch — no scipy/rtree) and price prediction via "
        "**Random Forest Regressor**. Data: King County House Sales dataset (~21K records) "
        "sourced from Supabase."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class PropertyResponse(BaseModel):
    id: str
    price: float
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    yr_built: float
    lat: float
    long: float
    distance_km: Optional[float] = Field(None, description="Distance from query point (km)")


class PredictRequest(BaseModel):
    lat: float         = Field(..., example=47.5112)
    long: float        = Field(..., example=-122.257)
    bedrooms: float    = Field(..., ge=0, example=3.0)
    bathrooms: float   = Field(..., ge=0, example=2.25)
    sqft_living: float = Field(..., gt=0, example=2570.0)
    yr_built: float    = Field(..., gt=0, example=1951.0)


class PredictResponse(BaseModel):
    predicted_price_usd: float
    input_features: PredictRequest
    rbf_weight: float = Field(0.0, description="Sum of RBF weights used to blend the local prediction")
    comparable_properties: list[dict] = Field([], description="List of top comparable coordinates and data")


class SyncResponse(BaseModel):
    status: str
    records_loaded: int
    tree_height: int
    message: str


class BenchmarkResult(BaseModel):
    dataset_size: int
    query_lat: float
    query_lon: float
    radius_km: float
    kdtree_ms: float
    brute_force_ms: float
    speedup_x: float
    kdtree_results: int
    brute_force_results: int
    results_match: bool


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _require_tree() -> None:
    if kd_tree.root is None:
        raise HTTPException(
            status_code=503,
            detail="KD-Tree not initialised. Call POST /sync to load data.",
        )


def _require_model() -> None:
    if rf_model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model not trained yet. Call POST /sync to load data.",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness / readiness check", tags=["System"])
def health():
    return {
        "status":        "ok",
        "tree_size":     kd_tree.size,
        "tree_height":   kd_tree.height(),
        "model_trained": rf_model is not None,
        "model_r2":      model_metrics.get("r2_score"),
    }


@app.get(
    "/search",
    response_model=List[PropertyResponse],
    summary="Radius search — all properties within N km",
    tags=["Spatial Search"],
)
def search(
    lat:       float = Query(..., description="Centre latitude"),
    lon:       float = Query(..., description="Centre longitude"),
    radius_km: float = Query(5.0, gt=0, description="Search radius in km"),
    limit:     int   = Query(50, ge=1, le=500, description="Max results"),
):
    """
    Queries the in-memory KD-Tree and returns all properties whose
    Haversine distance from `(lat, lon)` is ≤ `radius_km`.
    Results are sorted by ascending distance and capped at `limit`.
    """
    _require_tree()

    raw = kd_tree.search_within_radius(lat, lon, radius_km)
    raw.sort(key=lambda p: kd_tree.haversine(lat, lon, p.lat, p.long))
    raw = raw[:limit]

    return [
        PropertyResponse(
            **p.__dict__,
            distance_km=round(kd_tree.haversine(lat, lon, p.lat, p.long), 4),
        )
        for p in raw
    ]


@app.get(
    "/knn",
    response_model=List[PropertyResponse],
    summary="k-Nearest Neighbours — k closest properties",
    tags=["Spatial Search"],
)
def knn(
    lat: float = Query(..., description="Query latitude"),
    lon: float = Query(..., description="Query longitude"),
    k:   int   = Query(10, ge=1, le=200, description="Number of neighbours"),
):
    """
    Runs a k-nearest-neighbour query on the KD-Tree using Haversine distance.
    Returns at most `k` properties sorted by ascending distance.
    """
    _require_tree()

    results = kd_tree.k_nearest_neighbors(lat, lon, k)
    return [
        PropertyResponse(
            **p.__dict__,
            distance_km=round(dist_km, 4),
        )
        for dist_km, p in results
    ]


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict property median house value (USD)",
    tags=["ML Prediction"],
)
def predict(request: PredictRequest):
    """
    Predicts `price` using the trained Random Forest Regressor.
    Features: bedrooms, bathrooms, sqft_living, yr_built, lat, long.
    """
    _require_model()

    features = np.array(
        [[getattr(request, f) for f in FEATURE_NAMES]],
        dtype=np.float64,
    )
    rf_pred = float(rf_model.predict(features)[0])

    # 1. Spatial Filter: Fetch 30 nearest geographic neighbors
    neighbors = kd_tree.k_nearest_neighbors(request.lat, request.long, 30)
    
    weights = []
    prices = []
    
    # Pre-fetch dynamic scales built in /sync
    std_beds = feature_scales.get("bedrooms", 1.0)
    std_baths = feature_scales.get("bathrooms", 1.0)
    std_sqft = feature_scales.get("sqft_living", 1.0)
    std_yr = feature_scales.get("yr_built", 1.0)

    # Gamma decay parameters
    gamma_spatial = 2.0  # km
    gamma_feature = 1.0  # std devs
    
    for dist_km, prop in neighbors:
        # A. Spatial Distance Penalty (Gaussian RBF)
        w_spatial = math.exp(-(dist_km ** 2) / (2 * (gamma_spatial ** 2)))
        
        # B. Feature Similarity Penalty (Scaled)
        d_beds  = abs(request.bedrooms - prop.bedrooms) / std_beds
        d_baths = abs(request.bathrooms - prop.bathrooms) / std_baths
        d_sqft  = abs(request.sqft_living - prop.sqft_living) / std_sqft
        d_yr    = abs(request.yr_built - prop.yr_built) / std_yr
        
        feat_dist = d_beds + d_baths + d_sqft + d_yr
        w_feature = math.exp(-(feat_dist ** 2) / (2 * (gamma_feature ** 2)))
        
        # C. Combined weight
        w_total = w_spatial * w_feature
        weights.append(w_total)
        prices.append(prop.price)
        
    sum_w = sum(weights)
    local_pred = 0.0
    alpha = 0.0
    
    comparables = []

    if sum_w > 0:
        local_pred = sum(w * p for w, p in zip(weights, prices)) / sum_w
        # D. Dynamic Alpha Weighting
        # If sum_w is high (many close and similar neighbors), alpha goes up to 0.7 max
        alpha = min(0.7, sum_w / 10.0) 
        predicted = (alpha * local_pred) + ((1.0 - alpha) * rf_pred)
        
        # E. Extract Top Comparables
        # Pair weights with properties
        neighbor_pairs = list(zip(weights, [n[1] for n in neighbors]))
        # Sort by weight descending
        neighbor_pairs.sort(key=lambda x: x[0], reverse=True)
        # Take top 3-5 (limit to those with meaningful weight)
        top_pairs = [pair for pair in neighbor_pairs if pair[0] > 0.05][:5]
        
        comparables = [
            {
                "lat": p.lat,
                "long": p.long,
                "price": p.price,
                "bedrooms": p.bedrooms,
                "bathrooms": p.bathrooms,
                "sqft_living": p.sqft_living
            }
            for _, p in top_pairs
        ]
    else:
        predicted = rf_pred

    return PredictResponse(
        predicted_price_usd=round(predicted, 2),
        input_features=request,
        rbf_weight=round(alpha, 4),
        comparable_properties=comparables
    )


@app.get(
    "/metrics",
    summary="ML model evaluation metrics",
    tags=["ML Prediction"],
)
def metrics():
    """
    Returns R², RMSE, MAE, OOB score and feature importances
    computed on a 20% held-out test set during the last /sync.
    """
    _require_model()
    return model_metrics


@app.post(
    "/sync",
    response_model=SyncResponse,
    summary="Reload data from Supabase — rebuild tree & retrain model",
    tags=["System"],
)
def sync():
    """
    Fetches the latest data from Supabase, rebuilds the KD-Tree,
    and retrains the Random Forest — all in-memory, zero downtime.
    """
    try:
        props = fetch_all_properties()
        rebuild_tree_and_retrain(props)
        return SyncResponse(
            status="success",
            records_loaded=len(props),
            tree_height=kd_tree.height(),
            message=(
                f"KD-Tree rebuilt ({len(props)} nodes, height {kd_tree.height()}) "
                f"and RandomForest retrained. R²={model_metrics.get('r2_score', 'N/A')}"
            ),
        )
    except Exception as exc:
        logger.error("Sync failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Sync failed: {exc}") from exc


@app.get(
    "/benchmark",
    response_model=BenchmarkResult,
    summary="KD-Tree vs Brute-Force speed comparison",
    tags=["System"],
)
def benchmark(
    lat:       float = Query(37.5, description="Query latitude"),
    lon:       float = Query(-122.0, description="Query longitude"),
    radius_km: float = Query(50.0, gt=0, description="Search radius in km"),
):
    """
    Compares KD-Tree radius search against a brute-force linear scan
    over the entire in-memory dataset. Returns timings in milliseconds
    and a speedup factor to demonstrate algorithmic efficiency.
    """
    _require_tree()

    # ── KD-Tree search ───────────────────────────────────────────────────────
    t0 = time.perf_counter()
    kd_results = kd_tree.search_within_radius(lat, lon, radius_km)
    kdtree_ms  = (time.perf_counter() - t0) * 1000

    # ── Brute-Force search (collect all nodes via in-order traversal) ─────────
    all_props: List[Property] = []

    def _collect(node):
        if node is None:
            return
        _collect(node.left)
        all_props.append(node.property)
        _collect(node.right)

    _collect(kd_tree.root)

    t1 = time.perf_counter()
    bf_results = [
        p for p in all_props
        if KDTree.haversine(lat, lon, p.lat, p.long) <= radius_km
    ]
    brute_ms = (time.perf_counter() - t1) * 1000

    speedup = round(brute_ms / kdtree_ms, 2) if kdtree_ms > 0 else 0.0

    return BenchmarkResult(
        dataset_size=kd_tree.size,
        query_lat=lat,
        query_lon=lon,
        radius_km=radius_km,
        kdtree_ms=round(kdtree_ms, 3),
        brute_force_ms=round(brute_ms, 3),
        speedup_x=speedup,
        kdtree_results=len(kd_results),
        brute_force_results=len(bf_results),
        results_match=(len(kd_results) == len(bf_results)),
    )

# ---------------------------------------------------------------------------
# Live Scraper Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/scrape/trigger",
    summary="Scrape real listings from Craigslist Seattle",
    tags=["Live Scraper"],
)
def trigger_scrape():
    """Scrapes live real estate listings from Craigslist Seattle and enqueues them."""
    
    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    CL_LIST_URL = "https://seattle.craigslist.org/d/real-estate-for-sale/search/rea"
    
    scraped = []
    
    try:
        resp = httpx.get(CL_LIST_URL, headers=HEADERS, timeout=15, follow_redirects=True)
        soup = BeautifulSoup(resp.text, "html.parser")
        listings = soup.select("li.cl-static-search-result")
        random.shuffle(listings)  # randomize so each trigger fetches different ones
        
        for li in listings:
            if len(scraped) >= 15:
                break
            
            link_el = li.select_one("a")
            price_el = li.select_one(".price")
            loc_el = li.select_one(".location")
            title = li.get("title", "")
            title_lower = title.lower()
            if any(word in title_lower for word in ["lot", "land", "acre", "acres", "parcel"]):
                continue
            
            if not link_el or not price_el:
                continue
            
            link = link_el["href"]
            price_raw = re.sub(r"[^\d]", "", price_el.text)
            if not price_raw:
                continue
            price = float(price_raw)
            if price < 50000 or price > 10000000:  # filter outliers
                continue
            
            # Fetch detail page for lat/lon and housing attrs
            try:
                dr = httpx.get(link, headers=HEADERS, timeout=10, follow_redirects=True)
                dsoup = BeautifulSoup(dr.text, "html.parser")
                
                map_el = dsoup.select_one("#map")
                if not map_el:
                    continue
                lat = float(map_el.get("data-latitude", 0))
                lon = float(map_el.get("data-longitude", 0))
                if not (-180 <= lon <= 180) or not (-90 <= lat <= 90) or lat == 0:
                    continue
                
                # Parse beds, baths, sqft from housing text e.g. "/ 3br - 1800ft2 -"
                housing_el = dsoup.select_one(".housing")
                housing_text = housing_el.text if housing_el else ""
                
                beds_m = re.search(r"(\d+)br", housing_text)
                sqft_m = re.search(r"([\d,]+)ft2", housing_text.replace(",", ""))
                
                # If it doesn't explicitly state bedrooms or sqft, it's likely not a house
                if not beds_m or not sqft_m:
                    continue
                
                # Baths from full page text as Craigslist DOM structure varies
                baths_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:ba|bath|baths|bathrooms)\b", dsoup.text, re.IGNORECASE)
                baths = float(baths_m.group(1)) if baths_m else 1.0
                
                bedrooms = float(beds_m.group(1))
                sqft_living = float(sqft_m.group(1).replace(",", ""))
                
                prop = {
                    "price": price,
                    "bedrooms": bedrooms,
                    "bathrooms": baths,
                    "sqft_living": sqft_living,
                    "sqft_lot": sqft_living * 2,  # Craigslist rarely lists lot size
                    "floors": 1.0,
                    "yr_built": 2000.0,
                    "lat": lat,
                    "long": lon,
                    "source_url": link,
                    "source_title": title,
                }
                scraped.append(prop)
                scrape_queue.enqueue(prop)
                logger.info("Scraped: %s @ $%.0f", title[:50], price)
                
            except Exception as detail_err:
                logger.warning("Detail fetch failed for %s: %s", link, detail_err)
                continue
    
    except Exception as e:
        logger.error("Craigslist scrape failed: %s", e)
        # Fallback to Kaggle dataset sample
        if scraper_df is not None and not scraper_df.empty:
            sample = scraper_df.sample(n=10)
            for _, row in sample.iterrows():
                prop = {
                    "price": float(row["price"]),
                    "bedrooms": float(row["bedrooms"]),
                    "bathrooms": float(row["bathrooms"]),
                    "sqft_living": float(row["sqft_living"]),
                    "sqft_lot": float(row["sqft_lot"]),
                    "floors": float(row["floors"]),
                    "yr_built": float(row["yr_built"]),
                    "lat": float(row["lat"]),
                    "long": float(row["long"]),
                    "source_url": None,
                    "source_title": None,
                }
                scrape_queue.enqueue(prop)
            return {"message": "Craigslist unavailable. Loaded 10 from dataset fallback.", "queue_size": scrape_queue.size()}
    
    return {"message": f"{len(scraped)} live listings scraped from Craigslist and queued.", "queue_size": scrape_queue.size()}

@app.post(
    "/scrape/process",
    summary="Process queue and bulk insert to DB",
    tags=["Live Scraper"],
)
def process_queue():
    """Dequeues all mocked properties, inserts to Supabase, and updates Tree/Model."""
    if scrape_queue.is_empty():
        return {"message": "Queue is empty. Call /scrape/trigger first."}
    
    batch = []
    ui_enrichment = {}
    
    while not scrape_queue.is_empty():
        prop = scrape_queue.dequeue()
        # Temporarily store UI-only fields
        source_url = prop.pop("source_url", None)
        source_title = prop.pop("source_title", None)
        
        ui_enrichment[len(batch)] = {"source_url": source_url, "source_title": source_title}
        batch.append(prop)
        
    try:
        response = supabase.table("properties").insert(batch).execute()
        inserted_rows = response.data
        inserted_ids = [row["id"] for row in inserted_rows]
        
        # Re-inject UI fields for the frontend
        for i, row in enumerate(inserted_rows):
            if i in ui_enrichment:
                row["source_url"] = ui_enrichment[i]["source_url"]
                row["source_title"] = ui_enrichment[i]["source_title"]
        
        # Push to Undo Stack
        undo_stack.push(inserted_ids)
        
        # Resync Tree and Model
        props = fetch_all_properties()
        rebuild_tree_and_retrain(props)
        
        return {
            "message": f"Successfully processed and inserted {len(inserted_ids)} properties.",
            "undo_stack_size": undo_stack.size(),
            "new_tree_size": kd_tree.size,
            "inserted_properties": inserted_rows
        }
    except Exception as exc:
        logger.error("Failed to process queue to DB: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

@app.post(
    "/scrape/undo",
    summary="Undo the last processed scraper batch",
    tags=["Live Scraper"],
)
def undo_scrape():
    """Pops the last inserted batch IDs from the Stack and deletes them from DB."""
    if undo_stack.is_empty():
        return {"message": "Undo stack is empty. Nothing to undo."}
        
    try:
        last_batch_ids = undo_stack.pop()
        
        # Delete from Supabase
        supabase.table("properties").delete().in_("id", last_batch_ids).execute()
        
        # Resync
        props = fetch_all_properties()
        rebuild_tree_and_retrain(props)
        
        return {
            "message": f"Successfully undid last batch of {len(last_batch_ids)} properties.",
            "undo_stack_size": undo_stack.size(),
            "new_tree_size": kd_tree.size
        }
    except Exception as exc:
        logger.error("Failed to undo batch: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

# ---------------------------------------------------------------------------
# Serve Frontend
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
