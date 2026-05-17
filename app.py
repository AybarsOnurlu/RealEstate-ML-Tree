"""
app.py — Real Estate Search & Price Prediction API
====================================================
Stack:
  - FastAPI       : REST endpoints
  - KDTree        : custom in-memory spatial index (kdtree.py) — NO scipy/rtree
  - scikit-learn  : RandomForestRegressor for price prediction
  - supabase-py   : PostgreSQL (Supabase) data source

Schema (California Housing dataset — full columns):
  longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
  population, households, median_income, median_house_value, ocean_proximity

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
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from supabase import create_client, Client

from kdtree import KDTree, Property

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
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------------------------
# ML Feature configuration
# ---------------------------------------------------------------------------

# Features used for RandomForest training & prediction
# NOTE: these must exactly match the column names in the Property dataclass
FEATURE_NAMES = [
    "latitude",
    "longitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
]
TARGET = "median_house_value"

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
                longitude=float(row["longitude"]),
                latitude=float(row["latitude"]),
                housing_median_age=float(row["housing_median_age"]),
                total_rooms=float(row["total_rooms"]),
                total_bedrooms=float(row["total_bedrooms"]),
                population=float(row["population"]),
                households=float(row["households"]),
                median_income=float(row["median_income"]),
                median_house_value=float(row["median_house_value"]),
                ocean_proximity=str(row.get("ocean_proximity", "")),
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
    Both operations mutate module-level globals.
    """
    global kd_tree, rf_model, model_metrics

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
    y = np.array([p.median_house_value for p in properties], dtype=np.float64)

    # Train / test split for unbiased metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_model = RandomForestRegressor(
        n_estimators=150,
        min_samples_leaf=2,
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
        "**Random Forest Regressor**. Data: California Housing dataset (~20K records) "
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
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    median_house_value: float
    ocean_proximity: str
    distance_km: Optional[float] = Field(None, description="Distance from query point (km)")


class PredictRequest(BaseModel):
    latitude: float          = Field(..., example=37.88)
    longitude: float         = Field(..., example=-122.23)
    housing_median_age: float = Field(..., ge=0, example=41.0)
    total_rooms: float        = Field(..., gt=0, example=880.0)
    total_bedrooms: float     = Field(..., gt=0, example=129.0)
    population: float         = Field(..., gt=0, example=322.0)
    households: float         = Field(..., gt=0, example=126.0)
    median_income: float      = Field(..., gt=0, example=8.3252)


class PredictResponse(BaseModel):
    predicted_price_usd: float
    input_features: PredictRequest


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
    raw.sort(key=lambda p: kd_tree.haversine(lat, lon, p.latitude, p.longitude))
    raw = raw[:limit]

    return [
        PropertyResponse(
            **p.__dict__,
            distance_km=round(kd_tree.haversine(lat, lon, p.latitude, p.longitude), 4),
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
    Predicts `median_house_value` using the trained Random Forest Regressor.
    Features: latitude, longitude, housing_median_age, total_rooms,
    total_bedrooms, population, households, median_income.
    """
    _require_model()

    features = np.array(
        [[getattr(request, f) for f in FEATURE_NAMES]],
        dtype=np.float64,
    )
    predicted = float(rf_model.predict(features)[0])

    return PredictResponse(
        predicted_price_usd=round(predicted, 2),
        input_features=request,
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
        if KDTree.haversine(lat, lon, p.latitude, p.longitude) <= radius_km
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
