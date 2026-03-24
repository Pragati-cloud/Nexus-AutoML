from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

import pandas as pd
import os
import asyncio
from functools import partial
from datetime import datetime
from uuid import uuid4
from urllib.parse import unquote
import joblib

from automl.engine import run_automl_pipeline

app = FastAPI()

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(BASE_DIR, "automl-ui")
MODELS_DIR = os.path.join(BASE_DIR, "trained_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# In-memory model registry
MODEL_REGISTRY = {}

# Redirect root → UI
@app.get("/")
def root():
    return RedirectResponse(url="/app")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/health")
async def health():
    return {"status": "ok", "message": "Backend is running"}

# -------------------------------
# 🚀 AutoML Endpoint (UPDATED)
# -------------------------------
@app.post("/automl")
async def run_automl(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    try:
        # 📂 Load dataset
        df = pd.read_csv(file.file)

        # ⚠️ Dataset size limit
        if len(df) > 50000:
            return JSONResponse(
                status_code=400,
                content={"error": "Dataset too large (max 50k rows allowed)"}
            )

        # ⚡ Sampling for performance
        if len(df) > 20000:
            df = df.sample(20000, random_state=42)

        loop = asyncio.get_event_loop()

        # ⏱️ Run pipeline with timeout + thread
        try:
            output = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    partial(run_automl_pipeline, df, target_column)
                ),
                timeout=60
            )
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=408,
                content={"error": "Processing timeout. Try smaller dataset."}
            )

        # 📊 Extract results
        results = output["results"]

        model_scores = {
            name: score for name, (model, score) in results.items()
        }

        best_model_name = output["best_model"]
        best_model_obj = results[best_model_name][0]

        # 💾 Save model
        run_id = str(uuid4())
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_filename = f"best_model_{timestamp}_{run_id}.pkl"
        model_path = os.path.join(MODELS_DIR, model_filename)

        joblib.dump(best_model_obj, model_path)

        MODEL_REGISTRY[run_id] = {
            "path": model_path,
            "filename": model_filename,
            "best_model": best_model_name
        }

        return {
            "best_model": best_model_name,
            "best_score": output["best_score"],
            "models": model_scores,
            "report": output["report"],
            "run_id": run_id,
            "model_download_url": f"/automl/download/{run_id}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )

# -------------------------------
# 📥 Download Model Endpoint
# -------------------------------
@app.get("/automl/download/{run_ref:path}")
async def download_best_model(run_ref: str):

    normalized = unquote(run_ref).strip().strip("/")
    if "/" in normalized:
        normalized = normalized.split("/")[-1]

    run_id = normalized
    model_info = MODEL_REGISTRY.get(run_id)

    if not model_info:
        raise HTTPException(status_code=404, detail="Model not found")

    if not os.path.exists(model_info["path"]):
        raise HTTPException(status_code=404, detail="Model file missing")

    return FileResponse(
        path=model_info["path"],
        filename=model_info["filename"],
        media_type="application/octet-stream"
    )

# -------------------------------
# 🌐 Frontend Mount
# -------------------------------
app.mount("/app", StaticFiles(directory=UI_PATH, html=True), name="ui")
