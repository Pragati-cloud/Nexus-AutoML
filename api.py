from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import pandas as pd
import numpy as np
import asyncio
from functools import partial

from automl.engine import run_automl_pipeline

app = FastAPI()

# Enable CORS
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


@app.post("/automl")
async def run_automl(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    try:

        # 📂 Load dataset
        df = pd.read_csv(file.file)

        # 🔽 Reduce memory (better version)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = pd.to_numeric(df[col], downcast="float")

        # ⚠️ Limit dataset size
        if len(df) > 50000:
            return JSONResponse(
                status_code=400,
                content={"error": "Dataset too large (max 50k rows allowed)"}
            )

        # ⚡ Sampling for performance
        if len(df) > 20000:
            df = df.sample(15000, random_state=42)

        # ⏱️ Run pipeline with timeout
        loop = asyncio.get_event_loop()

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

        # 📊 Results
        results = output["results"]

        model_scores = {
            name: score for name, (model, score) in results.items()
        }

        # 🧹 Free memory
        del df

        return {
            "best_model": output["best_model"],
            "best_score": output["best_score"],
            "models": model_scores,
            "report": output["report"]
        }

    except Exception as e:
        import traceback
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )


# 🌐 Frontend
app.mount("/", StaticFiles(directory="automl-ui", html=True), name="ui")
