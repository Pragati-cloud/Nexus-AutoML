from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from automl.engine import run_automl_pipeline

app = FastAPI()

# Enable CORS for frontend with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "message": "Backend is running"}

@app.post("/automl")
async def run_automl(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    try:

        df = pd.read_csv(file.file)

        output = run_automl_pipeline(df, target_column)

        results = output["results"]

        model_scores = {
            name: score for name,(model,score) in results.items()
        }

        return {
            "best_model": output["best_model"],
            "best_score": output["best_score"],
            "models": model_scores,
            "report": output["report"]
        }

    except Exception as e:
        print(f"Error: {str(e)}")  # Log to console
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__}
        )

app.mount("/", StaticFiles(directory="automl-ui", html=True), name="ui")
