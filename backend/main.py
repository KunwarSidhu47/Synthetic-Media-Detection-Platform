"""
Main FastAPI application entrypoint for Synthetic Media Detection & Analysis Platform.
"""

import os

# Set thread pool environment variables
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes import health, analyze, results, explain

app = FastAPI(
    title="Synthetic Media Detection & Analysis Platform",
    description="Multimodal deepfake video detection API combining Spatial ViT, 2D FFT Frequency analysis, and Bi-LSTM Temporal modeling.",
    version="1.0.0"
)

# Enable CORS for React frontend dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API router modules
app.include_router(health.router, prefix="/api")
app.include_router(analyze.router, prefix="/api")
app.include_router(results.router, prefix="/api")
app.include_router(explain.router, prefix="/api")

# Serve uploaded video files statically for dashboard video playback
upload_dir = os.path.abspath("data/raw")
os.makedirs(upload_dir, exist_ok=True)
app.mount("/static/videos", StaticFiles(directory=upload_dir), name="videos")


@app.get("/")
def root():
    return {
        "message": "Welcome to Synthetic Media Detection & Analysis Platform API",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
