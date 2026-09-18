"""
Video upload and multi-signal detection analysis endpoints.
"""

import os
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Body
from pydantic import BaseModel, Field

from backend.schemas.detection import DetectionPipelineResult
from backend.services.fusion_service import MultiSignalFusionService
from backend.routes.results import RESULTS_STORE

router = APIRouter(tags=["Analysis"])

# Initialize singleton MultiSignalFusionService instance
fusion_service = MultiSignalFusionService(
    w_spatial=0.50,
    w_frequency=0.30,
    w_temporal=0.20
)

UPLOAD_DIR = os.path.abspath("data/raw")
os.makedirs(UPLOAD_DIR, exist_ok=True)


class AnalyzeRequest(BaseModel):
    video_path: Optional[str] = Field(None, description="Path to video file on server")
    file_id: Optional[str] = Field(None, description="File ID returned from /api/upload")
    sample_rate: int = Field(10, description="Sample 1 frame every N frames (min 1)")


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for deepfake detection analysis.
    Supported formats: .mp4, .avi, .mov, .webm, .mkv
    """
    valid_extensions = [".mp4", ".avi", ".mov", ".webm", ".mkv"]
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Supported formats: {', '.join(valid_extensions)}"
        )

    file_id = str(uuid.uuid4())
    save_filename = f"{file_id}_{file.filename}"
    target_path = os.path.join(UPLOAD_DIR, save_filename)

    try:
        content = await file.read()
        with open(target_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    return {
        "file_id": file_id,
        "filename": file.filename,
        "video_path": os.path.abspath(target_path),
        "size_bytes": len(content)
    }


@router.post("/analyze", response_model=DetectionPipelineResult)
def analyze_video(request: AnalyzeRequest = Body(...)):
    """
    Trigger full multi-signal spatial, frequency, and temporal deepfake detection analysis.
    """
    target_path = request.video_path

    if not target_path and request.file_id:
        # Search for file by file_id in UPLOAD_DIR
        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith(request.file_id):
                target_path = os.path.join(UPLOAD_DIR, filename)
                break

    if not target_path or not os.path.exists(target_path):
        raise HTTPException(
            status_code=404,
            detail="Video file not found. Provide a valid 'video_path' or uploaded 'file_id'."
        )

    if request.sample_rate < 1:
        raise HTTPException(status_code=400, detail="sample_rate must be >= 1")

    try:
        # Cap max_frames=12 for fast processing on cloud free tier (0.1 CPU)
        result = fusion_service.analyze_video(target_path, sample_rate=request.sample_rate, max_frames=12)
        # Store in in-memory repository and database
        RESULTS_STORE[result.analysis_id] = result
        try:
            from backend.routes.results import db_manager
            db_manager.save_analysis(result)
        except Exception as db_err:
            print(f"Warning: Failed to persist analysis to database: {db_err}")

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during video detection analysis: {str(e)}"
        )
