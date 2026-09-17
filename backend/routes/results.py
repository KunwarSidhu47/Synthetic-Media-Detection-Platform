"""
Analysis results endpoints backed by DatabaseManager.
"""

from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException, Path

from backend.schemas.detection import DetectionPipelineResult
from backend.database import DatabaseManager

router = APIRouter(prefix="/results", tags=["Results"])

# In-memory database store for fallback
RESULTS_STORE: Dict[str, DetectionPipelineResult] = {}
db_manager = DatabaseManager()


@router.get("/{analysis_id}", response_model=DetectionPipelineResult)
def get_analysis_result(
    analysis_id: str = Path(..., description="UUID of the analysis session")
):
    """
    Retrieve full multi-signal detection result by analysis_id.
    """
    # 1. Try DB retrieval first
    res = db_manager.get_analysis(analysis_id)
    if res:
        return res

    # 2. Try in-memory store fallback
    if analysis_id in RESULTS_STORE:
        return RESULTS_STORE[analysis_id]

    raise HTTPException(
        status_code=404,
        detail=f"Analysis result not found for ID: {analysis_id}"
    )


@router.get("", response_model=List[Dict[str, Any]])
def list_analysis_results():
    """
    Retrieve list of recent analysis session summaries from database.
    """
    db_results = db_manager.list_analyses()
    if db_results:
        return db_results

    # Fallback to in-memory store summaries
    summaries = []
    for aid, res in RESULTS_STORE.items():
        summaries.append({
            "analysis_id": res.analysis_id,
            "prediction": res.prediction,
            "final_score": res.final_score,
            "confidence": res.confidence,
            "duration_seconds": res.video_metadata.duration_seconds,
            "suspicious_frames_count": len(res.suspicious_frames)
        })
    return summaries
