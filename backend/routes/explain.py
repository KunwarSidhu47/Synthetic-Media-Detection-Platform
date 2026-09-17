"""
LLM Explanation Layer API endpoints.
"""

from fastapi import APIRouter, HTTPException, Body

from backend.schemas.llm import LLMExplanationRequest, LLMExplanationResponse
from backend.services.llm_service import LLMExplanationService
from backend.routes.results import RESULTS_STORE

router = APIRouter(tags=["LLM Explanation"])

llm_service = LLMExplanationService()


@router.post("/explain", response_model=LLMExplanationResponse)
def generate_explanation(request: LLMExplanationRequest = Body(...)):
    """
    Generate a human-readable forensic explanation for a completed analysis session.
    """
    if request.analysis_id not in RESULTS_STORE:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis result not found for ID: {request.analysis_id}"
        )

    result = RESULTS_STORE[request.analysis_id]
    explanation = llm_service.generate_explanation(result)
    return explanation


@router.post("/report")
def generate_report(request: LLMExplanationRequest = Body(...)):
    """
    Generate a structured Markdown forensic report for a completed analysis session.
    """
    if request.analysis_id not in RESULTS_STORE:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis result not found for ID: {request.analysis_id}"
        )

    result = RESULTS_STORE[request.analysis_id]
    md_report = llm_service.generate_markdown_report(result)
    return {
        "analysis_id": request.analysis_id,
        "report_markdown": md_report
    }
