"""
Unit tests for LLM Explanation Layer service and API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.llm_service import LLMExplanationService
from backend.schemas.llm import LLMExplanationResponse
from backend.services.fusion_service import MultiSignalFusionService
from backend.routes.results import RESULTS_STORE
from scripts.generate_test_video import create_synthetic_face_video

client = TestClient(app)


@pytest.fixture(scope="module")
def sample_analysis_result(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("llm_test")
    path = f"{tmp_dir}/llm_sample.mp4"
    create_synthetic_face_video(path, duration_sec=2, fps=30)

    fusion = MultiSignalFusionService(device="cpu", pretrained_vit=False)
    res = fusion.analyze_video(path, sample_rate=10)
    RESULTS_STORE[res.analysis_id] = res
    return res


def test_generate_explanation_schema(sample_analysis_result):
    service = LLMExplanationService()
    explanation = service.generate_explanation(sample_analysis_result)

    assert isinstance(explanation, LLMExplanationResponse)
    assert explanation.analysis_id == sample_analysis_result.analysis_id
    assert len(explanation.executive_summary) > 0
    assert "spatial_vit" in explanation.signal_breakdown
    assert "RESPONSIBLE AI NOTICE" in explanation.responsible_ai_disclaimer


def test_generate_markdown_report(sample_analysis_result):
    service = LLMExplanationService()
    md = service.generate_markdown_report(sample_analysis_result)

    assert "# Synthetic Media Detection Forensic Report" in md
    assert "Executive Summary" in md
    assert "Vision Transformer" in md


def test_api_explain_endpoint(sample_analysis_result):
    res = client.post(
        "/api/explain",
        json={"analysis_id": sample_analysis_result.analysis_id}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["analysis_id"] == sample_analysis_result.analysis_id
    assert "executive_summary" in data


def test_api_report_endpoint(sample_analysis_result):
    res = client.post(
        "/api/report",
        json={"analysis_id": sample_analysis_result.analysis_id}
    )
    assert res.status_code == 200
    data = res.json()
    assert "report_markdown" in data
    assert len(data["report_markdown"]) > 0
