"""
Integration tests for FastAPI REST API endpoints.
"""

import os
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from scripts.generate_test_video import create_synthetic_face_video

client = TestClient(app)


@pytest.fixture(scope="module")
def api_test_video(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("api_test")
    path = os.path.join(tmp_dir, "api_sample.mp4")
    create_synthetic_face_video(path, duration_sec=2, fps=30)
    return path


def test_root_and_health():
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "docs" in res_root.json()

    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    data = res_health.json()
    assert data["status"] == "healthy"
    assert "torch_device" in data


def test_upload_and_analyze_pipeline(api_test_video):
    # 1. Test POST /api/upload
    with open(api_test_video, "rb") as f:
        res_upload = client.post(
            "/api/upload",
            files={"file": ("test_upload.mp4", f, "video/mp4")}
        )
    assert res_upload.status_code == 200
    upload_data = res_upload.json()
    assert "file_id" in upload_data
    file_id = upload_data["file_id"]

    # 2. Test POST /api/analyze with file_id
    res_analyze = client.post(
        "/api/analyze",
        json={"file_id": file_id, "sample_rate": 10}
    )
    assert res_analyze.status_code == 200
    analysis_data = res_analyze.json()
    assert "analysis_id" in analysis_data
    assert 0.0 <= analysis_data["final_score"] <= 1.0
    assert "spatial_score" in analysis_data
    assert "evidence_summary" in analysis_data
    analysis_id = analysis_data["analysis_id"]

    # 3. Test GET /api/results/{analysis_id}
    res_get = client.get(f"/api/results/{analysis_id}")
    assert res_get.status_code == 200
    assert res_get.json()["analysis_id"] == analysis_id

    # 4. Test GET /api/results list
    res_list = client.get("/api/results")
    assert res_list.status_code == 200
    assert len(res_list.json()) >= 1


def test_invalid_file_extension():
    res = client.post(
        "/api/upload",
        files={"file": ("test.exe", b"invalid_content", "application/octet-stream")}
    )
    assert res.status_code == 400
    assert "Unsupported file format" in res.json()["detail"]


def test_get_nonexistent_result():
    res = client.get("/api/results/non-existent-uuid-123")
    assert res.status_code == 404
