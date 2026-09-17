"""
Unit tests for DatabaseManager session persistence.
"""

import os
import pytest
from backend.database import DatabaseManager
from backend.services.fusion_service import MultiSignalFusionService
from scripts.generate_test_video import create_synthetic_face_video


@pytest.fixture(scope="module")
def temp_db(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("db_test")
    db_file = os.path.join(tmp_dir, "test_platform.db")
    db = DatabaseManager(db_path=db_file)
    return db


@pytest.fixture(scope="module")
def sample_detection_result(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("video_db_test")
    video_path = os.path.join(tmp_dir, "db_sample.mp4")
    create_synthetic_face_video(video_path, duration_sec=2, fps=30)

    fusion = MultiSignalFusionService(device="cpu", pretrained_vit=False)
    result = fusion.analyze_video(video_path, sample_rate=10)
    return result


def test_save_and_get_analysis(temp_db, sample_detection_result):
    # Save result to DB
    temp_db.save_analysis(sample_detection_result)

    # Retrieve from DB
    retrieved = temp_db.get_analysis(sample_detection_result.analysis_id)

    assert retrieved is not None
    assert retrieved.analysis_id == sample_detection_result.analysis_id
    assert retrieved.prediction == sample_detection_result.prediction
    assert retrieved.final_score == sample_detection_result.final_score
    assert retrieved.spatial_score == sample_detection_result.spatial_score
    assert retrieved.frequency_score == sample_detection_result.frequency_score
    assert retrieved.temporal_score == sample_detection_result.temporal_score


def test_list_analyses(temp_db, sample_detection_result):
    summaries = temp_db.list_analyses(limit=10)

    assert len(summaries) >= 1
    first = summaries[0]
    assert first["analysis_id"] == sample_detection_result.analysis_id
    assert "prediction" in first
    assert "duration_seconds" in first


def test_get_nonexistent_analysis(temp_db):
    res = temp_db.get_analysis("non-existent-uuid-999")
    assert res is None
