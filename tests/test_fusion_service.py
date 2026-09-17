"""
Unit tests for MultiSignalFusionService and end-to-end detection pipeline.
"""

import os
import pytest
import numpy as np

from backend.services.fusion_service import MultiSignalFusionService
from backend.schemas.detection import DetectionPipelineResult
from scripts.generate_test_video import create_synthetic_face_video


@pytest.fixture(scope="module")
def fusion_service():
    return MultiSignalFusionService(
        w_spatial=0.45,
        w_frequency=0.25,
        w_temporal=0.30,
        device="cpu",
        pretrained_vit=False
    )


@pytest.fixture(scope="module")
def test_video_path(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("fusion_test")
    path = os.path.join(tmp_dir, "fusion_sample.mp4")
    create_synthetic_face_video(path, duration_sec=2, fps=30)
    return path


def test_fuse_scores(fusion_service):
    # Test synthetic prediction category
    score, label, conf = fusion_service.fuse_scores(0.9, 0.8, 0.85)
    assert 0.8 <= score <= 1.0
    assert label == "Synthetic / Deepfake"
    assert conf >= 0.8

    # Test real prediction category
    score_real, label_real, conf_real = fusion_service.fuse_scores(0.1, 0.2, 0.15)
    assert 0.0 <= score_real <= 0.3
    assert label_real == "Real / Authentic"
    assert conf_real >= 0.7


def test_evidence_summary_generation(fusion_service):
    evidence = fusion_service.generate_evidence_summary(
        spatial_score=0.85,
        frequency_score=0.75,
        temporal_score=0.80,
        suspicious_frames=[10, 20]
    )
    assert "spatial_evidence" in evidence
    assert "frequency_evidence" in evidence
    assert "temporal_evidence" in evidence
    assert "limitations" in evidence
    assert len(evidence["limitations"]) > 0


def test_analyze_video_end_to_end(fusion_service, test_video_path):
    result = fusion_service.analyze_video(test_video_path, sample_rate=10)

    assert isinstance(result, DetectionPipelineResult)
    assert result.analysis_id is not None
    assert result.video_metadata.total_frames == 60
    assert 0.0 <= result.final_score <= 1.0
    assert result.prediction in ["Real / Authentic", "Potentially Synthetic", "Synthetic / Deepfake"]
    assert 0.5 <= result.confidence <= 1.0
    assert len(result.frame_details) == 6
    assert isinstance(result.suspicious_frames, list)
