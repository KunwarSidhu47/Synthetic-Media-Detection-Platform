"""
Unit tests for ViTSpatialService and evaluation metrics utilities.
"""

import pytest
import numpy as np

from backend.services.vit_service import ViTSpatialService
from backend.utils.metrics import compute_evaluation_metrics


@pytest.fixture(scope="module")
def vit_service():
    # Use CPU and pretrained=False for instant unit test execution without network weights download
    service = ViTSpatialService(device="cpu", pretrained=False)
    return service


@pytest.fixture
def synthetic_crop():
    """Create a dummy 224x224 RGB image array."""
    np.random.seed(42)
    crop = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return crop


def test_predict_crop(vit_service, synthetic_crop):
    result = vit_service.predict_crop(synthetic_crop)

    assert 0.0 <= result.spatial_score <= 1.0
    assert result.label in ["REAL", "SYNTHETIC"]
    assert 0.5 <= result.confidence <= 1.0
    assert result.feature_embedding is not None
    assert len(result.feature_embedding) == 768


def test_predict_batch(vit_service, synthetic_crop):
    crops = [synthetic_crop, synthetic_crop.copy(), synthetic_crop.copy()]
    results = vit_service.predict_batch(crops)

    assert len(results) == 3
    for res in results:
        assert 0.0 <= res.spatial_score <= 1.0
        assert len(res.feature_embedding) == 768


def test_compute_evaluation_metrics():
    y_true = [0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 0]
    y_prob = [0.1, 0.2, 0.9, 0.85, 0.4]

    metrics = compute_evaluation_metrics(y_true, y_pred, y_prob)

    assert metrics.accuracy == 0.8
    assert metrics.precision == 1.0
    assert metrics.recall == pytest.approx(0.666, 0.01)
    assert metrics.f1_score == pytest.approx(0.8, 0.01)
    assert metrics.confusion_matrix == {"tn": 2, "fp": 0, "fn": 1, "tp": 2}
