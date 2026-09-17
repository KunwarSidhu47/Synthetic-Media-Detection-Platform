"""
Unit tests for LSTMTemporalService and Bi-LSTM model.
"""

import pytest
import numpy as np
import torch

from models.lstm_temporal import LSTMTemporalModel
from backend.services.temporal_service import LSTMTemporalService


@pytest.fixture(scope="module")
def temporal_service():
    return LSTMTemporalService(device="cpu")


@pytest.fixture
def smooth_feature_sequence():
    """Generate a smooth temporal sequence of 10 768-dim vectors."""
    np.random.seed(42)
    base_vec = np.random.randn(768).astype(np.float32)
    seq = []
    for i in range(10):
        # Small smooth continuous drift
        noise = np.random.randn(768) * 0.05
        seq.append((base_vec + noise).tolist())
    return seq


@pytest.fixture
def flickering_feature_sequence(smooth_feature_sequence):
    """Introduce an abrupt temporal jump at frame index 5."""
    seq = [list(vec) for vec in smooth_feature_sequence]
    # Add large spike jump at index 5
    seq[5] = (np.array(seq[5]) + 5.0).tolist()
    return seq


def test_lstm_model_forward():
    model = LSTMTemporalModel(input_dim=768, hidden_dim=256)
    x = torch.randn(2, 8, 768)  # Batch=2, SeqLen=8, Dim=768
    logits, seq_out = model(x)

    assert logits.shape == (2, 2)
    assert seq_out.shape == (2, 8, 512)


def test_predict_sequence_smooth(temporal_service, smooth_feature_sequence):
    result = temporal_service.predict_sequence(smooth_feature_sequence)

    assert 0.0 <= result.temporal_score <= 1.0
    assert result.sequence_length == 10
    assert result.label in ["REAL", "SYNTHETIC"]
    assert 0.5 <= result.confidence <= 1.0
    assert isinstance(result.suspicious_frame_indices, list)


def test_predict_sequence_flicker_detection(temporal_service, flickering_feature_sequence):
    result = temporal_service.predict_sequence(flickering_feature_sequence)

    assert 5 in result.suspicious_frame_indices


def test_single_frame_sequence(temporal_service):
    single_frame_seq = [np.random.randn(768).tolist()]
    result = temporal_service.predict_sequence(single_frame_seq)

    assert result.sequence_length == 1
    assert 0.0 <= result.temporal_score <= 1.0
