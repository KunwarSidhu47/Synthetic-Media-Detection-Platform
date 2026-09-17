"""
Unit tests for VideoProcessor service.
"""

import os
import pytest
import numpy as np

from backend.services.video_processor import VideoProcessor
from scripts.generate_test_video import create_synthetic_face_video


@pytest.fixture(scope="module")
def sample_video_path(tmp_path_factory):
    """Fixture creating a temporary 2-second, 30fps synthetic video file."""
    tmp_dir = tmp_path_factory.mktemp("video_data")
    video_path = os.path.join(tmp_dir, "test_sample.mp4")
    create_synthetic_face_video(video_path, duration_sec=2, fps=30)
    return video_path


def test_get_metadata_valid(sample_video_path):
    metadata = VideoProcessor.get_metadata(sample_video_path)
    assert metadata.fps == 30.0
    assert metadata.total_frames == 60
    assert metadata.width == 640
    assert metadata.height == 480
    assert pytest.approx(metadata.duration_seconds, 0.1) == 2.0


def test_get_metadata_nonexistent():
    with pytest.raises(FileNotFoundError):
        VideoProcessor.get_metadata("non_existent_path.mp4")


def test_extract_frames_sample_rate(sample_video_path):
    # Total frames = 60. Sampling every 10 frames -> 6 frames (0, 10, 20, 30, 40, 50)
    frames = list(VideoProcessor.extract_frames(sample_video_path, sample_rate=10))
    assert len(frames) == 6

    for idx, (frame_num, timestamp, img) in enumerate(frames):
        assert frame_num == idx * 10
        assert timestamp == pytest.approx((idx * 10) / 30.0, 0.01)
        assert isinstance(img, np.ndarray)
        assert img.shape == (480, 640, 3)


def test_extract_frames_max_frames(sample_video_path):
    frames = list(VideoProcessor.extract_frames(sample_video_path, sample_rate=5, max_frames=3))
    assert len(frames) == 3
