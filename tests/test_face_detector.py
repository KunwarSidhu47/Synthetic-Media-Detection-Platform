"""
Unit tests for FaceDetector service.
"""

import pytest
import numpy as np
import cv2

from backend.services.face_detector import FaceDetector
from backend.schemas.video import BoundingBox


@pytest.fixture(scope="module")
def face_detector():
    detector = FaceDetector(min_detection_confidence=0.3)
    yield detector
    detector.close()


@pytest.fixture
def synthetic_face_image():
    """Create a BGR synthetic image with face features."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :] = (30, 30, 50)
    center_x, center_y = 320, 240
    cv2.ellipse(img, (center_x, center_y), (100, 130), 0, 0, 360, (180, 200, 230), -1)
    cv2.circle(img, (center_x - 40, center_y - 30), 15, (255, 255, 255), -1)
    cv2.circle(img, (center_x + 40, center_y - 30), 15, (255, 255, 255), -1)
    cv2.circle(img, (center_x - 40, center_y - 30), 6, (120, 50, 20), -1)
    cv2.circle(img, (center_x + 40, center_y - 30), 6, (120, 50, 20), -1)
    cv2.ellipse(img, (center_x, center_y + 40), (30, 15), 0, 0, 180, (50, 50, 180), 3)
    return img


def test_crop_face_custom_bbox(face_detector, synthetic_face_image):
    bbox = BoundingBox(ymin=0.2, xmin=0.3, width=0.4, height=0.5, confidence=0.9)
    crop_rgb = face_detector.crop_face(synthetic_face_image, bbox, target_size=(224, 224))
    
    assert isinstance(crop_rgb, np.ndarray)
    assert crop_rgb.shape == (224, 224, 3)


def test_crop_face_none_bbox(face_detector, synthetic_face_image):
    crop_rgb = face_detector.crop_face(synthetic_face_image, None, target_size=(224, 224))
    assert isinstance(crop_rgb, np.ndarray)
    assert crop_rgb.shape == (224, 224, 3)


def test_process_frame(face_detector, synthetic_face_image):
    metadata, crop_rgb = face_detector.process_frame(
        frame_bgr=synthetic_face_image,
        frame_number=12,
        timestamp=0.4
    )
    
    assert metadata.frame_number == 12
    assert metadata.timestamp_seconds == 0.4
    assert crop_rgb.shape == (224, 224, 3)
