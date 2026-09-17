"""
Video Processor Service using OpenCV for frame sampling, metadata extraction, and pre-processing.
"""

import os
from typing import Generator, Tuple, Optional
import cv2
import numpy as np

from backend.schemas.video import VideoMetadata


class VideoProcessor:
    """Service to handle OpenCV video loading, frame sampling, and metadata retrieval."""

    @staticmethod
    def get_metadata(video_path: str) -> VideoMetadata:
        """
        Extract video metadata including FPS, total frame count, duration, and resolution.
        
        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If OpenCV fails to open or read the video file.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at path: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video file: {video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Avoid division by zero if FPS or frame count is missing/invalid
            if fps <= 0:
                fps = 30.0
            duration = total_frames / fps if total_frames > 0 else 0.0

            # Decode codec integer
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

            return VideoMetadata(
                fps=fps,
                total_frames=total_frames,
                duration_seconds=duration,
                width=width,
                height=height,
                codec=codec or "unknown"
            )
        finally:
            cap.release()

    @staticmethod
    def extract_frames(
        video_path: str,
        sample_rate: int = 10,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, float, np.ndarray], None, None]:
        """
        Stream frames from video file at specified sampling interval.
        
        Args:
            video_path: Path to input video file.
            sample_rate: Sample 1 frame every N frames. Must be >= 1.
            max_frames: Optional upper limit on total extracted frames.
            
        Yields:
            Tuple of (frame_number: int, timestamp_seconds: float, frame_bgr: np.ndarray)
        """
        if sample_rate < 1:
            raise ValueError(f"Sample rate must be >= 1, got {sample_rate}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at path: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frame_idx = 0
        extracted_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_rate == 0:
                    timestamp = frame_idx / fps
                    yield frame_idx, timestamp, frame
                    extracted_count += 1

                    if max_frames is not None and extracted_count >= max_frames:
                        break

                frame_idx += 1
        finally:
            cap.release()
