"""
Pydantic data models for video metadata, frame extraction, and face detection results.
"""

from typing import Optional, List, Tuple
from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Metadata describing input video properties."""
    fps: float = Field(..., description="Frames per second of the video")
    total_frames: int = Field(..., description="Total number of frames in the video")
    duration_seconds: float = Field(..., description="Total video duration in seconds")
    width: int = Field(..., description="Frame width in pixels")
    height: int = Field(..., description="Frame height in pixels")
    codec: Optional[str] = Field(None, description="Video codec format indicator")


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected face region (normalized 0.0 to 1.0)."""
    ymin: float = Field(..., description="Normalized top coordinate (0.0 to 1.0)")
    xmin: float = Field(..., description="Normalized left coordinate (0.0 to 1.0)")
    width: float = Field(..., description="Normalized box width (0.0 to 1.0)")
    height: float = Field(..., description="Normalized box height (0.0 to 1.0)")
    confidence: float = Field(1.0, description="Face detection confidence score")

    def to_pixel_coords(self, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """
        Convert normalized bounding box to pixel coordinates (top, left, bottom, right).
        Clamped within image boundaries.
        """
        top = int(max(0, self.ymin * image_height))
        left = int(max(0, self.xmin * image_width))
        bottom = int(min(image_height, (self.ymin + self.height) * image_height))
        right = int(min(image_width, (self.xmin + self.width) * image_width))
        return top, left, bottom, right


class FrameMetadata(BaseModel):
    """Metadata associated with an extracted video frame."""
    frame_number: int = Field(..., description="Sequential index of the frame")
    timestamp_seconds: float = Field(..., description="Exact timestamp of the frame in seconds")
    face_detected: bool = Field(..., description="Whether a face was detected in this frame")
    bounding_box: Optional[BoundingBox] = Field(None, description="Bounding box if face was detected")
