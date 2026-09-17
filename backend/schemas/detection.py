"""
Data schemas for detection outputs, spatial/frequency/temporal signal scores, and model evaluation metrics.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from backend.schemas.video import VideoMetadata


class SpatialCropAnalysis(BaseModel):
    """Result of spatial analysis performed on a single facial crop image using Vision Transformer."""
    spatial_score: float = Field(..., description="Synthetic probability score from 0.0 (Real) to 1.0 (Synthetic)")
    label: str = Field(..., description="Classification label ('REAL' or 'SYNTHETIC')")
    confidence: float = Field(..., description="Model classification confidence score (0.5 to 1.0)")
    feature_embedding: Optional[List[float]] = Field(None, description="Spatial feature vector (e.g. 768-dim) for temporal analysis")


class FrequencyAnalysisResult(BaseModel):
    """Result of 2D FFT frequency-domain analysis on a face crop image."""
    frequency_score: float = Field(..., description="Frequency anomaly score from 0.0 (Natural) to 1.0 (Anomalous/Synthetic)")
    high_freq_power_ratio: float = Field(..., description="Ratio of spectral energy in high frequency outer bands relative to total power")
    spectral_centroid: float = Field(..., description="Center of mass of 2D FFT magnitude spectrum")
    azimuthal_profile: List[float] = Field(..., description="1D radial average of magnitude spectrum across concentric circles")
    details: Optional[Dict[str, float]] = Field(None, description="Detailed frequency metric metrics")


class TemporalAnalysisResult(BaseModel):
    """Result of Bi-LSTM temporal sequence analysis across consecutive video frames."""
    temporal_score: float = Field(..., description="Temporal anomaly score from 0.0 (Consistent/Real) to 1.0 (Inconsistent/Synthetic)")
    sequence_length: int = Field(..., description="Total number of frame feature vectors in sequence")
    label: str = Field(..., description="Temporal classification label ('REAL' or 'SYNTHETIC')")
    confidence: float = Field(..., description="Model temporal confidence score (0.5 to 1.0)")
    suspicious_frame_indices: List[int] = Field(..., description="List of frame indices flagged for highest temporal feature anomalies")


class FrameAnalysisResult(BaseModel):
    """Combined multi-signal detection result for a single video frame."""
    frame_number: int = Field(..., description="Sequential index of the frame")
    timestamp_seconds: float = Field(..., description="Exact timestamp of frame in seconds")
    face_detected: bool = Field(..., description="Whether a face region was detected")
    spatial_score: Optional[float] = Field(None, description="Spatial ViT synthetic score")
    frequency_score: Optional[float] = Field(None, description="Frequency FFT synthetic score")
    temporal_score: Optional[float] = Field(None, description="Temporal LSTM score")
    combined_score: Optional[float] = Field(None, description="Fused synthetic probability score")


class DetectionPipelineResult(BaseModel):
    """Master result model returned by end-to-end multi-signal deepfake detection pipeline."""
    analysis_id: str = Field(..., description="Unique UUID assigned to this analysis session")
    video_metadata: VideoMetadata = Field(..., description="Extracted video metadata")
    video_url: Optional[str] = Field(None, description="Static playback URL for the video")
    final_score: float = Field(..., description="Fused synthetic probability score (0.0 to 1.0)")
    prediction: str = Field(..., description="Prediction category ('Real / Authentic', 'Potentially Synthetic', 'Synthetic / Deepfake')")
    confidence: float = Field(..., description="Overall confidence score (0.5 to 1.0)")
    spatial_score: float = Field(..., description="Average spatial ViT synthetic score")
    frequency_score: float = Field(..., description="Average frequency FFT anomaly score")
    temporal_score: float = Field(..., description="Sequence temporal Bi-LSTM score")
    signal_weights: Dict[str, float] = Field(..., description="Weights applied to spatial, frequency, and temporal scores")
    suspicious_frames: List[int] = Field(..., description="List of frame indices flagged for temporal/spatial anomalies")
    evidence_summary: Dict[str, List[str]] = Field(..., description="Structured evidence observations for spatial, frequency, and temporal signals")
    frame_details: List[FrameAnalysisResult] = Field(..., description="Frame-by-frame analysis breakdowns")


class EvaluationMetrics(BaseModel):
    """Model evaluation metrics summary."""
    accuracy: float = Field(..., description="Overall classification accuracy")
    precision: float = Field(..., description="Precision score for synthetic class")
    recall: float = Field(..., description="Recall score for synthetic class")
    f1_score: float = Field(..., description="F1-score for synthetic class")
    roc_auc: Optional[float] = Field(None, description="Receiver Operating Characteristic Area Under Curve")
    confusion_matrix: Dict[str, int] = Field(..., description="Dict containing tp, fp, tn, fn counts")
