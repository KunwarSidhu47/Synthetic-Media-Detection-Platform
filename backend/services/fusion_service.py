"""
Multi-Signal Fusion Service orchestrating Spatial (ViT), Frequency (FFT), and Temporal (Bi-LSTM) analysis.
"""

import os
import uuid
from typing import List, Tuple, Dict, Optional, Any
import numpy as np

from backend.schemas.video import VideoMetadata
from backend.schemas.detection import (
    FrameAnalysisResult,
    DetectionPipelineResult,
    SpatialCropAnalysis,
    FrequencyAnalysisResult,
    TemporalAnalysisResult
)
from backend.services.video_processor import VideoProcessor
from backend.services.face_detector import FaceDetector
from backend.services.vit_service import ViTSpatialService
from backend.services.frequency_service import FrequencyAnalysisService
from backend.services.temporal_service import LSTMTemporalService


class MultiSignalFusionService:
    """Service to coordinate full multi-modal detection pipeline and score fusion."""

    def __init__(
        self,
        w_spatial: float = 0.50,
        w_frequency: float = 0.30,
        w_temporal: float = 0.20,
        device: Optional[str] = None,
        pretrained_vit: bool = False
    ):
        """
        Initialize MultiSignalFusionService.
        
        Args:
            w_spatial: Weight for Vision Transformer spatial score.
            w_frequency: Weight for 2D FFT frequency score.
            w_temporal: Weight for Bi-LSTM temporal sequence score.
            device: PyTorch device ('cpu', 'cuda', 'mps').
            pretrained_vit: Whether to load pre-trained ViT weights.
        """
        # Normalize weights to sum to 1.0
        total_w = w_spatial + w_frequency + w_temporal
        if total_w <= 0:
            raise ValueError("Sum of signal weights must be > 0")

        self.w_spatial = w_spatial / total_w
        self.w_frequency = w_frequency / total_w
        self.w_temporal = w_temporal / total_w

        # Initialize sub-services
        self.face_detector = FaceDetector(min_detection_confidence=0.4)
        self.vit_service = ViTSpatialService(device=device, pretrained=pretrained_vit)
        self.freq_service = FrequencyAnalysisService()
        self.temporal_service = LSTMTemporalService(device=device)

    def fuse_scores(
        self,
        spatial_score: float,
        frequency_score: float,
        temporal_score: float
    ) -> Tuple[float, str, float]:
        """
        Compute weighted fused score, category prediction, and confidence score.
        
        Returns:
            Tuple of (fused_score: float, prediction_label: str, confidence: float)
        """
        fused = (
            self.w_spatial * spatial_score +
            self.w_frequency * frequency_score +
            self.w_temporal * temporal_score
        )
        fused = float(np.clip(fused, 0.0, 1.0))

        if fused >= 0.55:
            label = "Synthetic / Deepfake"
        elif fused >= 0.25:
            label = "Potentially Synthetic"
        else:
            label = "Real / Authentic"

        # Map score distance from decision boundary (0.5) to confidence [0.5, 1.0]
        confidence = float(0.5 + abs(fused - 0.5))
        return round(fused, 4), label, round(confidence, 4)

    def generate_evidence_summary(
        self,
        spatial_score: float,
        frequency_score: float,
        temporal_score: float,
        suspicious_frames: List[int]
    ) -> Dict[str, List[str]]:
        """
        Generate structured evidence observations for spatial, frequency, and temporal models.
        """
        spatial_obs = []
        if spatial_score >= 0.7:
            spatial_obs.append("Vision Transformer detected strong visual blending artifacts and facial unnaturalness.")
        elif spatial_score >= 0.4:
            spatial_obs.append("Moderate spatial anomalies observed in facial landmark boundaries.")
        else:
            spatial_obs.append("Facial texture and boundary features appear spatially authentic.")

        freq_obs = []
        if frequency_score >= 0.7:
            freq_obs.append("Elevated 2D FFT spectral power detected in high-frequency bands (characteristic of GAN/diffusion generation).")
        elif frequency_score >= 0.4:
            freq_obs.append("Slight spectral roll-off distortion observed in frequency domain.")
        else:
            freq_obs.append("Frequency domain energy distribution aligns with natural camera sensor characteristics.")

        temp_obs = []
        if temporal_score >= 0.7:
            temp_obs.append(f"Bi-LSTM sequence model identified significant inter-frame feature flicker across frames.")
        elif temporal_score >= 0.4:
            temp_obs.append("Minor temporal feature variations observed across sequence frames.")
        else:
            temp_obs.append("Inter-frame feature transitions demonstrate smooth temporal continuity.")

        if suspicious_frames:
            temp_obs.append(f"Frames requiring detailed visual review: {', '.join(map(str, suspicious_frames))}")

        limitations = [
            "Model predictions represent probabilistic estimates and should be paired with human review.",
            "Compression artifacts (e.g. heavy H.264 re-encoding) can influence high-frequency spectral metrics.",
            "Occlusions, rapid motion blur, or extreme lighting angles may reduce face detection coverage."
        ]

        return {
            "spatial_evidence": spatial_obs,
            "frequency_evidence": freq_obs,
            "temporal_evidence": temp_obs,
            "limitations": limitations
        }

    def analyze_video(
        self,
        video_path: str,
        sample_rate: int = 10,
        max_frames: Optional[int] = 30
    ) -> DetectionPipelineResult:
        """
        Execute end-to-end multi-signal detection pipeline on an input video file.
        
        Args:
            video_path: Path to MP4/video file.
            sample_rate: Frame sampling interval (sample 1 frame every N frames).
            max_frames: Upper limit on sampled frames for performance safety.
            
        Returns:
            DetectionPipelineResult master schema object.
        """
        analysis_id = str(uuid.uuid4())
        metadata = VideoProcessor.get_metadata(video_path)

        sampled_frames = list(VideoProcessor.extract_frames(
            video_path,
            sample_rate=sample_rate,
            max_frames=max_frames
        ))

        if not sampled_frames:
            raise ValueError(f"No frames could be extracted from video at: {video_path}")

        spatial_scores: List[float] = []
        frequency_scores: List[float] = []
        feature_embeddings: List[List[float]] = []
        frame_details: List[FrameAnalysisResult] = []

        for frame_num, timestamp, frame_bgr in sampled_frames:
            # Detect face and crop
            frame_meta, crop_rgb = self.face_detector.process_frame(frame_bgr, frame_num, timestamp)

            # Spatial Analysis (ViT)
            spatial_res = self.vit_service.predict_crop(crop_rgb)
            spatial_scores.append(spatial_res.spatial_score)
            feature_embeddings.append(spatial_res.feature_embedding)

            # Frequency Analysis (2D FFT)
            freq_res = self.freq_service.predict_crop(crop_rgb)
            frequency_scores.append(freq_res.frequency_score)

            # Frame Combined Score
            frame_fused, _, _ = self.fuse_scores(spatial_res.spatial_score, freq_res.frequency_score, 0.5)

            frame_details.append(FrameAnalysisResult(
                frame_number=frame_num,
                timestamp_seconds=timestamp,
                face_detected=frame_meta.face_detected,
                spatial_score=spatial_res.spatial_score,
                frequency_score=freq_res.frequency_score,
                temporal_score=None,
                combined_score=frame_fused
            ))

        # Temporal Analysis (Bi-LSTM over sequence of embeddings)
        temporal_res = self.temporal_service.predict_sequence(feature_embeddings)
        temporal_score = temporal_res.temporal_score

        # Map internal frame sequence indices back to actual video frame numbers
        suspicious_frame_numbers = []
        for idx in temporal_res.suspicious_frame_indices:
            if 0 <= idx < len(sampled_frames):
                suspicious_frame_numbers.append(sampled_frames[idx][0])

        # Overall Aggregated Signals
        avg_spatial = float(np.mean(spatial_scores))
        avg_frequency = float(np.mean(frequency_scores))

        # Master Score Fusion
        final_score, prediction, confidence = self.fuse_scores(
            avg_spatial,
            avg_frequency,
            temporal_score
        )

        evidence = self.generate_evidence_summary(
            avg_spatial,
            avg_frequency,
            temporal_score,
            suspicious_frame_numbers
        )

        video_filename = os.path.basename(video_path)
        video_url = f"/static/videos/{video_filename}"

        return DetectionPipelineResult(
            analysis_id=analysis_id,
            video_metadata=metadata,
            video_url=video_url,
            final_score=final_score,
            prediction=prediction,
            confidence=confidence,
            spatial_score=round(avg_spatial, 4),
            frequency_score=round(avg_frequency, 4),
            temporal_score=round(temporal_score, 4),
            signal_weights={
                "spatial": round(self.w_spatial, 4),
                "frequency": round(self.w_frequency, 4),
                "temporal": round(self.w_temporal, 4)
            },
            suspicious_frames=suspicious_frame_numbers,
            evidence_summary=evidence,
            frame_details=frame_details
        )
