"""
Pre-baked demo endpoint that returns realistic, pre-computed results
for the three built-in sample videos WITHOUT running the full ML pipeline.

This avoids 502 OOM crashes on Render's free-tier 512MB RAM limit.
"""

import uuid
from fastapi import APIRouter, HTTPException
from backend.schemas.detection import (
    DetectionPipelineResult,
    FrameAnalysisResult,
    SpatialCropAnalysis,
    FrequencyAnalysisResult,
    TemporalAnalysisResult,
)
from backend.schemas.video import VideoMetadata
from backend.routes.results import RESULTS_STORE

router = APIRouter(tags=["Demo"])


def _make_frame_details(count: int, spatial_base: float, freq_base: float) -> list:
    """Generate a realistic sequence of per-frame analysis results."""
    import random
    random.seed(42)
    frames = []
    for i in range(count):
        ts = round(i * 0.33, 2)
        s = round(min(1.0, max(0.0, spatial_base + random.uniform(-0.08, 0.08))), 4)
        f = round(min(1.0, max(0.0, freq_base + random.uniform(-0.06, 0.06))), 4)
        combined = round(0.50 * s + 0.30 * f + 0.20 * 0.5, 4)
        frames.append(FrameAnalysisResult(
            frame_number=i * 10,
            timestamp_seconds=ts,
            face_detected=True,
            spatial_score=s,
            frequency_score=f,
            temporal_score=None,
            combined_score=combined,
        ))
    return frames


# ── pre-baked result factories ─────────────────────────────────────────────────

def _real_human_face_result(analysis_id: str) -> DetectionPipelineResult:
    frame_details = _make_frame_details(12, spatial_base=0.14, freq_base=0.18)
    return DetectionPipelineResult(
        analysis_id=analysis_id,
        video_metadata=VideoMetadata(
            fps=30.0,
            total_frames=120,
            duration_seconds=4.0,
            width=1280,
            height=720,
            codec="h264",
        ),
        video_url="/static/videos/real_human_face.mp4",
        final_score=0.1612,
        prediction="Real / Authentic",
        confidence=0.8388,
        spatial_score=0.1401,
        frequency_score=0.1823,
        temporal_score=0.1310,
        signal_weights={"spatial": 0.5, "frequency": 0.3, "temporal": 0.2},
        suspicious_frames=[],
        evidence_summary={
            "spatial_evidence": [
                "Facial texture and boundary features appear spatially authentic."
            ],
            "frequency_evidence": [
                "Frequency domain energy distribution aligns with natural camera sensor characteristics."
            ],
            "temporal_evidence": [
                "Inter-frame feature transitions demonstrate smooth temporal continuity."
            ],
            "limitations": [
                "Model predictions represent probabilistic estimates and should be paired with human review.",
                "Compression artifacts (e.g. heavy H.264 re-encoding) can influence high-frequency spectral metrics.",
                "Occlusions, rapid motion blur, or extreme lighting angles may reduce face detection coverage.",
            ],
        },
        frame_details=frame_details,
    )


def _authentic_sample_result(analysis_id: str) -> DetectionPipelineResult:
    frame_details = _make_frame_details(8, spatial_base=0.19, freq_base=0.21)
    return DetectionPipelineResult(
        analysis_id=analysis_id,
        video_metadata=VideoMetadata(
            fps=30.0,
            total_frames=82,
            duration_seconds=2.7,
            width=640,
            height=480,
            codec="h264",
        ),
        video_url="/static/videos/authentic_sample.mp4",
        final_score=0.2041,
        prediction="Real / Authentic",
        confidence=0.7959,
        spatial_score=0.1910,
        frequency_score=0.2208,
        temporal_score=0.1990,
        signal_weights={"spatial": 0.5, "frequency": 0.3, "temporal": 0.2},
        suspicious_frames=[],
        evidence_summary={
            "spatial_evidence": [
                "Facial texture and boundary features appear spatially authentic."
            ],
            "frequency_evidence": [
                "Frequency domain energy distribution aligns with natural camera sensor characteristics."
            ],
            "temporal_evidence": [
                "Inter-frame feature transitions demonstrate smooth temporal continuity."
            ],
            "limitations": [
                "Model predictions represent probabilistic estimates and should be paired with human review.",
                "Compression artifacts (e.g. heavy H.264 re-encoding) can influence high-frequency spectral metrics.",
                "Occlusions, rapid motion blur, or extreme lighting angles may reduce face detection coverage.",
            ],
        },
        frame_details=frame_details,
    )


def _synthetic_sample_result(analysis_id: str) -> DetectionPipelineResult:
    frame_details = _make_frame_details(10, spatial_base=0.74, freq_base=0.81)
    return DetectionPipelineResult(
        analysis_id=analysis_id,
        video_metadata=VideoMetadata(
            fps=30.0,
            total_frames=100,
            duration_seconds=3.3,
            width=640,
            height=480,
            codec="mp4v",
        ),
        video_url="/static/videos/synthetic_sample.mp4",
        final_score=0.7681,
        prediction="Synthetic / Deepfake",
        confidence=0.7681,
        spatial_score=0.7412,
        frequency_score=0.8134,
        temporal_score=0.7220,
        signal_weights={"spatial": 0.5, "frequency": 0.3, "temporal": 0.2},
        suspicious_frames=[10, 30, 60, 80],
        evidence_summary={
            "spatial_evidence": [
                "Vision Transformer detected strong visual blending artifacts and facial unnaturalness."
            ],
            "frequency_evidence": [
                "Elevated 2D FFT spectral power detected in high-frequency bands (characteristic of GAN/diffusion generation)."
            ],
            "temporal_evidence": [
                "Bi-LSTM sequence model identified significant inter-frame feature flicker across frames.",
                "Frames requiring detailed visual review: 10, 30, 60, 80",
            ],
            "limitations": [
                "Model predictions represent probabilistic estimates and should be paired with human review.",
                "Compression artifacts (e.g. heavy H.264 re-encoding) can influence high-frequency spectral metrics.",
                "Occlusions, rapid motion blur, or extreme lighting angles may reduce face detection coverage.",
            ],
        },
        frame_details=frame_details,
    )


_DEMO_FACTORIES = {
    "real_human_face": _real_human_face_result,
    "authentic_sample": _authentic_sample_result,
    "synthetic_sample": _synthetic_sample_result,
}


@router.get("/demo/{sample_name}", response_model=DetectionPipelineResult)
def get_demo_result(sample_name: str):
    """
    Return a pre-computed detection result for a named demo sample.

    Valid sample_name values:
    - real_human_face
    - authentic_sample
    - synthetic_sample
    """
    factory = _DEMO_FACTORIES.get(sample_name)
    if factory is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown demo sample '{sample_name}'. "
                   f"Valid options: {list(_DEMO_FACTORIES.keys())}",
        )

    analysis_id = str(uuid.uuid4())
    result = factory(analysis_id)

    # Store so /api/explain and /api/report still work
    RESULTS_STORE[analysis_id] = result
    return result
