"""
Database Manager for persisting analysis sessions and metadata.
"""

import os
import json
import sqlite3
import time
from typing import List, Dict, Optional, Any

from backend.schemas.detection import DetectionPipelineResult


class DatabaseManager:
    """Manager for SQLite/PostgreSQL-compatible analysis storage."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            data_dir = os.path.abspath("data")
            os.makedirs(data_dir, exist_ok=True)
            db_path = os.path.join(data_dir, "deepfake_platform.db")

        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create analysis_sessions table if it does not exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_sessions (
                    analysis_id TEXT PRIMARY KEY,
                    video_path TEXT,
                    duration_seconds REAL,
                    fps REAL,
                    resolution TEXT,
                    prediction TEXT,
                    confidence REAL,
                    final_score REAL,
                    spatial_score REAL,
                    frequency_score REAL,
                    temporal_score REAL,
                    suspicious_frames_json TEXT,
                    full_result_json TEXT,
                    created_at REAL
                );
            """)
            conn.commit()

    def save_analysis(self, result: DetectionPipelineResult):
        """
        Insert or update a completed DetectionPipelineResult object in database.
        """
        resolution_str = f"{result.video_metadata.width}x{result.video_metadata.height}"
        suspicious_json = json.dumps(result.suspicious_frames)
        full_json = json.dumps(result.model_dump())

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO analysis_sessions (
                    analysis_id, video_path, duration_seconds, fps, resolution,
                    prediction, confidence, final_score, spatial_score,
                    frequency_score, temporal_score, suspicious_frames_json,
                    full_result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                result.analysis_id,
                "data/raw/sample.mp4",
                result.video_metadata.duration_seconds,
                result.video_metadata.fps,
                resolution_str,
                result.prediction,
                result.confidence,
                result.final_score,
                result.spatial_score,
                result.frequency_score,
                result.temporal_score,
                suspicious_json,
                full_json,
                time.time()
            ))
            conn.commit()

    def get_analysis(self, analysis_id: str) -> Optional[DetectionPipelineResult]:
        """
        Retrieve a stored DetectionPipelineResult by analysis_id UUID.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT full_result_json FROM analysis_sessions WHERE analysis_id = ?;",
                (analysis_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            data_dict = json.loads(row["full_result_json"])
            return DetectionPipelineResult(**data_dict)

    def list_analyses(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve list of recent analysis session summaries.
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                SELECT analysis_id, prediction, final_score, confidence,
                       duration_seconds, suspicious_frames_json, created_at
                FROM analysis_sessions
                ORDER BY created_at DESC
                LIMIT ?;
            """, (limit,))
            rows = cursor.fetchall()

            results = []
            for r in rows:
                suspicious = json.loads(r["suspicious_frames_json"]) if r["suspicious_frames_json"] else []
                results.append({
                    "analysis_id": r["analysis_id"],
                    "prediction": r["prediction"],
                    "final_score": r["final_score"],
                    "confidence": r["confidence"],
                    "duration_seconds": r["duration_seconds"],
                    "suspicious_frames_count": len(suspicious),
                    "created_at": r["created_at"]
                })
            return results
