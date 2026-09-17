"""
Script to generate a synthetic test video with drawn facial features for unit testing.
"""

import os
import cv2
import numpy as np


def create_synthetic_face_video(output_path: str, duration_sec: int = 2, fps: int = 30) -> str:
    """
    Generate a simple synthetic video with an animated face-like pattern.
    
    Args:
        output_path: Path where output MP4 file will be saved.
        duration_sec: Video length in seconds.
        fps: Frames per second.
        
    Returns:
        Absolute path to generated video file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    width, height = 640, 480
    total_frames = duration_sec * fps
    
    # Use MJPG fourcc codec for reliable cross-platform headless video creation
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        # Fallback to mp4v codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        for i in range(total_frames):
            # Gradient background
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (40, 40, 60)
            
            # Oscillating center position to simulate head movement
            center_x = int(width // 2 + 30 * np.sin(2 * np.pi * i / fps))
            center_y = int(height // 2 + 15 * np.cos(2 * np.pi * i / fps))
            
            # Draw head oval (skin tone)
            cv2.ellipse(frame, (center_x, center_y), (100, 130), 0, 0, 360, (180, 200, 230), -1)
            
            # Draw eyes
            cv2.circle(frame, (center_x - 40, center_y - 30), 15, (255, 255, 255), -1)
            cv2.circle(frame, (center_x + 40, center_y - 30), 15, (255, 255, 255), -1)
            cv2.circle(frame, (center_x - 40, center_y - 30), 6, (120, 50, 20), -1)
            cv2.circle(frame, (center_x + 40, center_y - 30), 6, (120, 50, 20), -1)
            
            # Draw mouth
            cv2.ellipse(frame, (center_x, center_y + 40), (30, 15), 0, 0, 180, (50, 50, 180), 3)
            
            out.write(frame)
    finally:
        out.release()
        
    return os.path.abspath(output_path)


if __name__ == "__main__":
    target_file = "data/raw/test_video.mp4"
    saved_path = create_synthetic_face_video(target_file)
    print(f"Generated synthetic test video at: {saved_path}")
