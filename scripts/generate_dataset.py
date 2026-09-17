"""
Dataset generation script producing distinct Real/Authentic and Synthetic/Deepfake test videos.
"""

import os
import cv2
import numpy as np


def create_authentic_test_video(output_path: str, duration_sec: int = 3, fps: int = 30) -> str:
    """
    Generate a smooth, natural synthetic video representing REAL / AUTHENTIC camera footage.
    Includes natural skin gradients, smooth continuous head movement, and natural spectral power.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = 640, 480
    total_frames = duration_sec * fps

    codecs = ['mp4v', 'MJPG', 'XVID', 'avc1', 'H264']
    out = None
    for c in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*c)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                out = writer
                break
        except Exception:
            continue

    if out is None or not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for path {output_path}")

    try:
        for i in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (45, 45, 65)

            # Smooth continuous head movement (sinusoidal trajectory)
            center_x = int(width // 2 + 20 * np.sin(2 * np.pi * i / (fps * 2)))
            center_y = int(height // 2 + 10 * np.cos(2 * np.pi * i / (fps * 2)))

            # Face oval with soft natural skin gradient
            cv2.ellipse(frame, (center_x, center_y), (95, 125), 0, 0, 360, (185, 205, 235), -1)

            # Soft facial features
            cv2.circle(frame, (center_x - 35, center_y - 25), 14, (255, 255, 255), -1)
            cv2.circle(frame, (center_x + 35, center_y - 25), 14, (255, 255, 255), -1)
            cv2.circle(frame, (center_x - 35, center_y - 25), 6, (120, 50, 20), -1)
            cv2.circle(frame, (center_x + 35, center_y - 25), 6, (120, 50, 20), -1)

            # Mouth
            cv2.ellipse(frame, (center_x, center_y + 35), (28, 12), 0, 0, 180, (60, 60, 170), 2)

            # Gaussian blur to simulate natural lens anti-aliasing
            frame = cv2.GaussianBlur(frame, (3, 3), 0)
            out.write(frame)
    finally:
        out.release()

    return os.path.abspath(output_path)


def create_synthetic_test_video(output_path: str, duration_sec: int = 3, fps: int = 30) -> str:
    """
    Generate a video containing explicit DEEPFAKE / SYNTHETIC artifacts:
    1. Visible rectangular face swap boundary seam.
    2. High-frequency 2D grid noise artifacts.
    3. Abrupt temporal frame flicker spikes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    width, height = 640, 480
    total_frames = duration_sec * fps

    codecs = ['mp4v', 'MJPG', 'XVID', 'avc1', 'H264']
    out = None
    for c in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*c)
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer.isOpened():
                out = writer
                break
        except Exception:
            continue

    if out is None or not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for path {output_path}")

    try:
        np.random.seed(42)
        for i in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:, :] = (30, 30, 40)

            # Base position with abrupt random temporal position jumps every 3 frames
            if (i // 3) % 2 == 1:
                center_x = int(width // 2 + 120)
                center_y = int(height // 2 - 80)
            else:
                center_x = int(width // 2 - 120)
                center_y = int(height // 2 + 80)

            # Face oval
            cv2.ellipse(frame, (center_x, center_y), (105, 135), 0, 0, 360, (140, 160, 240), -1)
            cv2.circle(frame, (center_x - 40, center_y - 30), 16, (255, 255, 255), -1)
            cv2.circle(frame, (center_x + 40, center_y - 30), 16, (255, 255, 255), -1)
            cv2.circle(frame, (center_x - 40, center_y - 30), 7, (200, 20, 20), -1)
            cv2.circle(frame, (center_x + 40, center_y - 30), 7, (200, 20, 20), -1)
            cv2.ellipse(frame, (center_x, center_y + 40), (35, 18), 0, 0, 180, (20, 20, 220), 4)

            # ARTIFACT 1: Face Swap Boundary Seam Boxes
            rect_left = max(0, center_x - 110)
            rect_top = max(0, center_y - 140)
            rect_right = min(width, center_x + 110)
            rect_bottom = min(height, center_y + 140)
            cv2.rectangle(frame, (rect_left, rect_top), (rect_right, rect_bottom), (0, 255, 255), 8)
            cv2.rectangle(frame, (rect_left + 10, rect_top + 10), (rect_right - 10, rect_bottom - 10), (255, 0, 255), 6)

            # ARTIFACT 2: Extreme High-Frequency Salt & Pepper Spatial Noise
            noise = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            mask = np.random.rand(height, width) < 0.35
            frame[mask] = noise[mask]

            out.write(frame)
    finally:
        out.release()

    return os.path.abspath(output_path)


if __name__ == "__main__":
    auth_path = create_authentic_test_video("data/raw/authentic_sample.mp4")
    synth_path = create_synthetic_test_video("data/raw/synthetic_sample.mp4")
    print(f"Authentic Sample Video generated at: {auth_path}")
    print(f"Synthetic Deepfake Sample Video generated at: {synth_path}")
