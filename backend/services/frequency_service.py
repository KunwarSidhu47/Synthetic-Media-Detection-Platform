"""
Frequency-Domain Analysis Service using 2D FFT to detect spectral artifacts and GAN anomalies.
"""

from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np

from backend.schemas.detection import FrequencyAnalysisResult


class FrequencyAnalysisService:
    """Service to compute 2D FFT, spectral magnitude distributions, and frequency anomaly metrics."""

    def __init__(self, num_radial_bins: int = 30, high_freq_threshold_ratio: float = 0.6):
        """
        Initialize FrequencyAnalysisService.
        
        Args:
            num_radial_bins: Number of concentric radial bins for 1D azimuthal averaging.
            high_freq_threshold_ratio: Radius fraction (0.0 to 1.0) defining high-frequency spectrum boundary.
        """
        self.num_radial_bins = num_radial_bins
        self.high_freq_threshold_ratio = high_freq_threshold_ratio

    def compute_2d_fft(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 2D Fast Fourier Transform and logarithmic magnitude spectrum.
        
        Args:
            image_rgb: RGB NumPy array of shape (H, W, 3).
            
        Returns:
            Tuple of (fft_shifted_complex: np.ndarray, magnitude_spectrum: np.ndarray)
        """
        if image_rgb is None or image_rgb.size == 0:
            raise ValueError("Input image array is empty or None")

        # Convert to grayscale float
        if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image_rgb.astype(np.float32)

        # 2D Fast Fourier Transform
        fft_complex = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft_complex)

        # Linear magnitude and logarithmic magnitude spectrum
        abs_fft = np.abs(fft_shift)
        magnitude_spectrum = np.log(abs_fft + 1e-8)
        return abs_fft, magnitude_spectrum

    def extract_azimuthal_profile(self, magnitude_spectrum: np.ndarray) -> List[float]:
        """
        Calculate 1D azimuthal (radial) average profile of 2D magnitude spectrum.
        
        Args:
            magnitude_spectrum: 2D array of logarithmic magnitude spectrum.
            
        Returns:
            List of float radial bin averages.
        """
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_radius = np.sqrt(cx ** 2 + cy ** 2)

        bin_boundaries = np.linspace(0, max_radius, self.num_radial_bins + 1)
        azimuthal_avg = []

        for i in range(self.num_radial_bins):
            mask = (r >= bin_boundaries[i]) & (r < bin_boundaries[i + 1])
            if np.any(mask):
                azimuthal_avg.append(float(np.mean(magnitude_spectrum[mask])))
            else:
                azimuthal_avg.append(0.0)

        return azimuthal_avg

    def extract_spectral_features(self, abs_fft: np.ndarray, magnitude_spectrum: np.ndarray) -> Dict[str, float]:
        """
        Extract numerical frequency domain features using linear magnitude and logarithmic spectrum.
        
        Returns:
            Dict containing high_freq_power_ratio, spectral_centroid, and high_freq_variance.
        """
        h, w = abs_fft.shape
        cy, cx = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_radius = float(np.sqrt(cx ** 2 + cy ** 2))

        # Separate low vs high frequency regions (outer 40% radial band)
        high_freq_mask = r >= (max_radius * self.high_freq_threshold_ratio)
        low_freq_mask = ~high_freq_mask

        total_power = float(np.sum(abs_fft)) + 1e-8
        high_freq_power = float(np.sum(abs_fft[high_freq_mask]))
        low_freq_power = float(np.sum(abs_fft[low_freq_mask]))

        high_freq_ratio = high_freq_power / total_power if total_power > 0 else 0.0

        # Spectral Centroid (mean frequency weighted by linear magnitude)
        centroid = float(np.sum(r * abs_fft) / total_power)

        # Variance of high frequency spectrum (checkerboard artifacts produce high variance peaks)
        high_freq_vals = magnitude_spectrum[high_freq_mask]
        high_freq_var = float(np.var(high_freq_vals)) if high_freq_vals.size > 0 else 0.0

        return {
            "high_freq_power_ratio": float(np.clip(high_freq_ratio, 0.0, 1.0)),
            "spectral_centroid": float(centroid),
            "high_freq_variance": float(high_freq_var),
            "total_power": float(total_power),
            "low_freq_power": float(low_freq_power),
            "high_freq_power": float(high_freq_power)
        }

    def predict_crop(self, image_rgb: np.ndarray) -> FrequencyAnalysisResult:
        """
        Analyze a single RGB face crop image in the frequency domain.
        
        Args:
            image_rgb: NumPy array of shape (H, W, 3).
            
        Returns:
            FrequencyAnalysisResult schema.
        """
        abs_fft, mag_spectrum = self.compute_2d_fft(image_rgb)
        azimuthal_prof = self.extract_azimuthal_profile(mag_spectrum)
        features = self.extract_spectral_features(abs_fft, mag_spectrum)

        # Dual-spectrum frequency mapping:
        # 1. Real camera sensors exhibit 0.066 <= high_freq_power_ratio <= 0.32.
        # 2. AI Diffusion video generators exhibit artificial over-smoothing (hr < 0.066).
        # 3. GAN deepfakes & noise patterns exhibit high-frequency checkerboard spikes (hr > 0.32).
        high_ratio = features["high_freq_power_ratio"]

        if high_ratio < 0.076:
            frequency_score = float(np.clip((0.076 - high_ratio) / 0.035 * 0.65 + 0.35, 0.05, 0.95))
        elif high_ratio > 0.18:
            frequency_score = float(np.clip((high_ratio - 0.18) / 0.20 * 0.75 + 0.15, 0.05, 0.95))
        else:
            frequency_score = 0.08

        return FrequencyAnalysisResult(
            frequency_score=round(float(frequency_score), 4),
            high_freq_power_ratio=round(features["high_freq_power_ratio"], 4),
            spectral_centroid=round(features["spectral_centroid"], 4),
            azimuthal_profile=[round(v, 4) for v in azimuthal_prof],
            details={
                "high_freq_variance": round(features["high_freq_variance"], 4),
                "total_power": round(features["total_power"], 4)
            }
        )

    def predict_batch(self, images_rgb: List[np.ndarray]) -> List[FrequencyAnalysisResult]:
        """Analyze a batch of RGB face crop images."""
        return [self.predict_crop(img) for img in images_rgb]
