"""
Unit tests for FrequencyAnalysisService.
"""

import pytest
import numpy as np

from backend.services.frequency_service import FrequencyAnalysisService


@pytest.fixture(scope="module")
def freq_service():
    return FrequencyAnalysisService(num_radial_bins=30)


@pytest.fixture
def natural_smooth_crop():
    """Create a smooth gradient image simulating natural soft skin texture."""
    x = np.linspace(0, 1, 224)
    y = np.linspace(0, 1, 224)
    xx, yy = np.meshgrid(x, y)
    smooth_b = (xx * 200 + yy * 50).astype(np.uint8)
    smooth_g = (xx * 180 + yy * 60).astype(np.uint8)
    smooth_r = (xx * 190 + yy * 70).astype(np.uint8)
    return np.stack([smooth_r, smooth_g, smooth_b], axis=-1)


@pytest.fixture
def synthetic_grid_crop(natural_smooth_crop):
    """Add high-frequency checkerboard / grid noise to image to simulate GAN spectral artifacts."""
    grid_img = natural_smooth_crop.copy().astype(np.float32)
    # Add high-frequency alternating pixel grid pattern (Nyquist frequency component)
    grid_img[::2, ::2] += 80.0
    grid_img[1::2, 1::2] += 80.0
    return np.clip(grid_img, 0, 255).astype(np.uint8)


def test_compute_2d_fft(freq_service, natural_smooth_crop):
    fft_shift, mag_spectrum = freq_service.compute_2d_fft(natural_smooth_crop)
    assert fft_shift.shape == (224, 224)
    assert mag_spectrum.shape == (224, 224)


def test_azimuthal_profile_length(freq_service, natural_smooth_crop):
    _, mag_spectrum = freq_service.compute_2d_fft(natural_smooth_crop)
    azimuthal_profile = freq_service.extract_azimuthal_profile(mag_spectrum)
    assert len(azimuthal_profile) == 30


def test_predict_crop_bounds(freq_service, natural_smooth_crop):
    result = freq_service.predict_crop(natural_smooth_crop)
    assert 0.0 <= result.frequency_score <= 1.0
    assert 0.0 <= result.high_freq_power_ratio <= 1.0
    assert len(result.azimuthal_profile) == 30


def test_grid_noise_spectral_contrast(freq_service, natural_smooth_crop, synthetic_grid_crop):
    res_smooth = freq_service.predict_crop(natural_smooth_crop)
    res_grid = freq_service.predict_crop(synthetic_grid_crop)

    # Synthetic grid artifacts should produce higher high-frequency ratio and anomaly score
    assert res_grid.high_freq_power_ratio > res_smooth.high_freq_power_ratio
    assert res_grid.frequency_score > res_smooth.frequency_score
