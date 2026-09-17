"""
Health check route endpoint.
"""

from fastapi import APIRouter
import torch
import time

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    """
    Service health check endpoint.
    Returns API status, PyTorch compute device, and server timestamp.
    """
    if torch.cuda.is_available():
        device_type = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    return {
        "status": "healthy",
        "service": "Synthetic Media Detection & Analysis Platform API",
        "version": "1.0.0",
        "torch_device": device_type,
        "timestamp": time.time()
    }
