"""
Health check route endpoint.
"""

import time
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check():
    """
    Service health check endpoint.
    Returns API status, PyTorch compute device (if available), and server timestamp.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_type = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
        else:
            device_type = "cpu"
    except ImportError:
        device_type = "cpu (torch not installed)"

    return {
        "status": "healthy",
        "service": "Synthetic Media Detection & Analysis Platform API",
        "version": "1.0.0",
        "torch_device": device_type,
        "timestamp": time.time()
    }
