"""
Device Detection Utilities for SUJBOT2

Provides unified device detection with automatic fallback:
CUDA → MPS → CPU

This module centralizes device selection logic across all components
(embeddings, reranking, indexing) to ensure consistent behavior.
"""

import logging
from typing import Optional, Literal
import torch

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu", "auto"]


class DeviceManager:
    """
    Manages device selection with automatic fallback.

    Priority order:
    1. CUDA (NVIDIA GPUs) - Best performance for large batch processing
    2. MPS (Apple Silicon) - Good performance on M1/M2/M3 Macs
    3. CPU (fallback) - Universal compatibility

    Usage:
        >>> device_mgr = DeviceManager(preferred_device="auto")
        >>> device = device_mgr.get_device()
        'cuda'  # if NVIDIA GPU available

        >>> device_mgr = DeviceManager(preferred_device="mps")
        >>> device = device_mgr.get_device()
        'mps'  # if Apple Silicon available, else fallback
    """

    def __init__(self, preferred_device: DeviceType = "auto"):
        """
        Initialize device manager.

        Args:
            preferred_device: Preferred device type
                - "auto": Automatic detection (cuda > mps > cpu)
                - "cuda": Prefer CUDA, fallback to MPS then CPU
                - "mps": Prefer MPS, fallback to CPU
                - "cpu": Force CPU
        """
        self.preferred_device = preferred_device
        self._detected_device: Optional[str] = None
        self._capabilities: Optional[dict] = None

    def get_device(self) -> str:
        """
        Get best available device based on preferences.

        Returns:
            Device string: "cuda", "mps", or "cpu"
        """
        if self._detected_device is not None:
            return self._detected_device

        # Force CPU if requested
        if self.preferred_device == "cpu":
            self._detected_device = "cpu"
            logger.info("Using CPU (forced by configuration)")
            return "cpu"

        # Auto-detection with fallback chain
        if self.preferred_device == "auto":
            device = self._detect_auto()
        elif self.preferred_device == "cuda":
            device = self._detect_cuda_with_fallback()
        elif self.preferred_device == "mps":
            device = self._detect_mps_with_fallback()
        else:
            logger.warning(f"Unknown device preference: {self.preferred_device}, using auto")
            device = self._detect_auto()

        self._detected_device = device
        return device

    def _detect_auto(self) -> str:
        """
        Auto-detect best device: CUDA > MPS > CPU

        Returns:
            Best available device
        """
        # Try CUDA first
        if torch.cuda.is_available():
            try:
                # Test CUDA availability
                device_name = torch.cuda.get_device_name(0)
                device_count = torch.cuda.device_count()
                logger.info(
                    f"Using CUDA device: {device_name} "
                    f"(detected {device_count} GPU(s))"
                )
                return "cuda"
            except Exception as e:
                logger.warning(f"CUDA detected but unusable: {e}, trying MPS")

        # Try MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                # Test MPS by creating a small tensor
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                logger.info("Using MPS device (Apple Silicon)")
                return "mps"
            except Exception as e:
                logger.warning(f"MPS detected but unusable: {e}, falling back to CPU")

        # Fallback to CPU
        logger.info("Using CPU (no GPU acceleration available)")
        return "cpu"

    def _detect_cuda_with_fallback(self) -> str:
        """
        Prefer CUDA, fallback to MPS then CPU

        Returns:
            CUDA if available, else MPS, else CPU
        """
        if torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(0)
                logger.info(f"Using CUDA device: {device_name}")
                return "cuda"
            except Exception as e:
                logger.warning(f"CUDA unavailable: {e}, trying MPS")
        else:
            logger.info("CUDA not available, trying MPS")

        # Fallback to MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                logger.info("Using MPS device (Apple Silicon)")
                return "mps"
            except Exception as e:
                logger.warning(f"MPS unavailable: {e}, falling back to CPU")
        else:
            logger.info("MPS not available, falling back to CPU")

        # Final fallback
        logger.info("Using CPU")
        return "cpu"

    def _detect_mps_with_fallback(self) -> str:
        """
        Prefer MPS, fallback to CPU (skip CUDA for Apple Silicon preference)

        Returns:
            MPS if available, else CPU
        """
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                logger.info("Using MPS device (Apple Silicon)")
                return "mps"
            except Exception as e:
                logger.warning(f"MPS unavailable: {e}, falling back to CPU")
        else:
            logger.info("MPS not available, falling back to CPU")

        logger.info("Using CPU")
        return "cpu"

    def get_capabilities(self) -> dict:
        """
        Get device capabilities and information.

        Returns:
            Dictionary with device info:
            - device: Current device
            - cuda_available: bool
            - mps_available: bool
            - cuda_device_count: int (0 if not available)
            - cuda_device_name: str (if available)
        """
        if self._capabilities is not None:
            return self._capabilities

        device = self.get_device()

        capabilities = {
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": (
                hasattr(torch.backends, 'mps')
                and torch.backends.mps.is_available()
            ),
            "cuda_device_count": 0,
            "cuda_device_name": None,
        }

        if torch.cuda.is_available():
            try:
                capabilities["cuda_device_count"] = torch.cuda.device_count()
                capabilities["cuda_device_name"] = torch.cuda.get_device_name(0)
            except Exception as e:
                logger.debug(f"Could not get CUDA device info: {e}")

        self._capabilities = capabilities
        return capabilities

    def log_device_info(self):
        """Log detailed device information"""
        caps = self.get_capabilities()
        logger.info("=" * 60)
        logger.info("Device Configuration")
        logger.info("=" * 60)
        logger.info(f"Selected device: {caps['device']}")
        logger.info(f"CUDA available: {caps['cuda_available']}")
        logger.info(f"MPS available: {caps['mps_available']}")

        if caps['cuda_available']:
            logger.info(f"CUDA device count: {caps['cuda_device_count']}")
            if caps['cuda_device_name']:
                logger.info(f"CUDA device name: {caps['cuda_device_name']}")

        logger.info("=" * 60)


def get_device(preferred_device: DeviceType = "auto") -> str:
    """
    Convenience function to get device with fallback.

    Args:
        preferred_device: Preferred device type (auto, cuda, mps, cpu)

    Returns:
        Best available device string

    Example:
        >>> device = get_device("auto")
        'cuda'  # if NVIDIA GPU available

        >>> device = get_device("cpu")
        'cpu'  # forced CPU
    """
    manager = DeviceManager(preferred_device)
    return manager.get_device()


def get_device_manager(preferred_device: DeviceType = "auto") -> DeviceManager:
    """
    Get a device manager instance.

    Args:
        preferred_device: Preferred device type

    Returns:
        DeviceManager instance

    Example:
        >>> mgr = get_device_manager("auto")
        >>> device = mgr.get_device()
        >>> caps = mgr.get_capabilities()
    """
    return DeviceManager(preferred_device)


__all__ = [
    "DeviceManager",
    "DeviceType",
    "get_device",
    "get_device_manager",
]
