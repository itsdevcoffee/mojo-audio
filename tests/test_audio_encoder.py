"""Tests for mojo_audio.models.AudioEncoder.

Level 1 tests (no download): no marker — run by default via test-models.
Level 2 tests (download required): @pytest.mark.slow — skipped by default.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
from max.driver import accelerator_count


# --- Fixtures ---

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def audio_1s(rng):
    """1 second of synthetic 16kHz audio, normalized."""
    return rng.standard_normal((1, 16000)).astype(np.float32)


@pytest.fixture
def cpu_device():
    from max.driver import CPU
    return CPU()


@pytest.fixture
def gpu_available():
    return accelerator_count() > 0


# --- Smoke tests (always pass if imports work) ---

def test_package_importable():
    """mojo_audio.models must be importable."""
    from models import AudioEncoder
    assert AudioEncoder is not None


def test_max_engine_importable():
    """MAX Engine must be accessible."""
    from max import engine
    from max.driver import accelerator_count
    assert True


def test_gpu_session_creatable():
    """GPU InferenceSession must work if GPU is available."""
    from max import engine
    from max.driver import Accelerator, CPU, accelerator_count
    if accelerator_count() > 0:
        session = engine.InferenceSession(devices=[Accelerator()])
    else:
        session = engine.InferenceSession(devices=[CPU()])
    assert session is not None
