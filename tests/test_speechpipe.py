import torch
import pytest
from tts_engine.speechpipe import convert_to_audio, CUSTOM_TOKEN_PREFIX

class DummyModel:
    def decode(self, codes):
        # Return tensor with shape [1,1,4096] to simulate audio output
        return torch.ones((1, 1, 4096), dtype=torch.float32)

@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    import tts_engine.speechpipe as sp
    monkeypatch.setattr(sp, 'model', DummyModel())
    return sp


def test_convert_to_audio_too_short():
    # Less than one frame of tokens should return None
    assert convert_to_audio([1,2,3,4,5,6], count=0) is None


def test_convert_to_audio_valid():
    # Two frames of tokens should return valid audio bytes
    data = list(range(14))
    audio_bytes = convert_to_audio(data, count=1)
    assert isinstance(audio_bytes, (bytes, bytearray))
    # Audio slice size = 4096-2048 = 2048 samples * 2 bytes/sample = 4096 bytes
    assert len(audio_bytes) == 2048 * 2
