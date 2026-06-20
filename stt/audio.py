import logging
import math
import time

import numpy as np


SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
VAD_FRAME_SAMPLES = 512
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * BYTES_PER_SAMPLE


def load_vad_model():
    import torch

    try:
        from silero_vad import load_silero_vad
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Thieu Silero VAD. Cai: pip install silero-vad"
        ) from exc

    started_at = time.perf_counter()
    torch_threads = torch.get_num_threads()
    try:
        model = load_silero_vad()
    finally:
        torch.set_num_threads(torch_threads)
    logging.info(
        "Silero VAD ready in %.2fs",
        time.perf_counter() - started_at,
    )
    return model


class AudioPreprocessor:
    def __init__(self, highpass_hz: float, gain_db: float) -> None:
        self.gain = 10 ** (gain_db / 20.0)
        self.x_prev = 0.0
        self.y_prev = 0.0
        self.highpass_r = (
            math.exp(-2.0 * math.pi * highpass_hz / SAMPLE_RATE)
            if highpass_hz > 0
            else 0.0
        )

    def process(self, pcm16: bytes) -> bytes:
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        if audio.size == 0:
            return b""

        if self.highpass_r:
            filtered = np.empty_like(audio)
            x_prev = self.x_prev
            y_prev = self.y_prev
            for index, sample in enumerate(audio):
                y = float(sample) - x_prev + self.highpass_r * y_prev
                filtered[index] = y
                x_prev = float(sample)
                y_prev = y
            self.x_prev = x_prev
            self.y_prev = y_prev
            audio = filtered

        audio *= self.gain
        audio = np.clip(audio, -0.999, 0.999)
        return (audio * 32767.0).astype("<i2").tobytes()


class StreamingVAD:
    def __init__(
        self,
        model,
        threshold: float,
        min_silence_ms: int,
        speech_pad_ms: int,
    ) -> None:
        try:
            from silero_vad import VADIterator
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Thieu Silero VAD. Cai: pip install silero-vad"
            ) from exc

        self.iterator = VADIterator(
            model,
            threshold=threshold,
            sampling_rate=SAMPLE_RATE,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )

    def process(self, pcm16_frame: bytes) -> dict | None:
        import torch

        audio = (
            np.frombuffer(pcm16_frame, dtype="<i2").astype(np.float32) / 32768.0
        )
        return self.iterator(torch.from_numpy(audio), return_seconds=False)

    def reset(self) -> None:
        self.iterator.reset_states()
