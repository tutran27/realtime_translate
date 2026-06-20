import asyncio
import base64
import concurrent.futures
import io
import logging
import re
import time
import unicodedata
import wave

import numpy as np


DEFAULT_TTS_MODE = "v3turbo"
DEFAULT_TTS_VOICE = "Bình An"
DEFAULT_FIRST_CHUNK_CHARS = 48
DEFAULT_MAX_CHUNK_CHARS = 72
SILENCE_THRESHOLD = 0.003
LEADING_SILENCE_KEEP_MS = 35
TRAILING_SILENCE_KEEP_MS = 45
MINOR_PAUSE_KEEP_MS = 80
SENTENCE_PAUSE_KEEP_MS = 140


class TTSRequest:
    def __init__(
        self,
        text: str,
        voice: str | None,
        output: asyncio.Queue,
    ) -> None:
        self.text = text
        self.voice = voice
        self.output = output
        self.queued_at = time.perf_counter()
        self.cancelled = False


def split_tts_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    first_chunk_chars: int = DEFAULT_FIRST_CHUNK_CHARS,
) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    clauses = re.split(r"(?<=[.!?;:,])\s+", text)
    chunks: list[str] = []
    current = ""
    for clause in clauses:
        chunk_limit = first_chunk_chars if not chunks else max_chars
        if len(clause) <= chunk_limit:
            candidate = f"{current} {clause}".strip()
            if len(candidate) <= chunk_limit:
                current = candidate
                continue
        if current:
            chunks.append(current)
            current = ""
        chunk_limit = first_chunk_chars if not chunks else max_chars
        while len(clause) > chunk_limit:
            split_at = clause.rfind(" ", 0, chunk_limit + 1)
            if split_at <= 0:
                split_at = chunk_limit
            chunks.append(clause[:split_at].strip())
            clause = clause[split_at:].strip()
            chunk_limit = max_chars
        current = clause
    if current:
        chunks.append(current)
    return chunks


def float_audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.asarray(audio, dtype=np.float32)
    if pcm.ndim != 1:
        pcm = pcm.reshape(-1)
    pcm = np.clip(pcm, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def trim_tts_silence(
    audio: np.ndarray,
    sample_rate: int,
    text: str,
) -> np.ndarray:
    pcm = np.asarray(audio, dtype=np.float32).reshape(-1)
    if pcm.size == 0:
        return pcm

    peak = float(np.max(np.abs(pcm)))
    threshold = max(SILENCE_THRESHOLD, peak * 0.02)
    active = np.flatnonzero(np.abs(pcm) >= threshold)
    if active.size == 0:
        return pcm

    leading_keep = int(sample_rate * LEADING_SILENCE_KEEP_MS / 1000)
    stripped = text.rstrip()
    if stripped.endswith((".", "!", "?", "…")):
        trailing_ms = SENTENCE_PAUSE_KEEP_MS
    elif stripped.endswith((",", ";", ":")):
        trailing_ms = MINOR_PAUSE_KEEP_MS
    else:
        trailing_ms = TRAILING_SILENCE_KEEP_MS
    trailing_keep = int(sample_rate * trailing_ms / 1000)

    start = max(0, int(active[0]) - leading_keep)
    end = min(pcm.size, int(active[-1]) + trailing_keep + 1)
    return pcm[start:end]


class TTSService:
    def __init__(
        self,
        mode: str = DEFAULT_TTS_MODE,
        default_voice: str = DEFAULT_TTS_VOICE,
    ) -> None:
        self.mode = mode
        self.default_voice = default_voice
        self.tts = None
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="vieneu-tts",
        )
        self.queue: asyncio.Queue[TTSRequest | None] = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None
        self.preset_voices: list[tuple[str, str]] = []

    def initialize_sync(self) -> None:
        if self.tts is not None:
            return
        started_at = time.perf_counter()
        try:
            from vieneu import Vieneu
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Thieu package vieneu. Cai bang: pip install vieneu"
            ) from exc
        self.tts = Vieneu(mode=self.mode)
        self.preset_voices = list(self.tts.list_preset_voices())
        self.default_voice = self.resolve_voice(self.default_voice)
        self.tts.infer(
            "Xin chào.",
            voice=self.default_voice,
            apply_watermark=False,
        )
        logging.info(
            "TTS model ready in %.2fs",
            time.perf_counter() - started_at,
        )

    def resolve_voice(self, voice: str | None) -> str | None:
        voice = (voice or "").strip()
        requested = voice or self.default_voice
        available = [
            candidate
            for label, value in self.preset_voices
            for candidate in (value, label)
            if candidate
        ]
        if requested in available:
            return requested

        normalized = self.normalize_voice(requested)
        for candidate in available:
            if self.normalize_voice(candidate) == normalized:
                return candidate
        if self.preset_voices:
            logging.warning(
                "Voice %r khong hop le, fallback ve %r",
                requested,
                self.preset_voices[0][1],
            )
            return self.preset_voices[0][1]
        return requested or None

    @staticmethod
    def normalize_voice(value: str) -> str:
        normalized = unicodedata.normalize("NFKD", value)
        return normalized.encode("ascii", "ignore").decode("ascii").casefold()

    def synthesize_sync(
        self,
        text: str,
        voice: str | None = None,
    ) -> dict:
        self.initialize_sync()
        text = text.strip()
        if not text:
            raise ValueError("Text TTS rong")

        selected_voice = self.resolve_voice(voice)
        audio = self.tts.infer(
            text,
            voice=selected_voice,
            apply_watermark=False,
        )
        sample_rate = int(getattr(self.tts, "sample_rate", 48000))
        audio = trim_tts_silence(audio, sample_rate, text)
        wav_bytes = float_audio_to_wav_bytes(audio, sample_rate)
        return {
            "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
            "sample_rate": sample_rate,
            "voice": selected_voice,
            "duration_ms": int(len(audio) * 1000 / sample_rate),
        }

    async def synthesize(
        self,
        text: str,
        voice: str | None = None,
    ) -> dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self.synthesize_sync,
            text,
            voice,
        )

    async def start(self) -> None:
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(
                self._run_worker(),
                name="vieneu-tts-worker",
            )

    async def _run_worker(self) -> None:
        while True:
            request = await self.queue.get()
            if request is None:
                return
            if request.cancelled:
                continue

            chunks = split_tts_text(request.text)
            queue_wait_ms = (
                time.perf_counter() - request.queued_at
            ) * 1000
            try:
                for index, chunk in enumerate(chunks):
                    if request.cancelled:
                        break
                    inference_started_at = time.perf_counter()
                    synthesized = await self.synthesize(
                        chunk,
                        voice=request.voice,
                    )
                    synthesized["text"] = chunk
                    synthesized["chunk_index"] = index
                    synthesized["chunk_count"] = len(chunks)
                    synthesized["queue_wait_ms"] = queue_wait_ms
                    synthesized["inference_ms"] = (
                        time.perf_counter() - inference_started_at
                    ) * 1000
                    await request.output.put(synthesized)
            except Exception as exc:
                await request.output.put(exc)
            finally:
                await request.output.put(None)

    async def synthesize_chunks(
        self,
        text: str,
        voice: str | None = None,
    ):
        await self.start()
        output: asyncio.Queue = asyncio.Queue()
        request = TTSRequest(text=text, voice=voice, output=output)
        await self.queue.put(request)
        try:
            while True:
                item = await output.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            request.cancelled = True

    async def stop(self) -> None:
        if self.worker_task is not None:
            await self.queue.put(None)
            await self.worker_task
            self.worker_task = None
        if self.tts is not None:
            try:
                self.tts.close()
            except Exception:
                pass
        self.executor.shutdown(wait=True, cancel_futures=True)
