import argparse
import asyncio
import base64
import collections
import concurrent.futures
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
import websockets
from stt.audio import (
    SAMPLE_RATE,
    VAD_FRAME_BYTES,
    VAD_FRAME_SAMPLES,
    AudioPreprocessor,
    StreamingVAD,
    load_vad_model,
)

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_TTS_VOICE = "Bình An"
VIETNAMESE_LANGUAGE_ALIASES = {"vi", "vietnamese", "vie_latn"}


@dataclass
class Session:
    model: str
    state: object
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    language: str | None = None
    context: str = ""
    pending_audio: bytearray = field(default_factory=bytearray)
    committed_text: str = ""
    segment_bytes: int = 0
    segment_has_speech: bool = False
    vad: object | None = None
    vad_buffer: bytearray = field(default_factory=bytearray)
    vad_active: bool = False
    pre_roll: collections.deque = field(default_factory=collections.deque)
    preprocessor: AudioPreprocessor | None = None
    segment_id: int = 0
    last_segment_text: str = ""
    translation_enabled: bool = False
    translation_target: str | None = None
    translation_source: str | None = None
    translation_tasks: set = field(default_factory=set)
    tts_enabled: bool = False
    tts_voice: str | None = None
    tts_source: str = "translation"
    tts_tasks: set = field(default_factory=set)


class QwenRealtimeServer:
    def __init__(
        self,
        model: str,
        language: str | None,
        context: str,
        gpu_memory_utilization: float,
        step_ms: int,
        chunk_size_sec: float,
        max_segment_sec: float,
        max_model_len: int,
        vad_enabled: bool,
        vad_threshold: float,
        vad_min_silence_ms: int,
        vad_speech_pad_ms: int,
        vad_pre_roll_ms: int,
        highpass_hz: float,
        audio_gain_db: float,
        translation_enabled: bool,
        translation_model: str,
        translation_device: str,
        tts_enabled: bool,
        tts_mode: str,
        tts_voice: str,
    ) -> None:
        if step_ms <= 0:
            raise ValueError("--step-ms phai > 0")
        if chunk_size_sec <= 0:
            raise ValueError("--chunk-size-sec phai > 0")
        if max_segment_sec <= 0:
            raise ValueError("--max-segment-sec phai > 0")
        if not 0 < vad_threshold < 1:
            raise ValueError("--vad-threshold phai nam trong (0, 1)")
        if vad_min_silence_ms < 0 or vad_speech_pad_ms < 0:
            raise ValueError("Tham so VAD theo ms phai >= 0")

        self.model_name = model
        self.language = language
        self.context = context
        self.step_bytes = int(SAMPLE_RATE * step_ms / 1000) * 2
        self.chunk_size_sec = chunk_size_sec
        self.max_segment_bytes = int(SAMPLE_RATE * max_segment_sec) * 2
        self.vad_enabled = vad_enabled
        self.vad_threshold = vad_threshold
        self.vad_min_silence_ms = vad_min_silence_ms
        self.vad_speech_pad_ms = vad_speech_pad_ms
        self.pre_roll_frames = max(
            1,
            math.ceil(vad_pre_roll_ms / (VAD_FRAME_SAMPLES * 1000 / SAMPLE_RATE)),
        )
        self.highpass_hz = highpass_hz
        self.audio_gain_db = audio_gain_db
        self.tts_default_voice = tts_voice
        self.vad_model = load_vad_model() if vad_enabled else None
        self.translation_service = None
        if translation_enabled:
            from translate.service import TranslationService

            self.translation_service = TranslationService(
                model_name=translation_model,
                device=translation_device,
            )
        self.tts_service = None
        if tts_enabled:
            from tts.service import TTSService

            self.tts_service = TTSService(
                mode=tts_mode,
                default_voice=tts_voice,
            )
            logging.info("Dang tai TTS mode %s truoc vLLM", tts_mode)
            self.tts_service.initialize_sync()
        self.asr_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="qwen-asr",
        )
        self.inference_lock = asyncio.Lock()

        logging.info("Dang tai model %s", model)
        started_at = time.perf_counter()
        from qwen_asr import Qwen3ASRModel

        self.asr = Qwen3ASRModel.LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_new_tokens=32,
            max_model_len=max_model_len,
        )
        warmup_started_at = time.perf_counter()
        warmup_state = self.asr.init_streaming_state(
            context="",
            language="English",
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=self.chunk_size_sec,
        )
        self.asr.streaming_transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            warmup_state,
        )
        self.asr.finish_streaming_transcribe(warmup_state)
        logging.info(
            "ASR model ready in %.2fs (warmup %.2fs)",
            time.perf_counter() - started_at,
            time.perf_counter() - warmup_started_at,
        )

    def new_state(self, language: str | None, context: str):
        return self.asr.init_streaming_state(
            context=context,
            language=language,
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=self.chunk_size_sec,
        )

    def new_session(self) -> Session:
        vad = (
            StreamingVAD(
                model=self.vad_model,
                threshold=self.vad_threshold,
                min_silence_ms=self.vad_min_silence_ms,
                speech_pad_ms=self.vad_speech_pad_ms,
            )
            if self.vad_enabled
            else None
        )
        return Session(
            model=self.model_name,
            state=self.new_state(self.language, self.context),
            language=self.language,
            context=self.context,
            vad=vad,
            pre_roll=collections.deque(maxlen=self.pre_roll_frames),
            preprocessor=AudioPreprocessor(
                highpass_hz=self.highpass_hz,
                gain_db=self.audio_gain_db,
            ),
            tts_enabled=bool(self.tts_service),
            tts_voice=self.tts_default_voice if self.tts_service else None,
        )

    @staticmethod
    def join_text(left: str, right: str) -> str:
        left = left.strip()
        right = right.strip()
        if not left:
            return right
        if not right:
            return left
        return f"{left} {right}"

    @staticmethod
    def segment_text(session: Session) -> str:
        return str(getattr(session.state, "text", "") or "")

    @staticmethod
    def common_prefix_length(left: str, right: str) -> int:
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        return index

    @staticmethod
    def is_vietnamese(language: str | None) -> bool:
        if not language:
            return False
        return language.strip().lower() in VIETNAMESE_LANGUAGE_ALIASES

    async def send_json(self, websocket, session: Session, payload: dict) -> None:
        async with session.send_lock:
            await websocket.send(json.dumps(payload, ensure_ascii=False))

    def schedule_tts(
        self,
        websocket,
        session: Session,
        *,
        segment_id: int,
        text: str,
        source: str,
        language: str | None,
    ) -> None:
        if not self.tts_service or not session.tts_enabled:
            return
        if not text.strip():
            return
        if source not in {"transcript", "translation"}:
            return
        if not self.is_vietnamese(language):
            return

        task = asyncio.create_task(
            self.synthesize_segment(
                websocket,
                session,
                segment_id=segment_id,
                text=text,
                voice=session.tts_voice,
                source=source,
                language=language,
            )
        )
        session.tts_tasks.add(task)
        task.add_done_callback(session.tts_tasks.discard)

    async def synthesize_segment(
        self,
        websocket,
        session: Session,
        *,
        segment_id: int,
        text: str,
        voice: str | None,
        source: str,
        language: str | None,
    ) -> None:
        try:
            started_at = time.perf_counter()
            total_audio_ms = 0
            async for synthesized in self.tts_service.synthesize_chunks(
                text,
                voice=voice,
            ):
                total_audio_ms += synthesized["duration_ms"]
                await self.send_json(
                    websocket,
                    session,
                    {
                        "type": "tts.segment.audio",
                        "segment_id": segment_id,
                        "chunk_index": synthesized["chunk_index"],
                        "chunk_count": synthesized["chunk_count"],
                        "text": synthesized["text"],
                        "voice": synthesized["voice"],
                        "sample_rate": synthesized["sample_rate"],
                        "duration_ms": synthesized["duration_ms"],
                        "language": language,
                        "source": source,
                        "mime_type": "audio/wav",
                        "audio": synthesized["audio_base64"],
                    },
                )
                if synthesized["chunk_index"] == 0:
                    logging.info(
                        "TTS segment=%s first_audio=%.0fms queue_wait=%.0fms "
                        "inference=%.0fms chunks=%s",
                        segment_id,
                        (time.perf_counter() - started_at) * 1000,
                        synthesized["queue_wait_ms"],
                        synthesized["inference_ms"],
                        synthesized["chunk_count"],
                    )
            logging.info(
                "TTS segment=%s total=%.0fms audio=%sms queue_depth=%s",
                segment_id,
                (time.perf_counter() - started_at) * 1000,
                total_audio_ms,
                self.tts_service.queue.qsize(),
            )
        except websockets.ConnectionClosed:
            return
        except Exception as exc:
            try:
                await self.send_json(
                    websocket,
                    session,
                    {
                        "type": "tts.segment.error",
                        "segment_id": segment_id,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    },
                )
            except websockets.ConnectionClosed:
                pass

    async def transcribe_pending(
        self, websocket, session: Session, *, flush: bool = False
    ) -> None:
        while len(session.pending_audio) >= self.step_bytes or (
            flush and session.pending_audio
        ):
            take = (
                self.step_bytes
                if len(session.pending_audio) >= self.step_bytes
                else len(session.pending_audio)
            )
            pcm16 = bytes(session.pending_audio[:take])
            del session.pending_audio[:take]
            audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0

            async with self.inference_lock:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self.asr_executor,
                    self.asr.streaming_transcribe, audio, session.state
                )

            await self.send_update(websocket, session)

    async def send_update(self, websocket, session: Session) -> None:
        text = self.segment_text(session)
        language = getattr(session.state, "language", None)
        if text == session.last_segment_text:
            return
        replace_from = self.common_prefix_length(
            session.last_segment_text, text
        )
        delta = text[replace_from:]
        session.last_segment_text = text
        await self.send_json(
            websocket,
            session,
            {
                "type": "transcription.updated",
                "text": delta,
                "replace_from": replace_from,
                "segment_id": session.segment_id,
                "language": language,
            },
        )

    async def finish_current_state(self, websocket, session: Session) -> None:
        if not session.segment_has_speech and not session.pending_audio:
            return
        await self.transcribe_pending(websocket, session, flush=True)
        async with self.inference_lock:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                self.asr_executor,
                self.asr.finish_streaming_transcribe, session.state
            )
        await self.send_update(websocket, session)

    async def commit_current_segment(
        self, websocket, session: Session
    ) -> None:
        segment_text = self.segment_text(session).strip()
        if not segment_text:
            return
        segment_language = getattr(session.state, "language", None)
        await self.send_json(
            websocket,
            session,
            {
                "type": "transcription.segment.done",
                "segment_id": session.segment_id,
                "text": segment_text,
                "language": segment_language,
            },
        )
        session.committed_text = self.join_text(
            session.committed_text, segment_text
        )
        if session.tts_source == "transcript":
            self.schedule_tts(
                websocket,
                session,
                segment_id=session.segment_id,
                text=segment_text,
                source="transcript",
                language=segment_language,
            )
        self.schedule_translation(
            websocket,
            session,
            segment_id=session.segment_id,
            text=segment_text,
            source_language=(
                session.translation_source
                or getattr(session.state, "language", None)
                or session.language
            ),
        )

    def schedule_translation(
        self,
        websocket,
        session: Session,
        *,
        segment_id: int,
        text: str,
        source_language: str | None,
    ) -> None:
        if (
            not self.translation_service
            or not session.translation_enabled
            or not session.translation_target
        ):
            return

        task = asyncio.create_task(
            self.translate_segment(
                websocket,
                session,
                segment_id=segment_id,
                text=text,
                source_language=source_language,
                target_language=session.translation_target,
            )
        )
        session.translation_tasks.add(task)
        task.add_done_callback(session.translation_tasks.discard)

    async def translate_segment(
        self,
        websocket,
        session: Session,
        *,
        segment_id: int,
        text: str,
        source_language: str | None,
        target_language: str,
    ) -> None:
        try:
            if not source_language:
                raise ValueError(
                    "Khong xac dinh duoc ngon ngu nguon de dich"
                )
            started_at = time.perf_counter()
            translation = await self.translation_service.translate(
                text,
                source_language,
                target_language,
            )
            translated_text = translation.text
            await self.send_json(
                websocket,
                session,
                {
                    "type": "translation.segment.done",
                    "segment_id": segment_id,
                    "text": translated_text,
                    "source_language": source_language,
                    "target_language": target_language,
                },
            )
            if session.tts_source == "translation":
                self.schedule_tts(
                    websocket,
                    session,
                    segment_id=segment_id,
                    text=translated_text,
                    source="translation",
                    language=target_language,
                )
            logging.info(
                "Translation segment=%s total=%.0fms queue_wait=%.0fms "
                "inference=%.0fms batch=%s",
                segment_id,
                (time.perf_counter() - started_at) * 1000,
                translation.queue_wait_ms,
                translation.inference_ms,
                translation.batch_size,
            )
        except websockets.ConnectionClosed:
            return
        except Exception as exc:
            try:
                await self.send_json(
                    websocket,
                    session,
                    {
                        "type": "translation.segment.error",
                        "segment_id": segment_id,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    },
                )
            except websockets.ConnectionClosed:
                pass

    async def wait_for_translations(self, session: Session) -> None:
        tasks = list(session.translation_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_for_tts(self, session: Session) -> None:
        tasks = list(session.tts_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def rollover_segment(self, websocket, session: Session) -> None:
        await self.finish_current_state(websocket, session)
        await self.commit_current_segment(websocket, session)
        session.state = self.new_state(session.language, session.context)
        session.segment_bytes = 0
        session.segment_has_speech = False
        session.segment_id += 1
        session.last_segment_text = ""

    async def append_asr_audio(
        self, websocket, session: Session, pcm16_bytes: bytes
    ) -> None:
        offset = 0
        while offset < len(pcm16_bytes):
            remaining = self.max_segment_bytes - session.segment_bytes
            take = min(remaining, len(pcm16_bytes) - offset)
            session.pending_audio.extend(pcm16_bytes[offset : offset + take])
            session.segment_bytes += take
            session.segment_has_speech = True
            offset += take
            await self.transcribe_pending(websocket, session)

            if session.segment_bytes >= self.max_segment_bytes:
                await self.rollover_segment(websocket, session)

    async def process_vad_frame(
        self, websocket, session: Session, frame: bytes
    ) -> None:
        event = session.vad.process(frame)

        if not session.vad_active:
            if event and "start" in event:
                session.vad_active = True
                buffered = b"".join(session.pre_roll)
                session.pre_roll.clear()
                if buffered:
                    await self.append_asr_audio(websocket, session, buffered)
                await self.append_asr_audio(websocket, session, frame)
            else:
                session.pre_roll.append(frame)
            return

        await self.append_asr_audio(websocket, session, frame)
        if event and "end" in event:
            await self.rollover_segment(websocket, session)
            session.vad_active = False
            session.pre_roll.clear()

    async def append_audio(
        self, websocket, session: Session, pcm16_bytes: bytes
    ) -> None:
        processed = session.preprocessor.process(pcm16_bytes)

        if not session.vad:
            await self.append_asr_audio(websocket, session, processed)
            return

        session.vad_buffer.extend(processed)
        while len(session.vad_buffer) >= VAD_FRAME_BYTES:
            frame = bytes(session.vad_buffer[:VAD_FRAME_BYTES])
            del session.vad_buffer[:VAD_FRAME_BYTES]
            await self.process_vad_frame(websocket, session, frame)

    async def finish(self, websocket, session: Session) -> None:
        if session.vad_active and session.vad_buffer:
            await self.append_asr_audio(
                websocket, session, bytes(session.vad_buffer)
            )
        session.vad_buffer.clear()
        session.pre_roll.clear()
        await self.finish_current_state(websocket, session)
        await self.commit_current_segment(websocket, session)
        final_text = session.committed_text
        await self.wait_for_translations(session)
        await self.wait_for_tts(session)
        await self.send_json(
            websocket,
            session,
            {
                "type": "transcription.done",
                "text": final_text,
                "language": getattr(session.state, "language", None),
            },
        )
        session.state = self.new_state(session.language, session.context)
        session.pending_audio.clear()
        session.segment_bytes = 0
        session.segment_has_speech = False
        session.vad_buffer.clear()
        session.vad_active = False
        session.pre_roll.clear()
        if session.vad:
            session.vad.reset()
        session.segment_id += 1
        session.last_segment_text = ""

    async def handle(self, websocket) -> None:
        session = None
        request_path = getattr(websocket, "path", None)
        if request_path is None:
            request_path = getattr(
                getattr(websocket, "request", None), "path", "/"
            )
        if request_path != "/v1/realtime":
            await websocket.close(code=1008, reason="Endpoint khong hop le")
            return

        try:
            session = self.new_session()
            await self.send_json(
                websocket,
                session,
                {
                    "type": "session.created",
                    "id": f"sess_{uuid.uuid4().hex}",
                    "model": self.model_name,
                    "tts": {
                        "enabled": bool(self.tts_service),
                        "default_voice": self.tts_default_voice,
                        "voices": [
                            {
                                "label": label,
                                "value": voice_id,
                            }
                            for label, voice_id in (
                                self.tts_service.preset_voices
                                if self.tts_service
                                else []
                            )
                        ],
                    },
                },
            )

            async for raw_message in websocket:
                if isinstance(raw_message, bytes):
                    await self.append_audio(websocket, session, raw_message)
                    continue

                message = json.loads(raw_message)
                event_type = message.get("type")

                if event_type == "session.update":
                    requested_model = message.get("model")
                    if requested_model and requested_model != self.model_name:
                        raise ValueError(
                            f"Server dang phuc vu {self.model_name}, "
                            f"khong phai {requested_model}"
                        )
                    has_language = "language" in message
                    has_context = "context" in message
                    has_translation = any(
                        key in message
                        for key in (
                            "translation_enabled",
                            "translation_target",
                            "translation_source",
                        )
                    )
                    has_tts = any(
                        key in message
                        for key in (
                            "tts_enabled",
                            "tts_voice",
                            "tts_source",
                        )
                    )
                    if has_language:
                        session.language = message.get("language") or None
                    if has_context:
                        session.context = str(message.get("context") or "")
                    if has_translation:
                        session.translation_enabled = bool(
                            message.get("translation_enabled", True)
                        )
                        session.translation_target = (
                            message.get("translation_target") or None
                        )
                        session.translation_source = (
                            message.get("translation_source") or None
                        )
                    if has_tts:
                        session.tts_enabled = bool(
                            message.get("tts_enabled", bool(self.tts_service))
                        ) and bool(self.tts_service)
                        session.tts_voice = (
                            message.get("tts_voice")
                            or self.tts_default_voice
                            or None
                        )
                        requested_tts_source = str(
                            message.get("tts_source") or "translation"
                        ).strip()
                        session.tts_source = (
                            requested_tts_source
                            if requested_tts_source
                            in {"transcript", "translation"}
                            else "translation"
                        )
                    if has_language or has_context:
                        session.pending_audio.clear()
                        session.committed_text = ""
                        session.segment_bytes = 0
                        session.segment_has_speech = False
                        session.vad_buffer.clear()
                        session.vad_active = False
                        session.pre_roll.clear()
                        if session.vad:
                            session.vad.reset()
                        session.preprocessor = AudioPreprocessor(
                            highpass_hz=self.highpass_hz,
                            gain_db=self.audio_gain_db,
                        )
                        session.segment_id = 0
                        session.last_segment_text = ""
                        session.state = self.new_state(
                            session.language, session.context
                        )
                elif event_type == "input_audio_buffer.append":
                    encoded = message.get("audio")
                    if not encoded:
                        continue
                    await self.append_audio(
                        websocket, session, base64.b64decode(encoded)
                    )
                elif event_type == "input_audio_buffer.commit":
                    if message.get("final"):
                        await self.finish(websocket, session)
                else:
                    raise ValueError(f"Event khong duoc ho tro: {event_type}")
        except websockets.ConnectionClosed:
            logging.info("Client da ngat ket noi")
        except Exception as exc:
            logging.exception("Loi trong phien realtime")
            try:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "error",
                            "error": {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        },
                        ensure_ascii=False,
                    )
                )
            except websockets.ConnectionClosed:
                pass
        finally:
            if session:
                for task in session.translation_tasks | session.tts_tasks:
                    task.cancel()


async def run_server(args) -> None:
    server = QwenRealtimeServer(
        model=args.model,
        language=args.language,
        context=args.context,
        gpu_memory_utilization=args.gpu_memory_utilization,
        step_ms=args.step_ms,
        chunk_size_sec=args.chunk_size_sec,
        max_segment_sec=args.max_segment_sec,
        max_model_len=args.max_model_len,
        vad_enabled=args.vad,
        vad_threshold=args.vad_threshold,
        vad_min_silence_ms=args.vad_min_silence_ms,
        vad_speech_pad_ms=args.vad_speech_pad_ms,
        vad_pre_roll_ms=args.vad_pre_roll_ms,
        highpass_hz=args.highpass_hz,
        audio_gain_db=args.audio_gain_db,
        translation_enabled=args.translation,
        translation_model=args.translation_model,
        translation_device=args.translation_device,
        tts_enabled=args.tts,
        tts_mode=args.tts_mode,
        tts_voice=args.tts_voice,
    )
    try:
        if server.translation_service:
            logging.info("Dang tai model dich tren luong CPU rieng")
            await server.translation_service.start()

        async with websockets.serve(
            server.handle,
            args.host,
            args.port,
            max_size=None,
            compression=None,
            ping_interval=20,
            ping_timeout=None,
        ):
            logging.info(
                "Qwen realtime server: ws://%s:%s/v1/realtime",
                args.host,
                args.port,
            )
            await asyncio.Future()
    finally:
        server.asr_executor.shutdown(wait=True, cancel_futures=True)
        if server.translation_service:
            await server.translation_service.stop()
        if server.tts_service:
            await server.tts_service.stop()


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-ASR realtime WebSocket server")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--language", default=None)
    parser.add_argument("--context", default="")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--step-ms", type=int, default=200)
    parser.add_argument("--chunk-size-sec", type=float, default=1.0)
    parser.add_argument("--max-segment-sec", type=float, default=15.0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--vad-min-silence-ms", type=int, default=900)
    parser.add_argument("--vad-speech-pad-ms", type=int, default=250)
    parser.add_argument("--vad-pre-roll-ms", type=int, default=300)
    parser.add_argument("--highpass-hz", type=float, default=40.0)
    parser.add_argument("--audio-gain-db", type=float, default=0.0)
    parser.add_argument(
        "--translation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--translation-model",
        default="VietAI/envit5-translation",
    )
    parser.add_argument(
        "--translation-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--tts",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--tts-mode", default="v3turbo")
    parser.add_argument("--tts-voice", default=DEFAULT_TTS_VOICE)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    asyncio.run(run_server(parse_args()))


if __name__ == "__main__":
    main()
