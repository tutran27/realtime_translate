import argparse
import asyncio
import base64
import json
import logging
import uuid
from dataclasses import dataclass, field

import numpy as np
import websockets
from qwen_asr import Qwen3ASRModel


SAMPLE_RATE = 16_000
DEFAULT_MODEL = "Qwen/Qwen3-ASR-0.6B"


@dataclass
class Session:
    model: str
    state: object
    pending_audio: bytearray = field(default_factory=bytearray)
    last_text: str = ""


class QwenRealtimeServer:
    def __init__(
        self,
        model: str,
        gpu_memory_utilization: float,
        step_ms: int,
        chunk_size_sec: float,
    ) -> None:
        self.model_name = model
        self.step_bytes = int(SAMPLE_RATE * step_ms / 1000) * 2
        self.chunk_size_sec = chunk_size_sec
        self.inference_lock = asyncio.Lock()

        logging.info("Dang tai model %s", model)
        self.asr = Qwen3ASRModel.LLM(
            model=model,
            gpu_memory_utilization=gpu_memory_utilization,
            max_new_tokens=32,
        )
        logging.info("Model da san sang")

    def new_session(self) -> Session:
        state = self.asr.init_streaming_state(
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=self.chunk_size_sec,
        )
        return Session(model=self.model_name, state=state)

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
                await asyncio.to_thread(
                    self.asr.streaming_transcribe, audio, session.state
                )

            await self.send_update(websocket, session)

    async def send_update(self, websocket, session: Session) -> None:
        text = str(getattr(session.state, "text", "") or "")
        language = getattr(session.state, "language", None)
        if text == session.last_text:
            return
        session.last_text = text
        await websocket.send(
            json.dumps(
                {
                    "type": "transcription.updated",
                    "text": text,
                    "language": language,
                },
                ensure_ascii=False,
            )
        )

    async def finish(self, websocket, session: Session) -> None:
        await self.transcribe_pending(websocket, session, flush=True)
        async with self.inference_lock:
            await asyncio.to_thread(
                self.asr.finish_streaming_transcribe, session.state
            )
        await self.send_update(websocket, session)
        await websocket.send(
            json.dumps(
                {
                    "type": "transcription.done",
                    "text": str(getattr(session.state, "text", "") or ""),
                    "language": getattr(session.state, "language", None),
                },
                ensure_ascii=False,
            )
        )

    async def handle(self, websocket) -> None:
        request_path = getattr(websocket, "path", None)
        if request_path is None:
            request_path = getattr(
                getattr(websocket, "request", None), "path", "/"
            )
        if request_path != "/v1/realtime":
            await websocket.close(code=1008, reason="Endpoint khong hop le")
            return

        session = self.new_session()
        await websocket.send(
            json.dumps(
                {
                    "type": "session.created",
                    "id": f"sess_{uuid.uuid4().hex}",
                    "model": self.model_name,
                }
            )
        )

        try:
            async for raw_message in websocket:
                message = json.loads(raw_message)
                event_type = message.get("type")

                if event_type == "session.update":
                    requested_model = message.get("model")
                    if requested_model and requested_model != self.model_name:
                        raise ValueError(
                            f"Server dang phuc vu {self.model_name}, "
                            f"khong phai {requested_model}"
                        )
                elif event_type == "input_audio_buffer.append":
                    encoded = message.get("audio")
                    if not encoded:
                        continue
                    session.pending_audio.extend(base64.b64decode(encoded))
                    await self.transcribe_pending(websocket, session)
                elif event_type == "input_audio_buffer.commit":
                    if message.get("final"):
                        await self.finish(websocket, session)
                        return
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


async def run_server(args) -> None:
    server = QwenRealtimeServer(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        step_ms=args.step_ms,
        chunk_size_sec=args.chunk_size_sec,
    )
    async with websockets.serve(
        server.handle,
        args.host,
        args.port,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
    ):
        logging.info("Qwen realtime server: ws://%s:%s/v1/realtime", args.host, args.port)
        await asyncio.Future()


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-ASR realtime WebSocket server")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--step-ms", type=int, default=500)
    parser.add_argument("--chunk-size-sec", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    asyncio.run(run_server(parse_args()))


if __name__ == "__main__":
    main()
