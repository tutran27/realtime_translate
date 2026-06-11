"""
Test vLLM Realtime STT bằng cách stream audio từ FILE (không cần micro).

Cài đặt tối thiểu:
    pip install websockets librosa soundfile numpy

Chạy ví dụ:
    python realtime_test_from_audio_file.py --audio_path audio/shoes.mp3 --host localhost --port 8000
"""

import argparse
import asyncio
import base64
import json
from dataclasses import dataclass

import numpy as np
import websockets


SAMPLE_RATE = 16_000


def load_pcm16_mono(audio_path: str, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Load audio bất kỳ (mp3/wav/...) và convert sang PCM16 mono @16kHz."""
    try:
        import librosa
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Thiếu `librosa`. Cài: pip install librosa soundfile"
        ) from exc

    audio, _sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype("<i2")
    return pcm16.tobytes()


def build_realtime_uri(host: str, port: int) -> str:
    host = host.strip()
    if host.startswith(("http://", "https://", "ws://", "wss://")):
        base = host.rstrip("/")
        base = base.replace("https://", "wss://").replace("http://", "ws://")
        return f"{base}/v1/realtime"
    return f"ws://{host}:{port}/v1/realtime"


@dataclass
class StreamConfig:
    host: str
    port: int
    model: str
    chunk_ms: int
    speed: float


async def recv_transcript(ws) -> str:
    """Nhận transcript và in realtime ra console."""
    collected: list[str] = []
    async for raw in ws:
        event = json.loads(raw)
        t = event.get("type")

        if t == "transcription.delta":
            delta = event.get("delta", "")
            if delta:
                print(delta, end="", flush=True)
                collected.append(delta)
        elif t == "transcription.done":
            text = event.get("text")
            if text and not collected:
                print(text, end="", flush=True)
                return text
            return "".join(collected)
        elif t == "error":
            raise RuntimeError(f"Realtime STT trả lỗi: {event.get('error', event)}")

    return "".join(collected)


async def send_audio(ws, pcm16: bytes, cfg: StreamConfig) -> None:
    samples_per_chunk = int(SAMPLE_RATE * (cfg.chunk_ms / 1000.0))
    bytes_per_chunk = samples_per_chunk * 2  # int16

    if bytes_per_chunk <= 0:
        raise ValueError("--chunk_ms phải > 0.")
    if cfg.speed <= 0:
        raise ValueError("--speed phải > 0.")

    # Gửi audio theo nhịp "gần realtime"
    chunk_duration = (samples_per_chunk / SAMPLE_RATE) / cfg.speed
    for i in range(0, len(pcm16), bytes_per_chunk):
        chunk = pcm16[i : i + bytes_per_chunk]
        if not chunk:
            continue

        encoded = base64.b64encode(chunk).decode("ascii")
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": encoded}))
        await asyncio.sleep(chunk_duration)

    await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))


async def main_async(args) -> int:
    cfg = StreamConfig(
        host=args.host,
        port=int(args.port),
        model=args.model,
        chunk_ms=int(args.chunk_ms),
        speed=float(args.speed),
    )
    uri = build_realtime_uri(cfg.host, cfg.port)
    pcm16 = load_pcm16_mono(args.audio_path)

    print(f"Connecting: {uri}", flush=True)
    try:
        ws_cm = websockets.connect(uri, max_size=None)
        async with ws_cm as ws:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            if event.get("type") != "session.created":
                raise RuntimeError(f"Phản hồi khởi tạo không hợp lệ: {event}")

            await ws.send(json.dumps({"type": "session.update", "model": cfg.model}))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": False}))

            print("Transcription: ", end="", flush=True)
            receiver = asyncio.create_task(recv_transcript(ws))
            sender = asyncio.create_task(send_audio(ws, pcm16, cfg))

            await sender
            final_text = await receiver
    except (ConnectionRefusedError, OSError) as exc:
        raise RuntimeError(
            "Không kết nối được tới server realtime.\n"
            f"- URI: {uri}\n"
            "- Nguyên nhân thường gặp: server vLLM chưa chạy hoặc sai host/port.\n"
            "- Nếu bạn chạy local: hãy đảm bảo có service đang listen port đó (vd 8000).\n"
            "- Nếu bạn chạy Modal/remote: truyền `--host` là URL (http(s)://...) để script tự đổi sang ws/wss.\n"
        ) from exc

    print("\n\nFinal transcription:", final_text)
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Stream audio file -> vLLM Realtime STT (/v1/realtime)")
    p.add_argument("--audio_path", required=True, help="Đường dẫn file audio (mp3/wav/...).")
    p.add_argument("--host", default="localhost", help="Host (vd: localhost hoặc http(s)://...).")
    p.add_argument("--port", default=8000, type=int, help="Port server realtime.")
    p.add_argument(
        "--model",
        default="mistralai/Voxtral-Mini-4B-Realtime-2602",
        help="Model name giống server đang serve.",
    )
    p.add_argument(
        "--chunk_ms",
        default=100,
        type=int,
        help="Kích thước chunk audio gửi lên (ms). 50-200ms thường ổn.",
    )
    p.add_argument(
        "--speed",
        default=1.0,
        type=float,
        help="Tốc độ stream so với realtime. 1.0 = realtime, 2.0 = nhanh gấp đôi.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
