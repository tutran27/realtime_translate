"""
Streamlit demo thu am micro va gui lien tuc toi vLLM Realtime STT.

Cai dat:
    pip install streamlit streamlit-webrtc websockets av numpy

Chay:
    streamlit run streamlit_realtime_stt.py
"""

import asyncio
import base64
import json
import queue
import threading
from dataclasses import dataclass, field

import av
import numpy as np
import streamlit as st
import websockets
from streamlit_webrtc import AudioProcessorBase, WebRtcMode, webrtc_streamer


SAMPLE_RATE = 16_000
STOP_SIGNAL = object()


def build_realtime_uri(host: str, port: int) -> str:
    host = host.strip()
    if host.startswith(("http://", "https://", "ws://", "wss://")):
        base = host.rstrip("/")
        base = base.replace("https://", "wss://").replace("http://", "ws://")
        return f"{base}/v1/realtime"
    return f"ws://{host}:{port}/v1/realtime"


@dataclass
class RealtimeSTT:
    host: str
    port: int
    model: str
    audio_queue: queue.Queue = field(
        default_factory=lambda: queue.Queue(maxsize=200)
    )
    transcript_parts: list[str] = field(default_factory=list)
    status: str = "Chua ket noi"
    error: str = ""
    running: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    thread: threading.Thread | None = None

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.error = ""
        self.status = "Dang ket noi..."
        self.thread = threading.Thread(target=self._thread_main, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self.status = "Dang dung..."
        self._put(STOP_SIGNAL)

    def clear_transcript(self) -> None:
        with self.lock:
            self.transcript_parts.clear()

    def get_transcript(self) -> str:
        with self.lock:
            return "".join(self.transcript_parts)

    def add_audio(self, pcm16_bytes: bytes) -> None:
        if self.running and pcm16_bytes:
            self._put(pcm16_bytes)

    def _put(self, item: object) -> None:
        try:
            self.audio_queue.put_nowait(item)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(item)
            except queue.Full:
                pass

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._websocket_main())
        except Exception as exc:
            self.error = str(exc)
            self.status = "Loi ket noi"
        finally:
            self.running = False

    async def _websocket_main(self) -> None:
        uri = build_realtime_uri(self.host, self.port)
        async with websockets.connect(uri, max_size=None) as ws:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            if event.get("type") != "session.created":
                raise RuntimeError(f"Phan hoi khoi tao khong hop le: {event}")

            await ws.send(
                json.dumps({"type": "session.update", "model": self.model})
            )
            await ws.send(
                json.dumps(
                    {"type": "input_audio_buffer.commit", "final": False}
                )
            )
            self.status = "Da ket noi, dang nhan audio"

            sender = asyncio.create_task(self._send_audio(ws))
            receiver = asyncio.create_task(self._receive_transcript(ws))

            await sender
            try:
                await asyncio.wait_for(receiver, timeout=10)
            except asyncio.TimeoutError:
                receiver.cancel()
            self.status = "Da dung"

    async def _send_audio(self, ws) -> None:
        while True:
            try:
                item = await asyncio.to_thread(
                    self.audio_queue.get, True, 0.2
                )
            except queue.Empty:
                if self.running:
                    continue
                item = STOP_SIGNAL

            if item is STOP_SIGNAL:
                await ws.send(
                    json.dumps(
                        {"type": "input_audio_buffer.commit", "final": True}
                    )
                )
                return

            encoded = base64.b64encode(item).decode("ascii")
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": encoded,
                    }
                )
            )

    async def _receive_transcript(self, ws) -> None:
        async for raw_message in ws:
            event = json.loads(raw_message)
            event_type = event.get("type")

            if event_type == "transcription.delta":
                delta = event.get("delta", "")
                if delta:
                    with self.lock:
                        self.transcript_parts.append(delta)
            elif event_type == "transcription.done":
                final_text = event.get("text")
                if final_text and not self.get_transcript():
                    with self.lock:
                        self.transcript_parts.append(final_text)
                return
            elif event_type == "error":
                error = event.get("error", event)
                raise RuntimeError(f"Realtime STT tra ve loi: {error}")


class MicrophoneAudioProcessor(AudioProcessorBase):
    def __init__(self, runtime: RealtimeSTT):
        self.runtime = runtime
        self.resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=SAMPLE_RATE,
        )

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        for resampled_frame in self.resampler.resample(frame):
            samples = resampled_frame.to_ndarray()
            pcm16 = np.asarray(samples, dtype="<i2").reshape(-1).tobytes()
            self.runtime.add_audio(pcm16)
        return frame


st.set_page_config(page_title="Realtime STT", page_icon="🎙️")
st.title("Realtime Speech-to-Text")

with st.sidebar:
    st.header("Cau hinh")
    host = st.text_input("Host", value="localhost")
    port = st.number_input(
        "Port", min_value=1, max_value=65535, value=8000
    )
    model = st.text_input(
        "Model",
        value="mistralai/Voxtral-Mini-4B-Realtime-2602",
    )

config = (host.strip(), int(port), model.strip())
runtime: RealtimeSTT | None = st.session_state.get("stt_runtime")
runtime_config = st.session_state.get("stt_runtime_config")

if runtime is None or (runtime_config != config and not runtime.running):
    runtime = RealtimeSTT(host=config[0], port=config[1], model=config[2])
    st.session_state.stt_runtime = runtime
    st.session_state.stt_runtime_config = config

left, middle, right = st.columns(3)
with left:
    if st.button(
        "Ket noi STT",
        type="primary",
        disabled=runtime.running,
        use_container_width=True,
    ):
        runtime.start()
with middle:
    if st.button(
        "Dung STT",
        disabled=not runtime.running,
        use_container_width=True,
    ):
        runtime.stop()
with right:
    if st.button("Xoa transcript", use_container_width=True):
        runtime.clear_transcript()

st.caption(
    "Nhan START ben duoi va cho phep trinh duyet truy cap micro. "
    "Sau do nhan Ket noi STT de bat dau gui audio."
)

webrtc_streamer(
    key="realtime-stt-microphone",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=lambda: MicrophoneAudioProcessor(runtime),
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True,
)


@st.fragment(run_every=0.3)
def render_live_result() -> None:
    if runtime.error:
        st.error(runtime.error)
    elif runtime.running:
        st.success(runtime.status)
    else:
        st.info(runtime.status)

    st.subheader("Transcript")
    transcript = runtime.get_transcript()
    st.text_area(
        "Ket qua nhan dang",
        value=transcript,
        height=260,
        disabled=True,
        label_visibility="collapsed",
    )


render_live_result()
