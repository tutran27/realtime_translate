import json
from typing import Any

import aiohttp
import modal
whisper_live_kit_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "soxr",
        "librosa",
        "soundfile",
    )
    .run_commands(
        "python -m pip install -U pip setuptools wheel",
        "python -m pip install -U --pre vllm --extra-index-url https://wheels.vllm.ai/nightly",
        "python -m pip install git+https://github.com/huggingface/transformers.git",
    )
)

app = modal.App("voxtral-mini-realtime-serve")

N_GPU = 1
MINUTES = 60  # seconds
VOXTRAL_MINI_REALTIME_4B_PORT=8000
volumes = {
    "/mnt/VOICE": modal.Volume.from_name(name="VOICE", create_if_missing=True)
}

@app.function(
    image=whisper_live_kit_image,
    # gpu=f"H100:{N_GPU}",
    # gpu=f"A10:{N_GPU}",
    gpu=f"L40S:{N_GPU}",
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes=volumes
)


# Due to size and the BF16 format of the weights - Voxtral-Mini-4B-Realtime-2602 can run on a single GPU with >= 16GB memory.

# The model can be launched in both "eager" mode:

# VLLM_DISABLE_COMPILE_CACHE=1 vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --compilation_config '{"cudagraph_mode": "PIECEWISE"}'

# Additional flags:

# You can set --max-num-batched-tokens to balance throughput and latency, higher means higher throughput but higher latency.
# You can reduce the default --max-model-len to allocate less memory for the pre-computed RoPE frequencies, if you are certain that you won't have to transcribe for more than X hours. By default the model uses a --max-model-len of 131072 (> 3h).

@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
@modal.web_server(port=VOXTRAL_MINI_REALTIME_4B_PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    subprocess.Popen([
        "vllm",
        "serve",
        "/mnt/VOICE/models/Voxtral-Mini-4B-Realtime-2602_delay480",
        # "/mnt/VOICE/models/Voxtral-Mini-4B-Realtime-2602-delay2400",
        "--compilation-config",
        '{"cudagraph_mode":"PIECEWISE"}',
        "--gpu-memory-utilization",
        "0.9",
    ])

# def main():
#     import os
#     # Use the multi-GPU script
#     os.system("python WhisperLiveKit/test_debug_logging.py")
