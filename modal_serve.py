import modal


qwen_asr_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .uv_pip_install(
        "qwen-asr[vllm]",
        "websockets",
    )
    .add_local_file(
        "qwen_realtime_server.py",
        "/root/qwen_realtime_server.py",
    )
)

app = modal.App("qwen3-asr-realtime-serve")

N_GPU = 1
MINUTES = 60  # seconds
PORT = 8000

@app.function(
    image=qwen_asr_image,
    gpu=f"L4:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=PORT, startup_timeout=10 * MINUTES)
def serve():
    import subprocess

    subprocess.Popen(
        [
            "python",
            "/root/qwen_realtime_server.py",
            "--model",
            "Qwen/Qwen3-ASR-0.6B",
            "--port",
            str(PORT),
            "--gpu-memory-utilization",
            "0.8",
        ]
    )
