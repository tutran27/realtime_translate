# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
This script uses the local Qwen3-ASR Realtime WebSocket service to transcribe
one audio file or a batch JSON file.

Requirements:
- websockets
- librosa
- numpy

The script:
1. Connects to the Realtime WebSocket endpoint
2. Converts an audio file to PCM16 @ 16kHz
3. Sends audio chunks to the server
4. Receives and prints transcription as it streams
"""

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import base64
import librosa
import websockets

def audio_to_pcm16_bytes(audio_path: str) -> bytes:
    """
    Load an audio file and convert it to PCM16 @ 16kHz bytes.
    """
    # Load audio and resample to 16kHz mono
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    # Convert to PCM16
    pcm16 = (audio * 32767).astype(np.int16)
    return pcm16.tobytes()


def build_realtime_uri(host: str, port: int) -> str:
    """
    Build the Realtime WebSocket URI from either a hostname or a full base URL.
    """
    if host:
        if host.startswith(("http://", "https://", "ws://", "wss://")):
            base = host.rstrip("/")
            base = base.replace("https://", "wss://").replace("http://", "ws://")
            return f"{base}/v1/realtime"
        return f"ws://{host}:{port}/v1/realtime"
    return f"ws://localhost:{port}/v1/realtime"


async def realtime_transcribe_pcm16(
    audio_bytes: bytes, host: str, port: int, model: str
):
    """
    Connect to the Realtime API and transcribe PCM16 audio bytes.
    """
    uri = build_realtime_uri(host, port)
    current_text = ""

    async with websockets.connect(uri) as ws:
        # Wait for session.created
        response = json.loads(await ws.recv())
        if response["type"] == "session.created":
            print(f"Session created: {response['id']}")
        else:
            print(f"Unexpected response: {response}")
            raise RuntimeError(f"Unexpected initial response: {response}")

        # Validate model
        await ws.send(json.dumps({"type": "session.update", "model": model}))

        # Signal ready to start
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Send audio in chunks (4KB of raw audio = ~8KB base64)
        chunk_size = 4096
        total_chunks = (len(audio_bytes) + chunk_size - 1) // chunk_size

        print(f"Sending {total_chunks} audio chunks...")
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i : i + chunk_size]
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("utf-8"),
                    }
                )
            )

        # Signal all audio is sent
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "final": True}))
        print("Audio sent. Waiting for transcription...\n")

        # Receive transcription
        print("Transcription: ", end="", flush=True)
        while True:
            response = json.loads(await ws.recv())
            # print(response)
            if response["type"] == "transcription.updated":
                current_text = response.get("text", "")
                print(f"\rTranscription: {current_text}", end="", flush=True)
            elif response["type"] == "transcription.done":
                final_text = response.get("text") or current_text
                print(f"\n\nFinal transcription: {final_text}")
                if response.get("usage"):
                    print(f"Usage: {response['usage']}")
                return final_text
            elif response["type"] == "error":
                error = response.get("error", response)
                print(f"\nError: {error}")
                raise RuntimeError(f"Realtime transcription failed: {error}")


async def realtime_transcribe(audio_path: str, host: str, port: int, model: str):
    """
    Load an audio file, then connect to the Realtime API and transcribe it.
    """
    print(f"Loading audio from: {audio_path}")
    audio_bytes = audio_to_pcm16_bytes(audio_path)
    return await realtime_transcribe_pcm16(audio_bytes, host, port, model)


async def realtime_transcribe_n_times(
    audio_path: str,
    host: str,
    port: int,
    model: str,
    num_runs: int,
    *,
    continue_on_error: bool,
) -> list[str]:
    """
    Reuse the same prepared audio and run realtime transcription multiple times.
    """
    if num_runs < 1:
        raise ValueError("--num_runs must be greater than or equal to 1.")

    print(f"Loading audio from: {audio_path}")
    audio_bytes = audio_to_pcm16_bytes(audio_path)
    responses = []

    for run_index in range(1, num_runs + 1):
        print(f"\nRun {run_index}/{num_runs}")
        try:
            transcription = await realtime_transcribe_pcm16(
                audio_bytes, host, port, model
            )
            responses.append(transcription)
        except Exception as exc:
            if not continue_on_error:
                raise
            print(
                "Skipping failed run "
                f"{run_index}/{num_runs} for audio: {audio_path}. Error: {exc}"
            )

    return responses


def load_batch_requests(input_json_path: str) -> tuple[list[dict], Path]:
    """
    Load the batch JSON file and return its items plus the input directory.
    """
    input_path = Path(input_json_path).expanduser().resolve()
    with input_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of objects.")

    return data, input_path.parent


def resolve_audio_path(audio_path: str, base_dir: Path) -> str:
    """
    Resolve a possibly-relative audio path.
    Relative paths are resolved against the input JSON directory first.
    """
    candidate = Path(audio_path).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    json_relative_candidate = (base_dir / candidate).resolve()
    if json_relative_candidate.exists():
        return str(json_relative_candidate)

    return str(candidate.resolve())


def resolve_output_json_path(output_json: str | None, input_json: str) -> Path:
    """
    Determine where to write batch transcription results.
    """
    if output_json:
        return Path(output_json).expanduser().resolve()

    input_path = Path(input_json).expanduser().resolve()
    return input_path.with_name(f"{input_path.stem}_asr_results.json")


async def realtime_transcribe_batch(
    items: list[dict],
    input_base_dir: Path,
    host: str,
    port: int,
    model: str,
    num_runs: int,
) -> list[dict]:
    """
    Transcribe each audio entry from the input JSON and return output JSON rows.
    """
    results = []
    total_items = len(items)

    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {index - 1} is not a JSON object.")

        name = item.get("name", f"item_{index}")
        ground_truth = item.get("ground_truth", "")
        raw_audio_path = item.get("audio_path")

        if not raw_audio_path:
            raise ValueError(f"Item '{name}' is missing 'audio_path'.")

        audio_path = resolve_audio_path(str(raw_audio_path), input_base_dir)

        print(f"\n[{index}/{total_items}] Processing: {name}")
        print(f"Resolved audio path: {audio_path}")

        try:
            asr_responses = await realtime_transcribe_n_times(
                audio_path=audio_path,
                host=host,
                port=port,
                model=model,
                num_runs=num_runs,
                continue_on_error=True,
            )
        except Exception as exc:
            print(f"Failed to transcribe '{name}': {exc}")
            asr_responses = []

        results.append(
            {
                "name": name,
                "ground_truth": ground_truth,
                "asr_responses": asr_responses,
            }
        )

    return results


def write_json_output(output_path: Path, payload: list[dict]) -> None:
    """
    Persist JSON output with UTF-8 and readable formatting.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON output to: {output_path}")


def main(args):
    if args.input_json:
        items, input_base_dir = load_batch_requests(args.input_json)
        output_path = resolve_output_json_path(args.output_json, args.input_json)
        results = asyncio.run(
            realtime_transcribe_batch(
                items=items,
                input_base_dir=input_base_dir,
                host=args.host,
                port=args.port,
                model=args.model,
                num_runs=args.num_runs,
            )
        )
        write_json_output(output_path, results)
        return

    if args.audio_path:
        audio_path = args.audio_path
    else:
        raise ValueError("Phai truyen --audio_path hoac --input_json.")

    transcriptions = asyncio.run(
        realtime_transcribe_n_times(
            audio_path=audio_path,
            host=args.host,
            port=args.port,
            model=args.model,
            num_runs=args.num_runs,
            continue_on_error=False,
        )
    )

    if args.output_json:
        write_json_output(
            Path(args.output_json).expanduser().resolve(),
            [
                {
                    "name": Path(audio_path).stem,
                    "ground_truth": "",
                    "asr_responses": transcriptions,
                }
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR Realtime WebSocket Transcription Client"
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to a single audio file to transcribe.",
    )
    input_group.add_argument(
        "--input_json",
        type=str,
        default=None,
        help=(
            "Path to a JSON file containing a list of items with "
            "'name', 'ground_truth', and 'audio_path'."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-ASR-0.6B",
        help="Model that is served and should be pinged.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Optional output JSON path. In batch mode, defaults to "
            "<input_json_stem>_asr_results.json."
        ),
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="vLLM server host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default: 8000)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of transcription runs to execute for each audio (default: 1)",
    )
    args = parser.parse_args()
    main(args)
