import argparse
from pathlib import Path

from qwen_asr import Qwen3ASRModel


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-ASR offline transcription")
    parser.add_argument("audio_path", nargs="?", default="audio/shoes.mp3")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--language", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Khong tim thay file audio: {audio_path}")

    model = Qwen3ASRModel.LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_new_tokens=256,
    )
    result = model.transcribe(
        audio=str(audio_path),
        language=args.language,
    )[0]
    print(f"Language: {result.language}")
    print(result.text)


if __name__ == "__main__":
    main()
