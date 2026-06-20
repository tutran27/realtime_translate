import argparse
import re
import time
import unicodedata
from pathlib import Path

from vieneu import Vieneu


DEFAULT_TEXT = (
    "[cười] Đây là bài test tổng hợp để nghe thử tất cả giọng nói có sẵn "
    "trong VieNeu. Nếu bạn đang nghe đoạn này thì voice này đã tổng hợp "
    "thành công."
)


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value).strip("_").lower()
    return slug or "voice"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test tat ca preset voice cua Vieneu.")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "samples"),
    )
    parser.add_argument("--mode", default="v3turbo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tts = Vieneu(mode=args.mode)
    try:
        voices = list(tts.list_preset_voices())
        print(f"Found {len(voices)} preset voices")

        success_count = 0
        for index, (label, voice_id) in enumerate(voices, start=1):
            started_at = time.time()
            output_path = output_dir / f"{index:02d}_{slugify(voice_id)}.wav"
            try:
                audio = tts.infer(
                    args.text,
                    voice=voice_id,
                    apply_watermark=False,
                )
                tts.save(audio, str(output_path))
                elapsed = time.time() - started_at
                success_count += 1
                print(
                    f"[OK] {index:02d}/{len(voices)} label={label} "
                    f"voice_id={voice_id} file={output_path} time={elapsed:.2f}s"
                )
            except Exception as exc:
                print(
                    f"[FAIL] {index:02d}/{len(voices)} label={label} "
                    f"voice_id={voice_id} error={exc}"
                )

        print(f"Done: {success_count}/{len(voices)} voices synthesized")
    finally:
        tts.close()


if __name__ == "__main__":
    main()
