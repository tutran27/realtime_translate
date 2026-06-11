# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Nemotron realtime (streaming-like) transcription from an audio file.
"""

import argparse
import re
import tempfile
from functools import lru_cache

import numpy as np

DEFAULT_NEMO_MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"


def load_audio_array(audio_path: str, sample_rate: int = 16000) -> tuple[np.ndarray, int]:
    try:
        import librosa
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError("Thieu thu vien `librosa`. Cai: pip install librosa soundfile") from e

    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32), sample_rate


def normalize_token(token: str) -> str:
    return re.sub(r"\W+", "", token).lower()


def merge_transcript(existing_text: str, new_text: str, max_overlap_words: int = 20) -> tuple[str, str]:
    existing_words = existing_text.split()
    new_words = new_text.split()
    overlap = 0

    max_size = min(max_overlap_words, len(existing_words), len(new_words))
    for size in range(max_size, 0, -1):
        if [normalize_token(w) for w in existing_words[-size:]] == [normalize_token(w) for w in new_words[:size]]:
            overlap = size
            break

    delta_words = new_words[overlap:]
    merged_words = existing_words + delta_words
    return " ".join(merged_words).strip(), " ".join(delta_words).strip()


def strip_lang_tags(text: str) -> str:
    return re.sub(r"\s*<[a-z]{2}-[A-Z]{2}>", "", text).strip()


def maybe_set_nemo_target_lang(asr_model, target_lang: str | None) -> None:
    if not target_lang:
        return
    for attr in ("set_target_lang", "set_language", "set_lang"):
        fn = getattr(asr_model, attr, None)
        if callable(fn):
            try:
                fn(target_lang)
                return
            except Exception:
                pass
    for key in ("target_lang", "lang", "language"):
        if hasattr(asr_model, key):
            try:
                setattr(asr_model, key, target_lang)
                return
            except Exception:
                pass


@lru_cache(maxsize=2)
def build_nemo_asr_model(model_id: str):
    try:
        import importlib

        importlib.import_module("nemo.collections.asr.models.rnnt_bpe_models_prompt")
    except ModuleNotFoundError as e:
        # Trường hợp phổ biến: chưa cài NeMo nên thiếu hẳn module `nemo`.
        if getattr(e, "name", "") == "nemo":
            raise ModuleNotFoundError(
                "Ban chua cai NeMo (`nemo_toolkit`) trong dung python env.\n"
                "Cach cai nhanh:\n"
                "  python -m pip install -U pip\n"
                "  python -m pip install \"setuptools<82\"\n"
                "  python -m pip install \"nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main\"\n"
            ) from e
        raise ModuleNotFoundError(
            "NeMo ban dang dung thieu module `nemo.collections.asr.models.rnnt_bpe_models_prompt`.\n"
            "Model Nemotron 3.5 ASR can NeMo moi.\n"
            "Cach fix (trong dung env dang chay python):\n"
            "  pip uninstall -y nemo_toolkit nemo-toolkit nemo\n"
            "  pip install -U pip setuptools wheel\n"
            "  pip install \"nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main\"\n"
        ) from e

    import nemo.collections.asr as nemo_asr

    print(f"Loading NeMo model: {model_id}", flush=True)
    candidates = [
        ("ASRModel", "ASRModel"),
        ("EncDecRNNTBPEModelWithPrompt", "EncDecRNNTBPEModelWithPrompt"),
        ("EncDecRNNTModelWithPrompt", "EncDecRNNTModelWithPrompt"),
        ("EncDecRNNTBPEModel", "EncDecRNNTBPEModel"),
        ("EncDecRNNTModel", "EncDecRNNTModel"),
        ("EncDecHybridRNNTCTCModel", "EncDecHybridRNNTCTCModel"),
        ("EncDecCTCModel", "EncDecCTCModel"),
    ]

    last_error: Exception | None = None
    for label, class_name in candidates:
        cls = getattr(nemo_asr.models, class_name, None)
        if cls is None or not hasattr(cls, "from_pretrained"):
            continue
        try:
            return cls.from_pretrained(model_name=model_id)
        except TypeError as exc:
            msg = str(exc)
            if "abstract class" in msg or "abstract methods" in msg:
                last_error = exc
                continue
            raise
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "Khong load duoc model NeMo. Thu cai NeMo ban moi nhat (git main) theo model card: "
        "pip install \"nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main\""
    ) from last_error


def nemotron_transcribe_file(
    audio_path: str,
    model_id: str,
    target_lang: str,
    chunk_seconds: float,
    overlap_seconds: float,
    *,
    remove_lang_tags: bool,
) -> str:
    asr_model = build_nemo_asr_model(model_id)
    maybe_set_nemo_target_lang(asr_model, target_lang)

    audio, sample_rate = load_audio_array(audio_path)
    chunk_samples = int(chunk_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)

    if chunk_samples <= 0:
        raise ValueError("--chunk_seconds phai > 0.")
    if overlap_samples < 0 or overlap_samples >= chunk_samples:
        raise ValueError("--overlap_seconds phai >= 0 va nho hon chunk_seconds.")

    step_samples = chunk_samples - overlap_samples
    transcript = ""

    print(f"Processing file: {audio_path}", flush=True)
    print("Transcription: ", end="", flush=True)

    def transcribe_chunk(chunk_audio: np.ndarray) -> str:
        """
        NeMo API thay đổi theo version/model:
        - Một số model hỗ trợ transcribe(audio=..., ...)
        - Nhiều model chỉ hỗ trợ transcribe(paths2audio_files=[...], ...)
        Fallback an toàn: ghi chunk ra WAV tạm rồi transcribe theo path.
        """
        try:
            outputs = asr_model.transcribe(
                audio=chunk_audio,
                batch_size=1,
                return_hypotheses=False,
                verbose=False,
            )
            chunk_text = outputs[0] if isinstance(outputs, list) and outputs else outputs
            return str(chunk_text).strip()
        except TypeError:
            pass

        try:
            import soundfile as sf
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "NeMo model khong ho tro transcribe(audio=...). Can `soundfile` de ghi WAV tam.\n"
                "Cai: pip install soundfile"
            ) from exc

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as fp:
            sf.write(fp.name, chunk_audio, samplerate=sample_rate)
            # Thu cac signature pho bien cua NeMo
            for kwargs in (
                {"paths2audio_files": [fp.name]},
                {"audio_files": [fp.name]},
            ):
                try:
                    outputs = asr_model.transcribe(
                        **kwargs,
                        batch_size=1,
                        return_hypotheses=False,
                        verbose=False,
                    )
                    chunk_text = outputs[0] if isinstance(outputs, list) and outputs else outputs
                    return str(chunk_text).strip()
                except TypeError:
                    continue

        raise RuntimeError(
            "Khong goi duoc asr_model.transcribe cho chunk audio. "
            "Hay paste traceback + version nemo_toolkit de minh sua dung signature."
        )

    for start in range(0, len(audio), step_samples):
        end = min(len(audio), start + chunk_samples)
        chunk = audio[start:end]
        if chunk.size == 0:
            continue

        chunk_text = transcribe_chunk(chunk)
        if remove_lang_tags:
            chunk_text = strip_lang_tags(chunk_text)
        if not chunk_text:
            continue

        transcript, delta_text = merge_transcript(transcript, chunk_text)
        if delta_text:
            if transcript != delta_text:
                print(" ", end="", flush=True)
            print(delta_text, end="", flush=True)

    if remove_lang_tags:
        transcript = strip_lang_tags(transcript)
    print(f"\n\nFinal transcription: {transcript}")
    return transcript


def parse_args():
    parser = argparse.ArgumentParser(description="Nemotron 3.5 ASR streaming from file")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to an audio file.")
    parser.add_argument("--model", type=str, default=DEFAULT_NEMO_MODEL, help="NeMo model name.")
    parser.add_argument(
        "--target_lang",
        type=str,
        default="vi-VN",
        help='Language ID for prompt models (e.g. "vi-VN" or "auto").',
    )
    parser.add_argument(
        "--strip_lang_tags",
        action="store_true",
        help="Remove <xx-XX> language tags from output.",
    )
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=1.12,
        help="Chunk size in seconds.",
    )
    parser.add_argument(
        "--overlap_seconds",
        type=float,
        default=0.24,
        help="Overlap size in seconds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    nemotron_transcribe_file(
        args.audio_path,
        args.model,
        args.target_lang,
        args.chunk_seconds,
        args.overlap_seconds,
        remove_lang_tags=args.strip_lang_tags,
    )


if __name__ == "__main__":
    main()
