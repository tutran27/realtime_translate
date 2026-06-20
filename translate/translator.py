import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL_NAME = "VietAI/envit5-translation"

ENVIT5_LANGUAGE_MAP = {
    "en": "en",
    "english": "en",
    "eng_latn": "en",
    "vi": "vi",
    "vietnamese": "vi",
    "vie_latn": "vi",
}


def resolve_envit5_language(language: str | None) -> str | None:
    if not language:
        return None
    return ENVIT5_LANGUAGE_MAP.get(language.strip().lower())


class MachineTranslator:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError(f"Thiet bi dich khong duoc ho tro: {device}")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA khong san sang cho model dich")

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
    ) -> str:
        return self.translate_batch([(text, src_lang, tgt_lang)])[0]

    def translate_batch(
        self,
        requests: list[tuple[str, str, str]],
    ) -> list[str]:
        results: list[str | None] = [None] * len(requests)
        model_inputs: list[str] = []
        target_languages: list[str] = []
        model_indexes: list[int] = []

        for index, (text, src_lang, tgt_lang) in enumerate(requests):
            text = text.strip()
            source_language = resolve_envit5_language(src_lang)
            target_language = resolve_envit5_language(tgt_lang)
            if not text:
                results[index] = ""
                continue
            if not source_language or not target_language:
                raise ValueError(
                    "VietAI/envit5-translation chi ho tro tieng Anh va tieng Viet"
                )
            if source_language == target_language:
                results[index] = text
                continue

            model_inputs.append(f"{source_language}: {text}")
            target_languages.append(target_language)
            model_indexes.append(index)

        if not model_inputs:
            return [result or "" for result in results]

        with torch.inference_mode():
            encoded = self.tokenizer(
                model_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            ).to(self.device)
            generated = self.model.generate(
                **encoded,
                max_new_tokens=128,
                num_beams=1,
                use_cache=True,
            )

        translated_batch = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )
        for index, translated, target_language in zip(
            model_indexes,
            translated_batch,
            target_languages,
        ):
            translated = translated.strip()
            target_prefix = f"{target_language}:"
            if translated.lower().startswith(target_prefix):
                translated = translated[len(target_prefix) :].strip()
            results[index] = translated

        return [result or "" for result in results]
