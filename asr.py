import os

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.clean_up_tokenization_spaces = False
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = None
model.generation_config.begin_suppress_tokens = None

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device,
)

audio_path = "outputs/standard_Gia Bảo.wav"

if not os.path.exists(audio_path):
    raise FileNotFoundError(f"Khong tim thay file audio: {audio_path}")

result = pipe(
    audio_path,
    return_timestamps=True,
    generate_kwargs={"task": "transcribe", "language": "vi"},
)
print(result["text"])
