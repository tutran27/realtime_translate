import os

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from dotenv import load_dotenv

load_dotenv()

#  ----------- Config ----------- #
HF_TOKEN = os.getenv("HF_TOKEN")
model_id = "openai/whisper-large-v3-turbo"
audio_path = "audio/shoes.mp3"
# ------------------------------- #

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("Loading model...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

print("Loading processor...")
processor = AutoProcessor.from_pretrained(model_id)
processor.tokenizer.clean_up_tokenization_spaces = False
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = None
model.generation_config.begin_suppress_tokens = None

print("Loading pipeline...")
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    dtype=dtype,
    device=device,
)

if not os.path.exists(audio_path):
    raise FileNotFoundError(f"Khong tim thay file audio: {audio_path}")

print("Running pipeline...")

result = pipe(
    audio_path,
    return_timestamps=True,
    generate_kwargs={"task": "transcribe", "language": "vi"},
)
print(result["text"])