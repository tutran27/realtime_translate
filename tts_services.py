import os

from vieneu import Vieneu


class TTSService:
    def __init__(self, text, output_dir="outputs"):
        self.tts = Vieneu(backbone_device="cuda")
        self.text = text
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def synthesize(self):
        voices = self.tts.list_preset_voices()
        if not voices:
            return None

        _, voice_id = voices[5]
        voice = self.tts.get_preset_voice(voice_id)
        audio = self.tts.infer(text=self.text, voice=voice)
        output_path = os.path.join(self.output_dir, "output.wav")
        self.tts.save(audio, output_path)
        print(f"Saved: {output_path}")
        return output_path

    def close(self):
        self.tts.close()


def main():
    service = TTSService(
        text="Computer vision là một lĩnh vực trong Trí tuệ nhân tạo. Nó giúp nhận diện, phân loại đối tượng một cách rõ ràng thông qua các phương thức như object detection, segmentation... Thể hiện qua các bounding box hay phân đoạn",
        output_dir="outputs",
    )
    try:
        service.synthesize()
    finally:
        service.close()


if __name__ == "__main__":
    main()
