# Realtime STT và Machine Translation

Pipeline hiện tại:

```text
Microphone hoặc file audio
    -> frontend AudioWorklet, PCM16 mono 16 kHz
    -> WebSocket
    -> Silero VAD trên CPU
    -> Qwen3-ASR 0.6B trên GPU
    -> VietAI/envit5-translation trên một worker CPU riêng
    -> VieNeu TTS synth audio trong RAM
    -> frontend hiển thị transcript, bản dịch và phát audio realtime
```

## Cấu trúc

```text
stt/
  realtime_server.py    Backend WebSocket, VAD và Qwen3-ASR
  audio.py              Tiền xử lý PCM và Silero VAD dùng chung
translate/
  translator.py         Model VietAI/envit5-translation
  service.py            Worker dịch bất đồng bộ trên CPU
tts/
  service.py            Worker TTS bất đồng bộ, trả WAV bytes trong memory
frontend/
  main.html             Giao diện realtime + phát TTS trực tiếp
  audio-worklet.js      Resample audio trình duyệt
requirements.txt
```

## Yêu cầu

- WSL2 hoặc Linux.
- Python 3.11.
- GPU NVIDIA với CUDA tương thích PyTorch/vLLM.
- Chrome, Edge hoặc trình duyệt Chromium.

## Cài đặt

```bash
cd /mnt/d/translate_extention
conda activate streaming
pip install -r requirements.txt
```

Neu env da bi nang sai version va bao conflict voi `qwen-asr`/`vllm`,
chay them:

```bash
pip install "accelerate==1.12.0" "huggingface-hub==0.36.0" "transformers==4.57.6" --upgrade
pip check
```

Model Qwen, EnviT5 và VieNeu được tải trong lần chạy đầu. Backend chỉ mở
WebSocket sau khi cả ASR, EnviT5 và VieNeu đã warm-up xong.

## Chạy backend

Từ WSL:

```bash
PYTHONPATH=/mnt/d/translate_extention \
/home/dinhtu2207/miniforge3/envs/streaming/bin/python -m stt.realtime_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model Qwen/Qwen3-ASR-0.6B \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.6 \
  --step-ms 200 \
  --chunk-size-sec 1.0 \
  --max-segment-sec 10 \
  --vad-threshold 0.5 \
  --vad-min-silence-ms 900 \
  --vad-speech-pad-ms 250 \
  --vad-pre-roll-ms 300 \
  --translation \
  --translation-device cpu \
  --translation-model VietAI/envit5-translation \
  --tts \
  --tts-mode v3turbo \
  --tts-voice "Bình An"
```

`PYTHONPATH` giúp lệnh chạy được dù terminal đang đứng ở thư mục khác.

Sau khi tất cả model đã được tải đầy đủ ít nhất một lần, có thể tránh các
request kiểm tra Hugging Face khi startup:

```bash
export HF_HUB_OFFLINE=1
```

Server sẵn sàng khi hiển thị:

```text
Qwen realtime server: ws://0.0.0.0:8000/v1/realtime
```

## Ý nghĩa tham số

| Tham số | Ý nghĩa | Tác động khi điều chỉnh |
|---|---|---|
| `--host 0.0.0.0` | Địa chỉ backend lắng nghe. | Dùng `127.0.0.1` nếu chỉ truy cập từ cùng máy. |
| `--port 8000` | Cổng WebSocket. | Phải trùng với cổng trên frontend. |
| `--model` | Model nhận dạng giọng nói. | Model lớn hơn thường cần thêm VRAM và thời gian suy luận. |
| `--max-model-len 2048` | Context và KV cache của ASR. | Tăng khi đoạn dài bị thiếu; giảm để tiết kiệm VRAM. |
| `--gpu-memory-utilization 0.6` | Tỷ lệ VRAM tối đa vLLM sử dụng. | Tăng khi còn VRAM; giảm khi khởi tạo engine thất bại. |
| `--step-ms 200` | Lượng audio gom trước mỗi lần đẩy vào streaming ASR. | Giá trị thấp cập nhật dày hơn nhưng tăng số lần gọi Python. |
| `--chunk-size-sec 1.0` | Kích thước chunk nội bộ của Qwen streaming. | Nhỏ hơn phản hồi nhanh nhưng transcript dễ dao động. |
| `--max-segment-sec 15` | Thời lượng tối đa trước khi chốt và dịch segment. | Quá thấp dễ tạo dấu chấm giữa câu; quá cao làm dịch chậm xuất hiện. |
| `--vad-threshold 0.5` | Ngưỡng phát hiện giọng nói. | Giảm cho giọng nhỏ; tăng khi tiếng ồn bị nhận nhầm. |
| `--vad-min-silence-ms 900` | Khoảng im lặng để kết thúc segment. | Tăng để tránh cắt câu; giảm để dịch xuất hiện nhanh. |
| `--vad-speech-pad-ms 250` | Audio giữ thêm quanh vùng có tiếng nói. | Tăng nếu mất âm đầu hoặc cuối câu. |
| `--vad-pre-roll-ms 300` | Audio giữ trước lúc VAD phát hiện giọng nói. | Tăng nếu mất âm tiết đầu tiên. |
| `--translation` | Bật phase dịch. | Dùng `--no-translation` để chỉ chạy STT. |
| `--translation-device cpu` | Thiết bị chạy EnviT5. | CPU không tranh VRAM với Qwen. |
| `--translation-model` | Model dịch Anh-Vi. | Mặc định là `VietAI/envit5-translation`. |
| `--tts` | Bật phase TTS realtime. | Dùng `--no-tts` nếu chỉ cần STT hoặc STT + dịch. |
| `--tts-mode v3turbo` | Preset mode VieNeu chạy trên GPU. | Dùng chung GPU với Qwen nên cần chừa VRAM bằng `--gpu-memory-utilization`. |
| `--tts-voice "Bình An"` | Voice mặc định cho mỗi session. | Có thể đổi lại trên frontend sau khi kết nối. |

Bộ `15 giây / 900 ms` ưu tiên giữ câu đầy đủ và hạn chế dấu chấm giả giữa
câu. Nếu cần bản dịch nhanh hơn, giảm `--max-segment-sec`; nếu câu bị cắt,
tăng `--vad-min-silence-ms`.

## Chạy frontend

Mở terminal thứ hai trên Windows PowerShell:

```powershell
cd D:\translate_extention
python -m http.server 7800
```

Neu chay frontend tu WSL:

```bash
cd /mnt/d/translate_extention
python -m http.server 7800
```

Truy cập:

```text
http://localhost:7800/frontend/main.html
```

Thiết lập:

1. `Host`: `localhost`.
2. `Port`: `8000`.
3. `Model`: `Qwen/Qwen3-ASR-0.6B`.
4. `Language`: `English`, `Vietnamese` hoặc tự động.
5. `Dịch sang`: tiếng Việt, English hoặc tắt dịch.
6. `Bật TTS`: `Bật` nếu muốn nghe audio trực tiếp từ backend.
7. `Nguồn đọc`: `Bản dịch` hoặc `Transcript gốc`.
8. `Voice`: chọn một preset voice từ backend.
9. `Audio chunk`: `100 ms`.
10. `Tốc độ file`: `1.0x`.

Không mở HTML bằng `file://`; microphone cần `http://localhost` hoặc HTTPS.
EnviT5 hiện chỉ hỗ trợ dịch Anh-Vi và Vi-Anh.
TTS chỉ tự synth khi text đầu ra là tiếng Việt.

## Luồng TTS realtime

Backend không ghi file `.wav` ra ổ đĩa. Mỗi segment sau khi chốt sẽ:

1. STT realtime trả `transcription.segment.done`.
2. Nếu bật dịch, backend dịch và trả `translation.segment.done`.
3. Nếu bật TTS và đầu ra là tiếng Việt, backend synth trong memory.
4. Text được đưa vào một hàng đợi FIFO để không tạo nhiều inference TTS tranh GPU.
5. Cụm đầu giới hạn khoảng 48 ký tự; các cụm sau tối đa khoảng
   72 ký tự để giảm tổng số lần inference.
6. Mỗi cụm synth xong được gửi ngay qua event `tts.segment.audio`.
7. Backend cắt khoảng lặng thừa ở đầu/cuối mỗi WAV nhưng vẫn giữ pause theo dấu câu.
8. Frontend đệm khoảng 4 giây audio, sau đó lập lịch AudioBuffer nối tiếp với
   crossfade ngắn; nếu buffer cạn thì tự đệm lại trước khi phát tiếp.

Audio đầu vào được gửi bằng WebSocket binary frame, không dùng base64/JSON.
TTS vẫn trả WAV base64 vì cần gửi kèm metadata của segment.

EnviT5 dùng một worker CPU và tự gom tối đa 4 segment đang chờ thành một
micro-batch. Request đơn được dịch ngay; chỉ các request đã dồn hàng mới được
batch để tăng tốc độ xả backlog mà không nhân bản model trong RAM.

Log latency được tách thành:

- `queue_wait`: thời gian chờ trước khi model bắt đầu xử lý.
- `inference`: thời gian model xử lý batch hoặc chunk hiện tại.
- `total`: tổng thời gian từ lúc segment được đưa vào phase đến khi hoàn tất.

Cách này giảm thời gian chờ WAV đầu tiên và hạn chế khoảng im lặng giữa các
đoạn. Benchmark một câu dài trên máy hiện tại:

- Audio đầu tiên sẵn sàng sau khoảng `2.4 giây`.
- Các cụm sau thường được synth trong lúc cụm trước đang phát.
- Khoảng hở ước tính giữa các cụm giảm từ vài giây xuống gần `0-0.1 giây`.

## Latency tham khảo

Đo trên môi trường hiện tại sau warm-up:

| Phase | Latency |
|---|---:|
| Qwen ASR sau warm-up | khoảng `0.22 giây / 4 giây audio` trong benchmark |
| EnviT5 CPU, câu ngắn | khoảng `0.45-0.50 giây` |
| EnviT5 khi chạy cùng pipeline, segment dài | khoảng `2.0 giây` |
| VieNeu v3turbo, câu ngắn | khoảng `1.7 giây` |
| VieNeu, output audio khoảng 10 giây | khoảng `7.3 giây` |
| Chờ VAD chốt câu | `0.9 giây` im lặng |

VieNeu là phase xử lý chậm nhất. Backend preload và warm-up model trước khi
báo sẵn sàng. Warm-up ASR chuyển khoảng `16 giây` compile của inference đầu
tiên sang startup, vì vậy client đầu tiên không phải chịu chi phí này.
Với model đã cache và `HF_HUB_OFFLINE=1`, full startup đo được khoảng
`65 giây` trên máy hiện tại.

## Chỉ chạy STT

Thêm tùy chọn:

```bash
--no-translation
--no-tts
```

Trên frontend chọn `Tắt dịch`.

## Điều chỉnh nhanh

- Giọng nhỏ bị bỏ sót: giảm `--vad-threshold` xuống `0.35` đến `0.45`.
- Câu bị cắt sớm: tăng `--vad-min-silence-ms` lên `1000` đến `1200`.
- Mất âm đầu: tăng `--vad-pre-roll-ms`.
- Dịch xuất hiện chậm: giảm `--max-segment-sec` xuống `10`.
- Thiếu VRAM: giảm `--max-model-len` hoặc `--gpu-memory-utilization`.
