[OPEN] tts-cuda-not-used

# Muc tieu
- Xac dinh vi sao `Vieneu(backbone_device="cuda")` co ve khong chay bang GPU.

# Trieu chung
- User thay doi code sang CUDA nhung nghi rang backend van dang chay CPU.

# Gia thuyet
- H1: Thu vien `vieneu` trong moi truong hien tai khong phai ban ho tro GPU, nen bo qua `backbone_device="cuda"`.
- H2: May hien tai khong co CUDA/NVIDIA runtime phu hop, nen `Vieneu` fallback ve CPU.
- H3: Tham so `backbone_device="cuda"` dung, nhung model/backbone dang tai la backend chi ho tro CPU.
- H4: File dang sua chua phai file/thoi diem duoc chay thuc te, nen thay doi CUDA chua duoc ap dung vao runtime.
- H5: Co exception/canh bao luc khoi tao, nhung chua duoc quan sat nen user chi thay hieu ung "khong an vao".

# Ke hoach
- Thu thap bang chung runtime ve moi truong Python, GPU, va thong tin khoi tao `Vieneu`.
- Neu can, them instrumentation toi thieu vao file de in ra backend/device thuc te dang dung.
- Chi sua logic sau khi co bang chung xac nhan nguyen nhan.

# Bang chung
- Windows Python mac dinh cua terminal: `C:\Users\Admin\miniconda3\python.exe`, khong co package `vieneu`.
- Runtime thuc te cua user la WSL prompt `(streaming) ... /mnt/d/translate_extention$`.
- WSL nhin thay GPU: `nvidia-smi` hien `NVIDIA GeForce RTX 3060`, CUDA `13.2`.
- Trong env WSL `streaming`, Python la `/home/dinhtu2207/miniforge3/envs/streaming/bin/python`, version `3.11.15`.
- Trong env `streaming`, co `vieneu==3.0.5`.
- Trong env `streaming`, co `onnxruntime==1.26.0`.
- Trong env `streaming`, khong co `onnxruntime-gpu`, `lmdeploy`, `llama-cpp-python`.

# Danh gia gia thuyet
- H1: Xac nhan mot phan. Moi truong dang cai runtime backend thien ve CPU (`onnxruntime` thay vi `onnxruntime-gpu`).
- H2: Bac bo. WSL nhin thay GPU va CUDA binh thuong.
- H3: Rat co kha nang dung. `backbone_device=\"cuda\"` khong du de ep GPU neu backend da cai dat chi la CPU runtime.
- H4: Xac nhan. Lan kiem tra dau tien da o sai moi truong (Windows Python thay vi WSL `streaming`).
- H5: Chua can them bang chung; nguyen nhan chinh da du ro tu dependency runtime.

# Ket luan tam thoi
- Van de nam o moi truong chay, khong nam chinh o dong code `backbone_device=\"cuda\"`.
- Chi doi code sang `cuda` la chua du; env `streaming` can duoc cai GPU runtime phu hop cho VieNeu.
