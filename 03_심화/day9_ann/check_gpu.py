"""장치 선택과 기록만 담당하는 제공 코드입니다."""
import torch

def choose_device():
    # CUDA 사용 가능 시 GPU(cuda:0), 아니면 CPU 사용
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 현재 PyTorch 및 CUDA 환경 정보 수집
    info = {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(), "selected_device": str(device)}
    # GPU 사용 시 GPU 이름과 VRAM 크기 기록
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        # 바이트 → GiB 변환
        info["vram_gib"] = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return device, info
