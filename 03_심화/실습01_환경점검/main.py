"""실습 01: 실제 선택한 device와 PyTorch 빌드를 구분해 기록합니다."""
from pathlib import Path
import json
# PyTorch 라이브러리
import torch

# CUDA 사용 가능 여부를 보고 실행할 장치를 선택하세요.
# 드라이버가 있다는 사실만으로 CUDA를 선택하지 마세요.
# >>> H01
# TODO H01: 위 요구사항을 읽고 이 블록을 완성하세요.
# torch.cuda.is_available()가 True이면 CUDA GPU 사용, 그렇지 않으면 CPU 사용
# - 실제 계산 가능한 장치를 선택하는 것이 목적
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# <<< H01
# 2x2 실수 텐서 생성 후 선택한 장치(CPU 또는 GPU)로 이동
x = torch.tensor([[1., 2.], [3., 4.]]).to(device)
# 행렬 곱 수행 (x × x의 전치행렬)
result = x @ x.T
# 실행 환경 및 계산 결과를 저장할 정보 딕셔너리 생성
# - str(torch.__version__) : 현재 사용 중인 PyTorch 버전
# - torch.version.cuda : 현재 설치된 PyTorch가 어떤 CUDA 버전으로 빌드되었는지 //CPU 전용 빌드라면 None
# - torch.cuda.is_available() : 현재 시스템에서 CUDA 사용 가능 여부
# - str(device) : 실제 선택된 실행 장치
# - result.cpu().tolist() : 결과 텐서를 CPU로 옮긴 뒤 Python 리스트로 변환
record = {"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
          "cuda_available": torch.cuda.is_available(), "selected_device": str(device),
          "calculation": result.cpu().tolist()}
# GPU를 사용하는 경우 추가 하드웨어 정보 기록
# - torch.cuda.get_device_name(device): GPU 모델명
# - torch.cuda.get_device_properties(device).total_memory / 1024**3 : # 총 GPU 메모리 용량(GiB 단위)
if device.type == "cuda":
    record["gpu_name"] = torch.cuda.get_device_name(device)
    record["vram_gib"] = torch.cuda.get_device_properties(device).total_memory / 1024**3
# 현재 파이썬 파일이 있는 위치 기준으로 outputs 폴더 생성
out = Path(__file__).resolve().parent / "outputs"
# 폴더가 없으면 생성, 있으면 무시
out.mkdir(exist_ok=True)
# 수집한 환경 정보를 environment.json 파일로 저장
(out / "environment.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
# 결과를 화면에도 출력
print(json.dumps(record, ensure_ascii=False, indent=2))
