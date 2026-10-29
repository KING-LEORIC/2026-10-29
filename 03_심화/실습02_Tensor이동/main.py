"""실습 02: 모양 변환과 장치 이동 후 같은 장치에서 연산합니다."""
import torch
from torch import nn

# 실행 결과 재현을 위한 시드 고정
torch.manual_seed(42)
# GPU 사용 가능 시 CUDA 사용, 아니면 CPU 사용
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 0~11까지의 실수형 텐서 생성
values = torch.arange(12, dtype=torch.float32)
# 12개 원소를 (3,4)로 바꾸고 네 열에 [1,2,3,4]를 더하세요.
# 이후 앞의 두 열을 신경망 입력으로 사용합니다.
# >>> H02
# TODO H02: 위 요구사항을 읽고 이 블록을 완성하세요.
# 0~11 값을 3행 4열 텐서로 변환
matrix = values.reshape(3, 4)
# 열별 보정값 [1, 2, 3, 4]를 브로드캐스팅하여 더함
shifted = matrix + torch.tensor([1, 2, 3, 4], dtype=torch.float32)
# <<< H02
# 앞의 두 열만 입력 특성(feature)으로 사용
X = shifted[:, :2].to(device)
# 분류용 정답 레이블
y = torch.tensor([0, 0, 1], dtype=torch.long).to(device)
# 입력 2개 → 출력 2개인 선형 계층 생성
model = nn.Linear(2, 2).to(device)
# 모델 추론 수행
logits = model(X)
# 결과 확인
print("행렬:", matrix.tolist())
print("보정:", shifted.tolist())
print("입력/출력:", tuple(X.shape), tuple(logits.shape))
# 입력 데이터, 레이블, 모델이 모두 같은 장치(CPU/GPU)에 있는지 확인
print("device 일치:", X.device == y.device == next(model.parameters()).device)
