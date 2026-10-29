"""실습 05: 배치 평균들의 단순 평균이 아닌 샘플 수 가중 평균을 계산합니다."""
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# 앞의 8개 샘플은 [3, 0] (정답 클래스 0을 강하게 예측)
# 뒤의 2개 샘플은 [0, 3] (정답 클래스 1을 강하게 예측)
logits = torch.tensor([[3.,0.]]*8 + [[0.,3.]]*2)
# 모든 샘플의 실제 정답은 클래스 0
y = torch.zeros(10, dtype=torch.long)
# 배치 크기 4로 DataLoader 생성
# - 실제 배치는 (4, 4, 2)로 나뉨
loader = DataLoader(TensorDataset(logits, y), batch_size=4, shuffle=False)
# CrossEntropyLoss 기본값은 reduction='mean'
# - 즉 각 배치의 평균 loss를 반환
loss_fn = nn.CrossEntropyLoss()
# 전체 샘플 기준 평균 loss를 계산하기 위한 누적 변수
total_loss, count = 0.0, 0
# 각 배치의 평균 loss 저장용
batch_means = []
for xb, yb in loader:
    # 현재 배치의 평균 loss
    loss = loss_fn(xb, yb)
    # 현재 배치 평균 loss에 실제 배치 행 수를 곱해 total_loss에 더하세요.
    # 마지막 배치는 2행입니다. batch_size=4를 곱하면 안 됩니다.
    # >>> H05
    # TODO H05: 위 요구사항을 읽고 이 블록을 완성하세요.
    # 현재 배치 평균 loss에 실제 배치 크기를 곱해
    # 총 loss 합계에 누적
    # 마지막 배치는 2개 샘플이므로 len(yb)를 사용해야 함
    total_loss += loss.item() * len(yb)
    # <<< H05
    # 현재까지 처리한 샘플 수 누적
    count += len(yb)
    # 배치 평균 loss 저장
    batch_means.append(loss.item())
print("배치 크기: 4, 4, 2")
# 배치 평균들의 단순 평균
# - 배치 크기가 달라도 동일 가중치로 취급하는 잘못된 방식
print("잘못된 단순 평균:", round(sum(batch_means)/len(batch_means), 6))
# 각 배치를 샘플 수만큼 가중한 평균
# - 전체 데이터의 평균 loss와 일치
print("샘플 가중 평균:", round(total_loss/count, 6))
# 전체 10개 샘플을 한 번에 계산한 평균 loss
print("전체를 한 번에 계산:", round(loss_fn(logits, y).item(), 6))
