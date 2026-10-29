"""실습 06: 같은 모델의 train·valid 학습 기록을 남깁니다."""

from pathlib import Path
import csv
import json
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# 현재 파이썬 파일이 위치한 폴더 경로
base = Path(__file__).resolve().parent
# 결과 저장 폴더 생성
out = base / "outputs"
out.mkdir(exist_ok=True)
# CSV 데이터 읽기
with (base / "data" / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
# split 컬럼 기준으로 train/valid 분리
train_rows = [r for r in rows if r["split"] == "train"]
valid_rows = [r for r in rows if r["split"] == "valid"]
# 입력 특성(x1, x2)을 Tensor로 변환
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
# 정답 레이블을 long 타입 Tensor로 변환
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
# 검증 데이터 입력
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
# 검증 데이터 정답
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# test 40건은 모델에 전달하지 않습니다. 순서를 바꾸거나 다시 분할하지 않습니다.
# 재현 가능한 결과를 위한 랜덤 시드 고정
torch.manual_seed(42)
# GPU가 있으면 GPU, 없으면 CPU 사용
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 미니배치 학습용 DataLoader 생성
# - shuffle=True로 매 epoch마다 데이터 순서를 섞음
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(43), num_workers=0)
# 간단한 MLP 모델
# - 입력 2개 -> 은닉층 8개(ReLU) -> 출력 2개(이진분류)
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2)).to(device)
# 분류 문제용 손실 함수
loss_fn = nn.CrossEntropyLoss()
# SGD 최적화기
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# 검증 데이터 GPU/CPU 이동
Xv, yv = X_valid.to(device), y_valid.to(device)
# 전체 train 데이터도 평가용으로 이동
Xt, yt = X_train.to(device), y_train.to(device)
# 학습 기록 저장용 리스트
history = []

# epoch 0~80
for epoch in range(81):
    # epoch=0은 학습 전 성능 측정만 수행
    if epoch > 0:
        # 학습 모드
        model.train()
        # 미니배치 반복
        for xb, yb in loader:
            # 배치를 장치(GPU/CPU)로 이동
            xb, yb = xb.to(device), yb.to(device)
            # 이전 gradient 초기화
            optimizer.zero_grad()
            # 순전파(forward)
            loss = loss_fn(model(xb), yb)
            # 현재 batch loss의 역전파와 SGD 갱신을 작성하세요.
            # 검증은 아래 eval/no_grad 블록에서만 수행합니다.
            # >>> H06
            # TODO H06: 위 요구사항을 읽고 이 블록을 완성하세요.
            # pass
            # 기울기 계산(backpropagation)
            loss.backward()
            # 파라미터 업데이트
            optimizer.step()
            # <<< H06
    # epoch=0은 학습 전입니다. 로그는 갱신이 끝난 동일 모델로 측정합니다.
    # 10 epoch마다 성능 기록
    if epoch % 10 == 0:
        # 평가 모드
        model.eval()
        # gradient 계산 비활성화
        with torch.no_grad():
            # train 데이터 전체 손실
            train_loss = loss_fn(model(Xt), yt).item()
            # valid 예측값 계산
            logits = model(Xv)
            # valid 손실
            valid_loss = loss_fn(logits, yv).item()
            # valid 정확도
            accuracy = (logits.argmax(dim=1) == yv).float().mean().item()
        # 기록 저장
        history.append([epoch, train_loss, valid_loss, accuracy])
        # 콘솔 출력
        print(epoch, round(train_loss, 6), round(valid_loss, 6), round(accuracy, 4))

# 학습 기록 CSV 저장
with (out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    # 헤더 작성
    writer.writerow(["epoch", "train_loss_eval", "valid_loss", "valid_accuracy"])
    # 기록 작성
    writer.writerows(history)
