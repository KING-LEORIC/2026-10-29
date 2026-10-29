"""실습 07: 120건 학습, 40건 검증. test 40건은 다음 독립 평가용으로 보관합니다."""
from pathlib import Path
import csv
import json
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from check_gpu import choose_device
from model import SimpleANN
from train import train_one_epoch, evaluate
# 현재 파일(main.py)이 있는 폴더
base = Path(__file__).resolve().parent
# 결과 저장 폴더
out = base / "outputs"
# outputs 폴더가 없으면 생성
out.mkdir(exist_ok=True)
# CSV 읽기
with (base / "data" / "points.csv").open(encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))
# 학습(train) 데이터 추출
train_rows = [r for r in rows if r["split"] == "train"]
# 검증(valid) 데이터 추출
valid_rows = [r for r in rows if r["split"] == "valid"]
# Tensor 생성
# 입력 특성(x1, x2) → 학습용 텐서
X_train = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in train_rows])
# 정답(label) → 학습용 텐서
# - dtype=torch.long : CrossEntropyLoss가 요구하는 정수형 클래스 타입
y_train = torch.tensor([int(r["label"]) for r in train_rows], dtype=torch.long)
# 입력 특성(x1, x2) → 검증용 텐서
X_valid = torch.tensor([[float(r["x1"]), float(r["x2"])] for r in valid_rows])
# 정답(label) → 검증용 텐서
y_valid = torch.tensor([int(r["label"]) for r in valid_rows], dtype=torch.long)
# 시드(가중치 초기값) 고정 : 실행할 때마다 같은 결과를 얻기 위함
torch.manual_seed(42)
# CPU 또는 GPU 선택
device, info = choose_device()
# 모델 생성 후 선택한 장치(CPU/GPU)로 이동
model = SimpleANN().to(device)
# DataLoader
# - TensorDataset : 입력(X)과 정답(y)을 하나의 데이터셋으로 묶음
# - batch_size=32 : 한 번에 32개 샘플씩 학습
# - shuffle=True : 매 epoch마다 데이터 순서 섞기
# - generator : shuffle 결과 재현
# - num_workers=0 : 별도 프로세스 없이 데이터 로딩
loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True,
                    generator=torch.Generator().manual_seed(43), num_workers=0)
# 다중 클래스 분류용 손실 함수
loss_fn = nn.CrossEntropyLoss()
# SGD(확률적 경사하강법) 옵티마이저
# - model.parameters() : 학습할 가중치 목록
# - lr=0.1 : 학습률(가중치 수정 크기)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
# 학습 전 train loss 기록
initial = evaluate(model, X_train, y_train, loss_fn, device)[0]
# epoch별 결과 저장
history = []
# 총 80 epoch 학습
for epoch in range(1, 81):
    # train 데이터 학습
    online_loss = train_one_epoch(model, loader, loss_fn, optimizer, device)
    # valid 데이터 평가
    valid_loss, accuracy, prediction = evaluate(model, X_valid, y_valid, loss_fn, device)
    # 학습 기록 저장
    history.append([epoch, online_loss, valid_loss, accuracy])
    # 10 epoch마다 진행 상황 출력
    if epoch % 10 == 0:
        print(epoch, round(online_loss, 6), round(valid_loss, 6), round(accuracy, 4))
# 아래는 저장 과정입니다. 전체 Module 대신 직접 학습한 파라미터 사전만 저장합니다.
# model의 state_dict를 outputs/model_state.pt로 저장하세요.
# 배포 파일에는 미리 학습한 모델이 없습니다. 실행한 본인이 만든 파일만 복원합니다.
# >>> H09
# TODO H09: 위 요구사항을 읽고 이 블록을 완성하세요.
# pass

# model의 모든 학습된 가중치(weight)와 bias를 저장
# - 모델 구조는 저장하지 않고 파라미터(state_dict)만 저장
# - 나중에 같은 구조의 모델에서 복원 가능
torch.save(model.state_dict(), out / "model_state.pt")
# <<< H09
# epoch별 학습 기록 저장
with (out / "training_history.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w=csv.writer(f);w.writerow(["epoch","train_loss_online","valid_loss","valid_accuracy"]);w.writerows(history)
# 최종 성능 지표 저장
# - "initial_train_loss": initial : 학습 전 train loss
# - "final_train_loss" : 학습 후 train loss
# - "valid_loss": valid_loss : 최종 valid loss
# - "valid_accuracy": accuracy : # 최종 valid accuracy
metrics = {"epochs":80,"device":str(device),"train_rows":120,"valid_rows":40,"test_used":False,
           "initial_train_loss":initial,"final_train_loss":evaluate(model,X_train,y_train,loss_fn,device)[0],
           "valid_loss":valid_loss,"valid_accuracy":accuracy}
(out / "final_metrics.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf-8")
# GPU/CPU 정보 저장
(out / "gpu_check.txt").write_text("\n".join(f"{k}: {v}" for k,v in info.items()),encoding="utf-8")
# 검증 데이터 예측 결과 저장
with (out / "validation_predictions.csv").open("w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f);w.writerow(["id","actual","predicted"])
    w.writerows((r["id"],int(y),int(p)) for r,y,p in zip(valid_rows,y_valid,prediction))
print("저장 완료. 다음 순서: python restore.py")
