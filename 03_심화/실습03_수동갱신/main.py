"""실습 03: 수동 Gradient Descent와 gradient 누적을 구분합니다."""
from pathlib import Path
import csv
import torch

# 학습할 파라미터(가중치) 생성
# - requires_grad=True : 역전파 시 gradient를 계산하도록 설정
w = torch.tensor(0.0, requires_grad=True)
# 각 step의 결과를 저장할 리스트
records = []
# 총 10번 Gradient Descent 수행
for step in range(1, 11):
    # 현재 w 값을 저장 (업데이트 전 값)
    before = w.item()
    # 손실 함수 정의
    # - 최소값은 w==4일 때 0이 됨
    loss = (w - 4) ** 2
    # 역전파 수행
    # - loss를 w에 대해 미분한 값이 w.grad에 저장됨
    loss.backward()
    # 계산된 gradient 저장
    gradient = w.grad.item()
    # 파라미터 업데이트 시에는 autograd 추적을 끔
    with torch.no_grad():
        # lr=0.1로 w를 갱신하세요. 아래 grad.zero_()는 다음 반복에 이전 기울기가 섞이지 않게 합니다.
        # >>> H03
        # TODO H03: 위 요구사항을 읽고 이 블록을 완성하세요.
        # pass
        # 학습률(Learning Rate)
        lr = 0.1
        # Gradient Descent: w = w - lr × gradient
        w -= lr * w.grad
        # <<< H03
    # PyTorch는 gradient를 자동으로 누적(accumulate)하므로, 다음 반복에 영향을 주지 않도록 0으로 초기화
    w.grad.zero_()
    # 현재 step 결과 저장
    # - step : 반복 번호
    # - before : 업데이트 전 w
    # - gradient : 계산된 gradient
    # - w.item() : 업데이트 후 w
    # - (w.detach() - 4).square().item() : 업데이트 후 loss
    records.append([step, before, gradient, w.item(), (w.detach()-4).square().item()])
# 학습 과정 출력
# - r[0] : step
# - round(r[1], 6) : w_before
# - round(r[2], 6) : gradient
# - round(r[3], 6) : w_after
# - round(r[4], 6) : loss_after
for r in records:
    print(r[0], round(r[1], 6), round(r[2], 6), round(r[3], 6), round(r[4], 6))
# gradient 누적 예제
u = torch.tensor(1.0, requires_grad=True)
# 첫 번째 역전파
# - u^2를 u로 미분하면 2u = 2
(u**2).backward()
# 첫 번째 gradient 저장
first = u.grad.item()
# 같은 그래프를 다시 쓰는 대신 새로운 순전파 그래프를 만듭니다.
(u**2).backward()
# grad를 초기화하지 않았기 때문에, 두 번째 gradient가 기존 값에 누적됨
# - 결과: 2 -> 4
print("초기화하지 않은 gradient:", first, "->", u.grad.item())
# 출력 디렉터리 생성
out = Path(__file__).resolve().parent / "outputs"
out.mkdir(exist_ok=True)
# 학습 기록을 CSV 파일로 저장
# - writer.writerow(["step","w_before","gradient","w_after","loss_after"]) : CSV 헤더 저장
# - writer.writerows(records) : 학습 데이터 저장
with (out / "updates.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer=csv.writer(f); writer.writerow(["step","w_before","gradient","w_after","loss_after"]); writer.writerows(records)
