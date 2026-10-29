"""학습 한 epoch와 평가를 분리합니다. 빈 데이터·불량 자료는 이번 단계의 과제가 아닙니다."""
import torch

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    # 학습 모드
    # - Dropout, BatchNorm 같은 계층이 있으면 학습 방식으로 동작
    model.train()
    total, count = 0.0, 0
    # DataLoader가 만든 미니배치 반복
    for xb, yb in loader:
        # 배치 데이터를 지정한 장치(CPU/GPU)로 이동
        xb, yb = xb.to(device), yb.to(device)
        # 이전 배치의 gradient 제거 //PyTorch는 gradient를 누적하기 때문에 매 배치마다 초기화 필요
        optimizer.zero_grad()
        # 손실 계산 : 예측 결과가 정답과 얼마나 다른지 측정
        # - 예측 결과 계산 후 손실 함수 적용
        loss = loss_fn(model(xb), yb)
        # loss.backward() 후 optimizer.step()을 호출하세요.
        # 매 batch마다 초기화하고 새로운 gradient로 갱신해야 합니다.
        # >>> H07
        # TODO H07: 위 요구사항을 읽고 이 블록을 완성하세요.
        # pass

        # 역전파 수행
        # - 오차를 이용하여 각 가중치가 얼마나 잘못했는지 계산
        loss.backward()
        # 실제 가중치로 업데이트
        optimizer.step()
        # <<< H07
        # 현재 배치 손실 누적
        total += loss.item() * len(yb)
        # 현재 배치의 샘플 수 누적
        count += len(yb)
    # 마지막 작은 batch도 실제 샘플 수만큼 반영합니다.
    # 전체 데이터 평균 손실 반환
    return total / count


def evaluate(model, X, y, loss_fn, device):
    # 평가 모드
    # - Dropout 등이 꺼짐
    model.eval()
    # 평가에서는 학습을 하지 않으므로 gradient 계산 비활성화
    with torch.no_grad():
        # 예측 수행 : 평가 데이터 전달
        logits = model(X.to(device))
        # 손실 계산
        loss = loss_fn(logits, y.to(device)).item()
        # logits에서 클래스 인덱스를 구하고 CPU로 옮겨 prediction에 저장하세요.
        # 평가에서는 backward와 optimizer.step을 호출하지 않습니다.
        # >>> H08
        # TODO H08: 위 요구사항을 읽고 이 블록을 완성하세요.

        # 가장 높은 점수의 클래스 선택
        prediction = logits.argmax(dim=1).cpu()
        # <<< H08
    # 손실, 정확도, 예측 반환
    return loss, (prediction == y.cpu()).float().mean().item(), prediction
