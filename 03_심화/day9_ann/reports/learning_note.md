# ANN 학습 설명

## 파일별 책임
main / model / train / check_gpu / restore의 역할: 
1. main.py
데이터 읽기
train/valid 분리
모델 생성
학습 반복(epoch)
성능 평가
모델 저장(state_dict)
결과 파일 저장
2. model.py
ANN 구조 정의
입력 2개(x1, x2) → 은닉층 8개 → 출력 2개(label 0, 1)
3. train.py
1 epoch 학습(train_one_epoch)
검증(evaluate)
손실(loss) 및 정확도(accuracy) 계산
4. check_gpu.py
CUDA 사용 가능 여부 확인
학습 장치(CPU/GPU) 선택
GPU 이름, VRAM 등 환경 정보 기록
5. restore.py
저장한 state_dict 로드
새 프로세스에서 모델 복원
저장 당시 예측 결과와 동일한지 검증

## 학습 순서와 손실 집계
zero_grad → forward → loss → backward → step의 의미: 
1. zero_grad()
이전 batch에서 계산된 gradient 제거
PyTorch는 gradient를 누적하므로 매 batch마다 초기화 필요
2. forward
입력 데이터를 모델에 통과시켜 예측값(logits) 계산
3. loss
예측값과 정답(label)의 차이를 CrossEntropyLoss로 계산
4. backward()
오차를 역전파하여 각 가중치의 gradient 계산
5. step()
SGD optimizer가 gradient를 이용해 가중치 갱신

마지막 batch24건을 어떻게 집계했는가: 
학습 데이터는 총 120건이고 batch_size=32이므로 32 + 32 + 32 + 24 = 120으로 처리된다.
손실 집계는
total += loss.item() * len(yb)
count += len(yb)
방식을 사용하였다.
따라서 마지막 batch가 24건이라도 동일한 가중치로 평균에 반영되며,
return total / count
으로 전체 120건 기준 평균 loss를 계산하였다.

## 검증과 한계
실제로 사용한 device: GPU(cuda:0)

train/valid/test 각 표본 수와 사용 범위: 
구분	표본 수 	  사용 목적
train	120	        모델 학습
valid	 40	        성능 검증
test	 40	     미사용(독립 평가용 보관)

valid 결과: 
초기 상태
Train Loss : 0.565832
최종 결과
Train Loss : 0.089844
Valid Loss : 0.057217
Valid Accuracy : 1.0000
Accuracy는 epoch 23부터 100%에 도달하였으며 이후에도 유지되었다.
최종 학습 이력은 다음과 같다.
항목                    값
Epoch                  80
Final Train Loss       0.089844
Final Valid Loss       0.057217
Final Valid Accuracy   100.0%

단순 합성 데이터에서 높은 정확도가 실제 성능 증명이 아닌 이유: 
이번 데이터는 x1, x2 두 개의 수치만 사용한 단순 이진 분류 데이터이다.
데이터를 살펴보면 대체로
x1이 큰 양수 → label 1
x1이 큰 음수 → label 0
패턴이 비교적 명확하게 나타난다.
따라서 작은 ANN으로도 결정 경계를 쉽게 학습할 수 있다.
또한
데이터 수가 많지 않음
특성 수가 2개뿐임
생성 규칙이 비교적 단순함
실제 현장의 노이즈가 존재하지 않음
이라는 특징이 있다.
따라서 Validation Accuracy가 100%라고 해서 실제 복잡한 문제에서도 동일한 성능을 낸다고 결론 내릴 수는 없다.

## 복원
새 프로세스에서 ID 순서와 예측이 같은가: 
restore.py에서 model.load_state_dict(state_dict)로 저장된 파라미터를 불러온 뒤 검증 데이터를 다시 예측하였다.
검사 항목은
1. ID 순서 동일 여부
2. 예측 결과 동일 여부
이며, 정상적으로 복원되었다면 두 결과 모두 True가 된다.
이는 저장 전 모델과 복원 후 모델이 동일한 예측을 수행함을 의미한다.

학습 버전과 장치:
PyTorch : 학습 환경 기준 설치 버전 사용
Device  : cuda:0
Epoch   : 80
Train   : 120건
Valid   : 40건
Test    : 사용 안 함
학습 완료 후 모델은 model_state.pt에 state_dict 형태로 저장되었으며, restore.py에서 CPU 환경으로 복원 가능하도록 구현하였다.