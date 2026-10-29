"""실습 04: 같은 파라미터의 Module과 Sequential을 비교합니다."""
import torch
from torch import nn

class SimpleANN(nn.Module):
    def __init__(self):
        # nn.Module 부모 클래스 초기화
        super().__init__()
        # Sequential로 신경망 구성
        # - 입력 2차원 → 은닉층 8개 → ReLU 활성화 함수 → 출력 2차원
        self.net = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,2))

    def forward(self, x):
        # 입력 x를 self.net에 넣은 결과를 반환하세요.
        # 출력은 (N,2) logits이며 softmax를 추가하지 않습니다.
        # >>> H04
        # TODO H04: 위 요구사항을 읽고 이 블록을 완성하세요.
        
        # Sequential 모델(self.net)에 입력 x를 전달
        # - 결과는 (배치크기, 2) 형태의 logits
        return self.net(x)
        # <<< H04

# 난수 시드 고정 //매번 실행할 때 동일한 초기 가중치 생성
torch.manual_seed(42)
# 사용자 정의 Module 생성
custom = SimpleANN()
# 동일한 구조의 Sequential 모델 생성
sequential = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,2))
# custom.net의 가중치와 bias를 sequential에 복사 : 서로 다른 랜덤 초기값을 비교하지 않도록 같은 값을 복사
sequential.load_state_dict(custom.net.state_dict())
# 테스트 입력 데이터
# - 배치 크기 = 3, 특성 수 = 2
X = torch.tensor([[1.,2.], [0.,1.], [-1.,0.]])
# 두 모델에 같은 입력 전달
a, b = custom(X), sequential(X)
# 출력 텐서의 shape 확인
print("출력 shape:", tuple(a.shape), tuple(b.shape))
# 두 모델의 출력이 완전히 같은지 확인
# - 가중치가 같고 구조가 같으므로 True 출력
print("동일 출력:", torch.allclose(a, b))
# custom 모델의 전체 파라미터 수 계산
# Linear(2→8): 가중치 16개 + bias 8개 = 24개
# Linear(8→2): 가중치 16개 + bias 2개 = 18개
# 총 42개
print("파라미터:", sum(p.numel() for p in custom.parameters()))
