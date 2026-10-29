"""모델 구조만 정의합니다. 이 파일을 실행해도 학습하지 않습니다."""
from torch import nn

class SimpleANN(nn.Module):
    def __init__(self):
        super().__init__()
        # 입력 2개 : x1, x2
        # 은닉층 노드 8개
        # ReLU 활성화 함수
        # 출력 2개 노드 : 클래스 0, 1
        self.net = nn.Sequential(nn.Linear(2,8), nn.ReLU(), nn.Linear(8,2))

    def forward(self, x):
        # 입력 데이터를 신경망에 통과시켜 결과 반환
        return self.net(x)
