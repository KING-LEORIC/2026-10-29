"""실습 08: main.py가 만든 state_dict만 로드합니다. 새 프로세스 CPU에서 복원합니다."""
from pathlib import Path
import csv
import torch
from model import SimpleANN

base = Path(__file__).resolve().parent
# 검증 데이터만 읽기
with (base / "data" / "points.csv").open(encoding="utf-8",newline="") as f:
    valid_rows = [r for r in csv.DictReader(f) if r["split"] == "valid"]
# 입력 텐서 생성
X = torch.tensor([[float(r["x1"]),float(r["x2"])] for r in valid_rows])
# 동일한 모델 구조 생성
model = SimpleANN()
# 직접 저장한 model_state.pt를 map_location="cpu", weights_only=True로 읽고 모델에 넣으세요.
# 먼저 같은 모델 구조를 만들었습니다. 인터넷에서 받은 출처 불명 파일은 읽지 마세요.
# >>> H10
# TODO H10: 위 요구사항을 읽고 이 블록을 완성하세요.
# pass

# 저장했던 state_dict 파일 읽기
# - map_location="cpu" : GPU에서 저장했더라도 CPU로 강제 로드
# - weights_only=True : 가중치만 읽음 //전체 객체를 복원하지 않아 상대적으로 안전
state_dict = torch.load(base / "outputs" / "model_state.pt", map_location="cpu", weights_only=True)
# 읽어온 가중치를 현재 모델에 적용
model.load_state_dict(state_dict)
# <<< H10
# 평가 모드 전환
model.eval()
with torch.no_grad():
    # 예측 수행
    predictions = model(X).argmax(dim=1).tolist()
# 저장된 예측 파일 읽기
with (base / "outputs" / "validation_predictions.csv").open(encoding="utf-8-sig",newline="") as f:
    saved = list(csv.DictReader(f))
print("ID 순서 동일:", [r["id"] for r in valid_rows] == [r["id"] for r in saved])
print("예측 동일:", predictions == [int(r["predicted"]) for r in saved])
print("서로 다른 CPU/GPU의 마지막 비트까지 동일함을 보장하는 검사는 아닙니다.")
