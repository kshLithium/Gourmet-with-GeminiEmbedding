import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# 모델과 토크나이저 로드
model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 모델을 GPU로 이동
model = model.to(device)
model.eval()

# 입력 문장
sentence = "I am a huge fan of Gennaro's. It's got a thin crust, sweet and rich tomato sauce along w fresh mozzarella cheese. Also, the meatballs are excellent. Can't say the same about the service though."

# 분석할 측면
aspects = ["food", "service", "price", "ambience", "location"]

for aspect in aspects:
    # 토큰화
    inputs = tokenizer(sentence, aspect, return_tensors="pt")
    # 입력 텐서를 GPU로 이동
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 모델 실행
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

    # 모든 감정 점수 출력
    print(f"\n[{aspect}] 감정 분석 결과:")
    for i, label in model.config.id2label.items():
        score = probs[i].item()
        print(f"  {label}: {score:.3f}")

    # 최대 점수 감정
    pred_idx = probs.argmax().item()
    pred_label = model.config.id2label[pred_idx]
    confidence = probs[pred_idx].item()
    print(f"  최종 판단: {pred_label} ({confidence:.3f})")
