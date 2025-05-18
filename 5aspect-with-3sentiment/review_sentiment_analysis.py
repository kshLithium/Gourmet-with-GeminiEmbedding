import json
import torch
import tqdm
import os
import psutil
import GPUtil
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# 메모리 상태 출력 함수
def print_memory_status():
    try:
        # GPU 메모리 상태
        if torch.cuda.is_available():
            gpu = GPUtil.getGPUs()[0]  # 첫 번째 GPU 정보 가져오기
            print(f"\n=== GPU 메모리 상태 ===")
            print(f"GPU 메모리 사용량: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
            print(f"GPU 메모리 사용률: {gpu.memoryUtil*100:.1f}%")
            print(f"GPU 온도: {gpu.temperature}°C")

            # PyTorch 메모리 상태
            print(f"\n=== PyTorch 메모리 상태 ===")
            print(f"할당된 메모리: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
            print(f"캐시된 메모리: {torch.cuda.memory_reserved()/1024**2:.1f}MB")

        # 시스템 메모리 상태
        print(f"\n=== 시스템 메모리 상태 ===")
        memory = psutil.virtual_memory()
        print(
            f"시스템 메모리 사용량: {memory.used/1024**2:.1f}MB / {memory.total/1024**2:.1f}MB"
        )
        print(f"시스템 메모리 사용률: {memory.percent}%")
    except Exception as e:
        print(f"메모리 상태 확인 중 오류 발생: {e}")


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

print("\n=== 모델 로드 후 메모리 상태 ===")
print_memory_status()

# 분석할 측면
aspects = ["food", "service", "price", "ambience", "location"]


# 감정 분석 수행
def analyze_sentiment(text, aspect):
    try:
        # 토큰화 (max_length 제거)
        inputs = tokenizer(text, aspect, return_tensors="pt")
        # 입력 텐서를 GPU로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 모델 실행
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]

        # 각 감정 점수
        sentiment_scores = {}
        for i, label in model.config.id2label.items():
            sentiment_scores[label] = float(
                probs[i].item()
            )  # JSON 직렬화를 위해 float로 변환

        return {"scores": sentiment_scores, "status": "success"}
    except Exception as e:
        print(f"감정 분석 중 오류 발생: {e} (텍스트: {text[:50]}...)")
        # null 값 반환 (계산에 반영되지 않게)
        return {"scores": None, "status": "error", "error_message": str(e)}


# 리뷰 처리 및 JSONL 형식으로 저장
def process_reviews(input_file, output_file, batch_size=100, save_every=500):
    # 이미 처리한 리뷰 ID 불러오기
    processed_ids = set()
    if os.path.exists(output_file):
        print(f"기존 결과 파일 {output_file}에서 처리된 리뷰 ID 불러오는 중...")
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    processed_ids.add(item["review_id"])
                except:
                    continue
        print(f"{len(processed_ids)}개의 처리된 리뷰 ID를 불러왔습니다.")

    # 배치 처리를 위한 변수
    buffer = []
    count = 0
    skipped = 0

    try:
        with open(input_file, "r", encoding="utf-8") as f_in:
            for line in tqdm.tqdm(f_in, desc="리뷰 처리 중"):
                try:
                    line = line.strip()
                    if not line:  # 빈 줄 무시
                        continue

                    review = json.loads(line)
                    review_id = review["review_id"]

                    # 이미 처리한 리뷰는 건너뛰기
                    if review_id in processed_ids:
                        skipped += 1
                        if skipped % 10000 == 0:
                            print(f"{skipped}개의 리뷰 건너뜀")
                        continue

                    text = review["text"]

                    # 각 aspect별 감정 분석 결과
                    sentiment_results = {}
                    for aspect in aspects:
                        sentiment_results[aspect] = analyze_sentiment(text, aspect)

                    # 결과 저장
                    result = {
                        "review_id": review_id,
                        "text": text,
                        "sentiment": sentiment_results,
                    }
                    buffer.append(result)
                    count += 1

                    # 일정 개수마다 저장
                    if len(buffer) >= save_every:
                        # 파일에 추가 모드로 저장
                        with open(output_file, "a", encoding="utf-8") as f_out:
                            for r in buffer:
                                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

                        print(f"{count}개의 리뷰 처리 완료 ({skipped}개 건너뜀)")
                        print_memory_status()

                        # 버퍼 초기화
                        buffer = []

                except Exception as e:
                    print(f"리뷰 처리 중 오류 발생: {e}")
                    continue

    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")

    finally:
        # 남은 결과 저장
        if buffer:
            with open(output_file, "a", encoding="utf-8") as f_out:
                for r in buffer:
                    f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"총 {count}개의 리뷰 처리 완료 ({skipped}개 건너뜀)")
    print(f"결과가 {output_file}에 저장되었습니다.")
    print("\n=== 최종 메모리 상태 ===")
    print_memory_status()


# 메인 실행 함수
def main():
    input_file = "../dataset/review_5up.json"
    output_file = (
        "../dataset/review_5up_5aspect_3sentiment.jsonl"  # JSONL 형식으로 변경
    )

    print(f"리뷰 파일: {input_file}")
    print(f"결과 파일: {output_file}")

    # 배치 단위로 처리
    process_reviews(input_file, output_file, batch_size=100, save_every=500)


if __name__ == "__main__":
    main()
