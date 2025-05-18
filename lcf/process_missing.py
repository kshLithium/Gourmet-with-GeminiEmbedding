import json
from tqdm import tqdm
from pyabsa import ATEPCCheckpointManager
import torch

# === 모델 로드 ===
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint="english")

# === 파일 경로 ===
input_file = "review_5up.json"
output_file = "absa_atepc_results.jsonl"

# === 이미 처리된 리뷰 ID 수집 ===
processed_reviews = set()
with open(output_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
            processed_reviews.add((item["review_id"], item["text"]))
        except:
            continue

print(f"이미 처리된 리뷰 수: {len(processed_reviews)}")

# === 리뷰 처리 ===
batch_size = 128
batch_texts = []
batch_items = []

with open(input_file, "r", encoding="utf-8") as f_in, open(
    output_file, "a", encoding="utf-8"
) as f_out:
    for line in tqdm(f_in, desc="누락된 리뷰 처리 중"):
        try:
            item = json.loads(line)
            review_id = item.get("review_id")
            text = item["text"]

            # 이미 처리된 리뷰는 건너뛰기
            if (review_id, text) in processed_reviews:
                continue

            batch_texts.append(text)
            batch_items.append(item)

            if len(batch_texts) >= batch_size:
                # 배치 처리
                results = aspect_extractor.extract_aspect(
                    inference_source=batch_texts,
                    print_result=False,
                    pred_sentiment=True,
                    save_result=False,
                )

                # 결과 처리
                for item, result in zip(batch_items, results):
                    final_result = {
                        "review_id": item.get("review_id"),
                        "user_id": item.get("user_id"),
                        "business_id": item.get("business_id"),
                        "text": item["text"],
                        "aspects": [
                            {"term": t, "sentiment": s}
                            for t, s in zip(result["aspect"], result["sentiment"])
                        ],
                    }
                    f_out.write(json.dumps(final_result, ensure_ascii=False) + "\n")

                # 배치 초기화
                batch_texts = []
                batch_items = []

        except Exception as e:
            print(f"에러 발생: {str(e)}")
            continue

    # 남은 배치 처리
    if batch_texts:
        results = aspect_extractor.extract_aspect(
            inference_source=batch_texts,
            print_result=False,
            pred_sentiment=True,
            save_result=False,
        )

        for item, result in zip(batch_items, results):
            final_result = {
                "review_id": item.get("review_id"),
                "user_id": item.get("user_id"),
                "business_id": item.get("business_id"),
                "text": item["text"],
                "aspects": [
                    {"term": t, "sentiment": s}
                    for t, s in zip(result["aspect"], result["sentiment"])
                ],
            }
            f_out.write(json.dumps(final_result, ensure_ascii=False) + "\n")

print("처리 완료!")
