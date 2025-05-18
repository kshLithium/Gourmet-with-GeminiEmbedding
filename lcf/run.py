import json
import os
from tqdm import tqdm
from pyabsa import ATEPCCheckpointManager
import torch
import psutil
import GPUtil
import numpy as np


def print_memory_status():
    # GPU 메모리 상태
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


# === 모델 로드 ===
aspect_extractor = ATEPCCheckpointManager.get_aspect_extractor(checkpoint="english")
print("\n=== 모델 로드 후 메모리 상태 ===")
print_memory_status()

# === 파일 경로 ===
input_file = "review_5up.json"
output_file = "absa_atepc_results.jsonl"

# === 버퍼 및 설정 ===
buffer = []
save_every = 1000
batch_size = 128
count = 0

# === 중복 방지: 이미 처리한 리뷰 불러오기 ===
done_texts = set()
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                done_texts.add(item["text"])
            except:
                continue
    print(f"\n=== 처리 현황 ===")
    print(f"이미 처리된 리뷰 수: {len(done_texts)}개")
    print(f"이제부터 새로운 리뷰를 처리합니다.")

# === 리뷰 분석 + 중간 저장 ===
with open(input_file, "r", encoding="utf-8") as f_in:
    # 전체 리뷰 수 계산
    total_reviews = sum(1 for _ in f_in)
    f_in.seek(0)  # 파일 포인터를 다시 처음으로

    batch_texts = []
    batch_items = []

    for line in tqdm(f_in, desc="ABSA 추론 중"):
        try:
            item = json.loads(line)
            text = item["text"]

            if text in done_texts:
                continue

            batch_texts.append(text)
            batch_items.append(item)

            # 배치 크기에 도달하면 처리
            if len(batch_texts) >= batch_size:
                # # numpy 배열을 미리 변환하여 tensor 생성 속도 개선
                # batch_texts = np.array(batch_texts)

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
                    buffer.append(final_result)
                    count += 1

                # 중간 저장
                if len(buffer) >= save_every:
                    with open(output_file, "a", encoding="utf-8") as f_out:
                        for r in buffer:
                            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                    buffer = []

                    # 진행 상황 출력
                    print(f"\n=== 처리 현황 ===")
                    print(f"처리된 리뷰: {count}개")
                    print(f"남은 리뷰: {total_reviews - count}개")
                    print(f"전체 진행률: {(count/total_reviews)*100:.1f}%")

                    # 500개마다 메모리 상태 출력
                    print(f"\n=== {count}개 처리 후 메모리 상태 ===")
                    print_memory_status()

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
            buffer.append(final_result)
            count += 1

# === 마지막 남은 버퍼 저장 ===
if buffer:
    with open(output_file, "a", encoding="utf-8") as f_out:
        for r in buffer:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\n=== 최종 메모리 상태 ===")
print_memory_status()
