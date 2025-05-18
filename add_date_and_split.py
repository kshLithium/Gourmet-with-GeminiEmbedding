#!/usr/bin/env python
# coding: utf-8
"""
날짜 필드 추가 및 랜덤 분할 스크립트
-----------------------------------
1. 원본 review.json에서 review_id -> date 매핑 생성
2. review_5up_5aspect_3sentiment.jsonl 파일에 date 필드 추가
3. 결합 후 80/20으로 랜덤 분할
4. train_80_random_with_date.json, test_20_random_with_date.json 생성
"""
import json
import random
from pathlib import Path
from tqdm import tqdm

# 파일 경로
DATE_SOURCE_FILE = "dataset/review.json"  # date 정보를 가져올 원본 파일
TARGET_FILE = (
    "dataset/review_5up_5aspect_3sentiment_with_ids.jsonl"  # date 필드를 추가할 파일
)
OUTPUT_TRAIN = "dataset/train_80_random_with_date.json"
OUTPUT_TEST = "dataset/test_20_random_with_date.json"


def main():
    print("📅 원본 데이터에서 review_id -> date 매핑 구축 중...")
    review_id_to_date = {}

    # review.json에서 review_id와 date 매핑 생성
    with open(DATE_SOURCE_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="원본 처리 중"):
            review = json.loads(line)
            review_id_to_date[review["review_id"]] = review["date"]

    print(f"🔍 매핑된 review ID 수: {len(review_id_to_date)}")

    # 대상 파일에 date 필드 추가
    all_reviews = []

    print("🔄 리뷰 데이터에 날짜 추가 중...")
    with open(TARGET_FILE, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="리뷰 데이터"):
            review = json.loads(line)
            review_id = review["review_id"]
            if review_id in review_id_to_date:
                review["date"] = review_id_to_date[review_id]
                all_reviews.append(review)
            else:
                print(f"⚠️ 리뷰 ID를 찾을 수 없음: {review_id}")

    print(f"✅ 날짜가 추가된 총 리뷰 수: {len(all_reviews)}")

    # 데이터 랜덤 셔플 및 80/20 분할
    random.shuffle(all_reviews)
    split_idx = int(len(all_reviews) * 0.8)
    train_data = all_reviews[:split_idx]
    test_data = all_reviews[split_idx:]

    # 새 파일 저장
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for review in tqdm(train_data, desc="훈련 데이터 저장 중"):
            f.write(json.dumps(review) + "\n")

    with open(OUTPUT_TEST, "w", encoding="utf-8") as f:
        for review in tqdm(test_data, desc="테스트 데이터 저장 중"):
            f.write(json.dumps(review) + "\n")

    print(f"📊 결과 요약:")
    print(f"  - 원본 매핑 리뷰 ID 수: {len(review_id_to_date)}")
    print(f"  - 총 처리된 리뷰 수: {len(all_reviews)}")
    print(f"  - 새 훈련 데이터 수: {len(train_data)} ({OUTPUT_TRAIN})")
    print(f"  - 새 테스트 데이터 수: {len(test_data)} ({OUTPUT_TEST})")
    print(f"✨ 처리 완료!")


if __name__ == "__main__":
    main()
