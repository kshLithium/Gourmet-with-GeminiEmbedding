#!/usr/bin/env python
# coding: utf-8

import json
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# 설정
INPUT_FILE = "dataset/review_5up_5aspect_3sentiment_vectorized_clean.json"
TRAIN_FILE_RANDOM = "dataset/train_80_random.json"
TEST_FILE_RANDOM = "dataset/test_20_random.json"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def load_data(file_path):
    """데이터 파일 로드"""
    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    print("✨ 단순 랜덤 방식으로 데이터 스플릿 시작")

    # 데이터 로드
    print(f"📂 {INPUT_FILE} 파일 로드 중...")
    all_data = load_data(INPUT_FILE)
    print(f"📊 전체 데이터 수: {len(all_data)}")

    # 데이터를 랜덤하게 섞기
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.shuffle(all_data)

    # 8:2 비율로 분할
    split_idx = int(len(all_data) * TRAIN_RATIO)
    train_data = all_data[:split_idx]
    test_data = all_data[split_idx:]

    # 결과 저장
    with open(TRAIN_FILE_RANDOM, "w", encoding="utf-8") as f:
        for r in train_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(TEST_FILE_RANDOM, "w", encoding="utf-8") as f:
        for r in test_data:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 통계 정보
    train_users = set(item["user_id"] for item in train_data)
    test_users = set(item["user_id"] for item in test_data)
    common_users = train_users.intersection(test_users)

    print(f"✅ 훈련 데이터 개수: {len(train_data)} → {TRAIN_FILE_RANDOM}")
    print(f"✅ 평가 데이터 개수: {len(test_data)} → {TEST_FILE_RANDOM}")
    print(f"👥 훈련 데이터 유저 수: {len(train_users)}")
    print(f"👥 평가 데이터 유저 수: {len(test_users)}")
    print(
        f"🔄 훈련/평가 공통 유저 수: {len(common_users)} ({len(common_users)/len(train_users)*100:.1f}%)"
    )


if __name__ == "__main__":
    main()
