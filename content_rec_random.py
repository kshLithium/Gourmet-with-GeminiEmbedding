#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score
from tqdm import tqdm

# 랜덤 스플릿 파일 사용
TRAIN_FILE = "dataset/train_80_random.json"
TEST_FILE = "dataset/test_20_random.json"
TOP_N_LIST = [5, 10]  # 평가할 TOP_N 값 목록


def load_reviews(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            obj["vector"] = np.array(obj["sentiment_vector"])
            data.append(obj)
    return data


def compute_ndcg(pred, actual, k):
    """
    nDCG@K 계산 함수
    """
    if not actual:
        return 0.0

    # 예측 아이템이 실제 아이템에 있는지 리스트로 변환 (1: 있음, 0: 없음)
    relevance = np.array([1.0 if item in actual else 0.0 for item in pred[:k]])

    # DCG 계산
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))

    # IDCG 계산 (이상적인 경우)
    ideal_relevance = np.ones(min(len(actual), k))
    idcg = np.sum(ideal_relevance / np.log2(np.arange(2, len(ideal_relevance) + 2)))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def main():
    print("🔄 랜덤 스플릿 데이터로 추천 모델 평가 시작")

    # 훈련 및 테스트 데이터 로드
    train_reviews = load_reviews(TRAIN_FILE)
    test_reviews = load_reviews(TEST_FILE)

    print(f"📚 훈련 데이터 로드 완료: {len(train_reviews)}개")
    print(f"🧪 테스트 데이터 로드 완료: {len(test_reviews)}개")

    # 훈련 데이터로부터 사용자 및 비즈니스 벡터 계산
    user_vecs = defaultdict(list)
    biz_vecs = defaultdict(list)

    for r in train_reviews:
        user_vecs[r["user_id"]].append(r["vector"])
        biz_vecs[r["business_id"]].append(r["vector"])

    user_embed = {u: np.mean(vs, axis=0) for u, vs in user_vecs.items()}
    biz_embed = {b: np.mean(vs, axis=0) for b, vs in biz_vecs.items()}

    print(f"👤 사용자 임베딩 생성 완료: {len(user_embed)}명")
    print(f"🏪 식당 임베딩 생성 완료: {len(biz_embed)}개")

    # 정규화 및 유사도 계산
    user_ids = list(user_embed.keys())
    biz_ids = list(biz_embed.keys())

    user_matrix = normalize(np.stack([user_embed[u] for u in user_ids]))
    biz_matrix = normalize(np.stack([biz_embed[b] for b in biz_ids]))

    scores = np.dot(user_matrix, biz_matrix.T)
    print("✅ 유사도 계산 완료")

    # 추천 리스트 생성 (최대 TOP_N_LIST의 최대값까지)
    user2seen = defaultdict(set)
    for r in train_reviews:
        user2seen[r["user_id"]].add(r["business_id"])

    max_k = max(TOP_N_LIST)
    recommendations = {}

    print("🔍 추천 목록 생성 중...")
    for i, uid in enumerate(tqdm(user_ids)):
        user_score = scores[i]
        ranked_idx = np.argsort(user_score)[::-1]

        recs = []
        for j in ranked_idx:
            bid = biz_ids[j]
            if bid not in user2seen[uid]:  # 이미 본 식당은 제외
                recs.append(bid)
            if len(recs) == max_k:
                break
        recommendations[uid] = recs

    # 평가: Ground Truth 생성
    ground_truth = defaultdict(set)
    for r in test_reviews:
        ground_truth[r["user_id"]].add(r["business_id"])

    # 공통 사용자 찾기
    common_users = set(recommendations.keys()) & set(ground_truth.keys())

    print(f"📌 평가 대상 유저 수: {len(common_users)}")

    # 다양한 K 값에 대해 평가 지표 계산
    for k in TOP_N_LIST:
        precision_list = []
        recall_list = []
        ndcg_list = []

        for uid in common_users:
            pred = recommendations[uid][:k]  # TOP_K까지만 사용
            actual = ground_truth[uid]

            # Precision@K
            hit = len(set(pred) & actual)
            precision = hit / k if k > 0 else 0
            precision_list.append(precision)

            # Recall@K
            recall = hit / len(actual) if actual else 0
            recall_list.append(recall)

            # nDCG@K
            ndcg = compute_ndcg(pred, actual, k)
            ndcg_list.append(ndcg)

        # 결과 출력
        print(f"\n===== K = {k} 평가 지표 =====")
        print(f"🎯 Precision@{k}: {np.mean(precision_list):.4f}")
        print(f"🔍 Recall@{k}: {np.mean(recall_list):.4f}")
        print(f"📊 nDCG@{k}: {np.mean(ndcg_list):.4f}")


if __name__ == "__main__":
    main()
