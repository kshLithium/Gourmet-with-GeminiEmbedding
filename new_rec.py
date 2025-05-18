#!/usr/bin/env python
# coding: utf-8
"""
ABSA Random‑Split Recommender — Recency‑Weighted µ∥σ² Pooling
------------------------------------------------------------
피드백 반영 사항
1. **단순 평균 → (최근성 가중) 평균 + 분산(µ ∥ σ²) 벡터**
2. **TF‑IDF 제거** → 시간 감쇠(half‑life)만 사용
3. 나머지 파이프라인(랜덤 스플릿, user2seen 필터, 평가 루프) 유지
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import precision_score  # (남아 있지만 사용 안 함)
from sklearn.preprocessing import normalize
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
TRAIN_FILE = "dataset/train_80_random_with_date.json"
TEST_FILE = "dataset/test_20_random_with_date.json"
TOP_N_LIST = [5, 10]
HALF_LIFE_DAYS = 180  # 최근성 half‑life (≈ 6 개월)

# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------


def load_reviews(path: str):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)

            # sentiment 객체에서 벡터 생성 (5 aspects x 3 sentiments = 15 dimension)
            if "sentiment" in obj and "sentiment_vector" not in obj:
                sent_vector = []
                aspects = ["food", "service", "price", "ambience", "location"]
                sentiments = ["Negative", "Neutral", "Positive"]

                for aspect in aspects:
                    if aspect in obj["sentiment"]:
                        for sentiment in sentiments:
                            sent_vector.append(
                                obj["sentiment"][aspect]["scores"][sentiment]
                            )

                obj["sentiment_vector"] = sent_vector

            # 기존 벡터 필드 사용
            obj["vector"] = np.array(obj["sentiment_vector"], dtype=float)

            # 날짜 파싱 (ISO 8601 or "YYYY-MM-DD")
            obj["date"] = datetime.fromisoformat(obj["date"])
            data.append(obj)
    return data


def recency_weight(days: int) -> float:
    """지수 감쇠 가중치 (half‑life = HALF_LIFE_DAYS)."""
    if days < 0:
        days = 0
    return np.exp(-np.log(2) * days / HALF_LIFE_DAYS)


def agg_mu_sigma(rows: List[Tuple[np.ndarray, datetime]]) -> np.ndarray:
    """최근성 가중 평균(µ) + 분산(σ²) concat 반환."""
    if not rows:
        return np.zeros(10)  # placeholder; 실제 dim은 동적으로 처리 아래 참고

    now = datetime.now()
    vecs, ws = [], []
    for v, ts in rows:
        w = recency_weight((now - ts).days)
        vecs.append(v)
        ws.append(w)
    V = np.vstack(vecs)  # (n, d)
    W = np.array(ws).reshape(-1, 1)  # (n, 1)

    mu = (W * V).sum(axis=0) / W.sum()
    var = (W * (V - mu) ** 2).sum(axis=0) / W.sum()

    return np.concatenate([mu, var])


# ---------------------------------------------------------------------------
# nDCG 계산
# ---------------------------------------------------------------------------


def compute_ndcg(pred: List[str], actual: set, k: int) -> float:
    if not actual:
        return 0.0
    relevance = np.array([1.0 if bid in actual else 0.0 for bid in pred[:k]])
    dcg = np.sum(relevance / np.log2(np.arange(2, len(relevance) + 2)))
    ideal = np.ones(min(len(actual), k))
    idcg = np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2)))
    return dcg / idcg if idcg else 0.0


# ---------------------------------------------------------------------------
# 메인 파이프라인
# ---------------------------------------------------------------------------


def main():
    print("🔄 랜덤 스플릿 데이터로 추천 모델 평가 시작")

    # 데이터 로드
    train_reviews = load_reviews(TRAIN_FILE)
    test_reviews = load_reviews(TEST_FILE)
    print(
        f"📚 훈련 데이터: {len(train_reviews)}개,  🧪 테스트 데이터: {len(test_reviews)}개"
    )

    # --------------------------------------------------
    # 사용자·비즈니스 벡터 집계 (µ∥σ² pooling)
    # --------------------------------------------------
    user_rows = defaultdict(list)  # uid → [(vec, date), ...]
    biz_rows = defaultdict(list)  # bid → [(vec, date), ...]

    for r in train_reviews:
        user_rows[r["user_id"]].append((r["vector"], r["date"]))
        biz_rows[r["business_id"]].append((r["vector"], r["date"]))

    user_embed = {u: agg_mu_sigma(rows) for u, rows in user_rows.items()}
    biz_embed = {b: agg_mu_sigma(rows) for b, rows in biz_rows.items()}

    print(f"👤 사용자 임베딩: {len(user_embed)}명,  🏪 식당 임베딩: {len(biz_embed)}개")

    # 차원 확인 & 0‑벡터 제거
    user_embed = {u: v for u, v in user_embed.items() if np.linalg.norm(v) > 0}
    biz_embed = {b: v for b, v in biz_embed.items() if np.linalg.norm(v) > 0}

    user_ids = list(user_embed.keys())
    biz_ids = list(biz_embed.keys())

    user_matrix = normalize(np.stack([user_embed[u] for u in user_ids]))
    biz_matrix = normalize(np.stack([biz_embed[b] for b in biz_ids]))

    scores = user_matrix @ biz_matrix.T
    print("✅ 유사도 행렬 계산 완료")

    # --------------------------------------------------
    # 추천 리스트 생성
    # --------------------------------------------------
    user2seen = defaultdict(set)
    for r in train_reviews:
        user2seen[r["user_id"]].add(r["business_id"])

    max_k = max(TOP_N_LIST)
    recommendations = {}

    print("🔍 추천 목록 생성 중…")
    for i, uid in enumerate(tqdm(user_ids)):
        ranked_idx = np.argsort(scores[i])[::-1]
        recs = []
        for j in ranked_idx:
            bid = biz_ids[j]
            if bid not in user2seen[uid]:  # 재방문 제외 로직 유지
                recs.append(bid)
            if len(recs) == max_k:
                break
        recommendations[uid] = recs

    # --------------------------------------------------
    # 평가
    # --------------------------------------------------
    ground_truth = defaultdict(set)
    for r in test_reviews:
        ground_truth[r["user_id"]].add(r["business_id"])

    common_users = set(recommendations) & set(ground_truth)
    print(f"📌 평가 대상 유저: {len(common_users)}명")

    for k in TOP_N_LIST:
        precisions, recalls, ndcgs = [], [], []
        for uid in common_users:
            pred = recommendations[uid][:k]
            actual = ground_truth[uid]
            hit = len(set(pred) & actual)
            precisions.append(hit / k)
            recalls.append(hit / len(actual) if actual else 0)
            ndcgs.append(compute_ndcg(pred, actual, k))
        print(f"\n===== K = {k} 평가 지표 =====")
        print(f"🎯 Precision@{k}: {np.mean(precisions):.4f}")
        print(f"🔍 Recall@{k}:    {np.mean(recalls):.4f}")
        print(f"📊 nDCG@{k}:      {np.mean(ndcgs):.4f}")


if __name__ == "__main__":
    main()
