#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
간단한 Item-based CF 추천 시스템
================================
* 데이터: review_5up.json 사용 (user_id, business_id, stars)
* 알고리즘: Item-KNN CF
* 평가: Leave-One-Out으로 Precision@5, Recall@5, NDCG@5 측정
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_rating(path):
    """평점 데이터를 로드합니다."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append(
                {"user": d["user_id"], "biz": d["business_id"], "stars": d["stars"]}
            )
    return pd.DataFrame(rows)


def leave_one_out(ratings):
    """Leave-One-Out 방식으로 학습/테스트 데이터를 분리합니다."""
    np.random.seed(42)
    train_idx = []
    test = {}  # user->biz

    for u, grp in ratings.groupby("user"):
        idx = grp.index.values
        if len(idx) > 1:
            # 하나의 아이템을 테스트 셋으로
            hold = np.random.choice(idx)
            test[u] = ratings.loc[hold, "biz"]
            train_idx.extend([i for i in idx if i != hold])
        else:
            train_idx.extend(idx)

    return ratings.loc[train_idx].reset_index(drop=True), test


def build_item_user_maps(ratings):
    """아이템-사용자 및 사용자-아이템 맵을 구성합니다."""
    item_users = defaultdict(dict)
    user_items = defaultdict(dict)

    for _, r in ratings.iterrows():
        item_users[r["biz"]][r["user"]] = r["stars"]
        user_items[r["user"]][r["biz"]] = r["stars"]

    return item_users, user_items


def cosine_similarity(a, b):
    """두 벡터 간의 코사인 유사도를 계산합니다."""
    common = set(a) & set(b)
    if not common:
        return 0.0

    num = sum(a[u] * b[u] for u in common)
    den = (
        math.sqrt(sum(v * v for v in a.values()))
        * math.sqrt(sum(v * v for v in b.values()))
        + 1e-9
    )
    return num / den


def precompute_item_sim(item_users):
    """모든 아이템 쌍 간의 유사도를 미리 계산합니다."""
    items = list(item_users)
    sims = defaultdict(dict)

    for i, a in tqdm(enumerate(items), total=len(items), desc="아이템 유사도 계산"):
        for b in items[i + 1 :]:
            s = cosine_similarity(item_users[a], item_users[b])
            if s > 0:
                sims[a][b] = s
                sims[b][a] = s

    return sims


def recommend(user, user_items, item_sims, n=5):
    """사용자에게 추천할 상위 N개 아이템을 반환합니다."""
    # 이미 본 아이템은 제외
    seen = set(user_items[user].keys())

    # 예측 점수 계산
    scores = defaultdict(float)
    for item, rating in user_items[user].items():
        for similar_item, sim in item_sims.get(item, {}).items():
            if similar_item not in seen:
                scores[similar_item] += sim * rating

    # 점수 기준 내림차순 정렬
    return [item for item, _ in sorted(scores.items(), key=lambda x: -x[1])[:n]]


def precision_at_k(test, recs, k=5):
    """Precision@k 계산"""
    hit = sum(1 for u, gt in test.items() if gt in recs.get(u, [])[:k])
    return hit / len(test) if test else 0.0


def recall_at_k(test, recs, k=5):
    """Recall@k 계산 (Leave-One-Out에서는 Precision과 동일)"""
    return precision_at_k(test, recs, k)


def ndcg_at_k(test, recs, k=5):
    """NDCG@k 계산"""
    total = 0.0
    for u, gt in test.items():
        rec_list = recs.get(u, [])[:k]
        if gt in rec_list:
            idx = rec_list.index(gt)
            # Gain = 1, Discount = log2(idx+2)
            total += 1 / math.log2(idx + 2)
    return total / len(test) if test else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rating", default="dataset/review_5up.json")
    p.add_argument("--topn", type=int, default=5)
    p.add_argument("--min_ratings", type=int, default=5, help="사용자별 최소 평점 수")
    args = p.parse_args()

    print(f"데이터 로드 중: {args.rating}")
    ratings = load_rating(args.rating)

    # 최소 평점 수 필터링
    user_counts = ratings.groupby("user").size()
    valid_users = user_counts[user_counts >= args.min_ratings].index
    ratings = ratings[ratings["user"].isin(valid_users)]

    print(
        f"총 {len(ratings):,}개 평점, {ratings['user'].nunique():,}명 사용자, {ratings['biz'].nunique():,}개 아이템"
    )

    print("학습/테스트 데이터 분리 중...")
    train, test = leave_one_out(ratings)
    print(f"학습: {len(train):,}개, 테스트: {len(test):,}개")

    print("아이템-사용자 맵 구성 중...")
    item_users, user_items = build_item_user_maps(train)

    print("아이템 유사도 계산 중...")
    item_sims = precompute_item_sim(item_users)

    print(f"상위 {args.topn}개 아이템 추천 중...")
    recommendations = {}
    for user in tqdm(test.keys()):
        if user in user_items and len(user_items[user]) > 0:
            recommendations[user] = recommend(user, user_items, item_sims, args.topn)

    print("성능 평가 중...")
    p5 = precision_at_k(test, recommendations, 5)
    r5 = recall_at_k(test, recommendations, 5)
    n5 = ndcg_at_k(test, recommendations, 5)

    print(f"Precision@5: {p5:.4f}")
    print(f"Recall@5: {r5:.4f}")
    print(f"NDCG@5: {n5:.4f}")


if __name__ == "__main__":
    main()
