#!/usr/bin/env python
# coding: utf-8

# #### import

# In[2]:


import json
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
import random
import pandas as pd
from scipy.sparse import csr_matrix


# ### JSONL 파일에서 리뷰 로드

# In[3]:


TRAIN_FILE = "dataset/train_80.json"
TEST_FILE = "dataset/test_20.json"
TOP_N = 5
# 메모리 효율성을 위해 샘플링 비율 설정
SAMPLE_RATIO = 0.2  # 20%의 유저만 사용
USE_CF = True  # 협업 필터링 사용 여부
USE_CATEGORY = True  # 카테고리 기반 필터링 사용 여부


def load_reviews(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            obj["vector"] = np.array(obj["sentiment_vector"])
            data.append(obj)
    return data


print("리뷰 데이터 로드 중...")
train_reviews = load_reviews(TRAIN_FILE)
test_reviews = load_reviews(TEST_FILE)

# 유저 샘플링 (메모리 효율성을 위해)
all_user_ids = list(set(r["user_id"] for r in train_reviews))
sampled_user_ids = set(
    random.sample(all_user_ids, int(len(all_user_ids) * SAMPLE_RATIO))
)

print(
    f"총 {len(all_user_ids)}명의 유저 중 {len(sampled_user_ids)}명 샘플링 (메모리 효율성 위해)"
)

# 샘플링된 유저의 리뷰만 필터링 (훈련 데이터)
sampled_train_reviews = [r for r in train_reviews if r["user_id"] in sampled_user_ids]

print(f"샘플링 후 훈련 리뷰 수: {len(sampled_train_reviews)}/{len(train_reviews)}")

# 카테고리 정보 모의 추가 (메모리 효율성을 위해 필요한 식당만)
if USE_CATEGORY:
    print("식당 카테고리 정보 생성 중...")
    business_ids = set()
    for r in sampled_train_reviews:
        business_ids.add(r["business_id"])

    # 테스트 데이터에서 샘플링된 유저의 식당 추가
    for r in test_reviews:
        if r["user_id"] in sampled_user_ids:
            business_ids.add(r["business_id"])

    # 5개의 카테고리 생성
    categories = ["카테고리_" + str(i) for i in range(5)]
    biz_categories = {}
    for bid in business_ids:
        # 각 식당은 1~2개의 랜덤 카테고리를 가짐
        n_cats = random.randint(1, 2)
        biz_categories[bid] = random.sample(categories, n_cats)

    print(f"{len(business_ids)}개 식당에 카테고리 할당 완료")
else:
    business_ids = set()
    for r in sampled_train_reviews:
        business_ids.add(r["business_id"])
    print("카테고리 기반 필터링 비활성화")


# ### 1. 콘텐츠 기반 필터링 (기존 방식)

# #### 훈련 데이터 → 벡터 평균 계산

# In[3]:

print("\n[1] 콘텐츠 기반 필터링 준비 중...")

# 콘텐츠 기반 벡터 표현
user_vecs = defaultdict(list)
biz_vecs = defaultdict(list)

for r in sampled_train_reviews:
    user_vecs[r["user_id"]].append(r["vector"])
    biz_vecs[r["business_id"]].append(r["vector"])

user_embed = {u: np.mean(vs, axis=0) for u, vs in user_vecs.items()}
biz_embed = {b: np.mean(vs, axis=0) for b, vs in biz_vecs.items()}

# 가중치를 적용한 고급 벡터 표현 (특성 중요도 조정)
# 일반적으로 Positive 감성이 추천에 더 영향력이 큼
aspect_weights = np.ones(15)  # 15차원 감성 벡터 (5 aspects x 3 sentiments)
# Positive 감성 가중치 높임 (인덱스: 2, 5, 8, 11, 14)
for i in [2, 5, 8, 11, 14]:
    aspect_weights[i] = 2.5
# Negative 감성 가중치 약간 높임 (인덱스: 0, 3, 6, 9, 12)
for i in [0, 3, 6, 9, 12]:
    aspect_weights[i] = 1.5

# 가중치가 적용된 벡터 계산
user_embed_weighted = {}
biz_embed_weighted = {}

for u, vec in user_embed.items():
    user_embed_weighted[u] = vec * aspect_weights

for b, vec in biz_embed.items():
    biz_embed_weighted[b] = vec * aspect_weights


# #### 정규화 후 유사도 계산

# In[4]:

print("콘텐츠 유사도 계산 중...")

user_ids = list(user_embed_weighted.keys())
biz_ids = list(biz_embed_weighted.keys())

user_matrix = normalize(np.stack([user_embed_weighted[u] for u in user_ids]))
biz_matrix = normalize(np.stack([biz_embed_weighted[b] for b in biz_ids]))

scores_content = np.dot(user_matrix, biz_matrix.T)


# ### 2. 인기도 기반 추천 (베이스라인)

# In[5]:

print("\n[2] 인기도 기반 추천 준비 중...")

# 각 식당별 리뷰 수 계산
biz_popularity = defaultdict(int)
for r in sampled_train_reviews:
    biz_popularity[r["business_id"]] += 1

# 인기도 순으로 식당 정렬
popular_items = sorted(biz_popularity.items(), key=lambda x: x[1], reverse=True)
popular_biz_ids = [item[0] for item in popular_items]


# ### 3. 협업 필터링 구현 (메모리 효율 개선)

# In[6]:

scores_cf = {}
if USE_CF:
    print("\n[3] 협업 필터링 준비 중...")

    # 각 사용자가 방문한 식당 집합 생성
    user_items = defaultdict(set)

    for r in sampled_train_reviews:
        user_id = r["user_id"]
        biz_id = r["business_id"]
        user_items[user_id].add(biz_id)

    print(f"유저-아이템 상호작용 생성 완료: {len(user_items)}명의 유저")

    # 간단한 사용자 기반 협업 필터링 (User-based CF)
    # 자카드 유사도 사용 (두 집합의 교집합 / 합집합)
    scores_cf = defaultdict(dict)

    # 인기도 기준 상위 식당만 고려 (더 적은 계산량)
    top_popular_biz_ids = set(popular_biz_ids[:500])  # 상위 500개 식당만 사용

    for user_id in tqdm(sampled_user_ids, desc="협업 필터링 처리 중"):
        user_visited = user_items[user_id]

        # 유사 사용자 찾기 - 무작위 샘플링으로 계산량 감소
        sampled_users = random.sample(
            list(user_items.keys()), min(100, len(user_items))
        )
        similar_users = []

        for other_id in sampled_users:
            if other_id == user_id:
                continue

            other_visited = user_items[other_id]

            # 교집합과 합집합 크기 계산
            intersection = len(user_visited & other_visited)
            union = len(user_visited | other_visited)

            if union > 0 and intersection > 0:  # 유사도가 있는 경우만
                jaccard = intersection / union
                similar_users.append((other_id, jaccard))

        # 유사도 기준 상위 10명만 선택
        similar_users.sort(key=lambda x: x[1], reverse=True)
        top_similar_users = similar_users[:10]

        # 추천 식당 점수 계산
        for biz_id in top_popular_biz_ids:
            if biz_id in user_visited:  # 이미 방문한 곳은 제외
                continue

            score = 0
            for other_id, similarity in top_similar_users:
                if biz_id in user_items[other_id]:
                    score += similarity

            if score > 0:  # 점수가 있는 경우에만 저장
                scores_cf[user_id][biz_id] = score
else:
    print("협업 필터링 비활성화")


# ### 4. 카테고리 기반 필터링

# In[7]:

scores_category = {}
if USE_CATEGORY:
    print("\n[4] 카테고리 기반 필터링 준비 중...")

    # 사용자별 선호 카테고리 계산
    user_category_prefs = defaultdict(lambda: defaultdict(int))

    for r in sampled_train_reviews:
        user_id = r["user_id"]
        biz_id = r["business_id"]
        if biz_id in biz_categories:
            for category in biz_categories[biz_id]:
                user_category_prefs[user_id][category] += 1

    # 사용자 선호 카테고리 정규화
    for user_id, categories in user_category_prefs.items():
        total = sum(categories.values())
        for cat in categories:
            categories[cat] /= total

    # 카테고리 유사도 점수 계산
    scores_category = {}
    for user_id in sampled_user_ids:
        scores_category[user_id] = {}
        user_prefs = user_category_prefs.get(user_id, {})

        for biz_id in business_ids:
            if biz_id in biz_categories:
                score = 0
                for category in biz_categories[biz_id]:
                    score += user_prefs.get(category, 0)
                # 식당의 카테고리 수로 정규화
                score /= max(1, len(biz_categories[biz_id]))
                scores_category[user_id][biz_id] = score
else:
    print("카테고리 기반 필터링 비활성화")


# ### 5. 하이브리드 추천 시스템 구현

# In[8]:

print("\n[5] 하이브리드 추천 시스템 구현 중...")

# 각 추천 방식의 가중치 설정
weights = {}
weights["content"] = 0.4
weights["cf"] = 0.2 if USE_CF else 0
weights["category"] = 0.2 if USE_CATEGORY else 0
weights["popularity"] = 0.2

# 가중치 정규화
total = sum(weights.values())
for k in weights:
    weights[k] /= total

print(f"추천 가중치: {weights}")


# 인기도 기반 추천
def get_popular_recommendations(user_id, exclude_items, k=TOP_N):
    recommendations = []
    for item in popular_biz_ids:
        if item not in exclude_items:
            recommendations.append(item)
            if len(recommendations) >= k:
                break
    return recommendations


# 콘텐츠 기반 추천
def get_content_recommendations(user_id, exclude_items, k=TOP_N):
    if user_id not in user_ids:
        return []

    user_idx = user_ids.index(user_id)
    user_score = scores_content[user_idx]
    ranked_idx = np.argsort(user_score)[::-1]

    recs = []
    for j in ranked_idx:
        bid = biz_ids[j]
        if bid not in exclude_items:
            recs.append(bid)
        if len(recs) == k:
            break
    return recs


# 협업 필터링 기반 추천
def get_cf_recommendations(user_id, exclude_items, k=TOP_N):
    if not USE_CF or user_id not in scores_cf:
        return []

    user_scores = scores_cf[user_id]
    sorted_items = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)

    recs = []
    for bid, _ in sorted_items:
        if bid not in exclude_items:
            recs.append(bid)
        if len(recs) == k:
            break
    return recs


# 카테고리 기반 추천
def get_category_recommendations(user_id, exclude_items, k=TOP_N):
    if not USE_CATEGORY or user_id not in scores_category:
        return []

    user_scores = scores_category[user_id]
    sorted_items = sorted(user_scores.items(), key=lambda x: x[1], reverse=True)

    recs = []
    for bid, _ in sorted_items:
        if bid not in exclude_items:
            recs.append(bid)
        if len(recs) == k:
            break
    return recs


# #### 추천 리스트 생성 (하이브리드)

# In[9]:

print("하이브리드 추천 계산 중...")

user2seen = defaultdict(set)
for r in sampled_train_reviews:
    user2seen[r["user_id"]].add(r["business_id"])

# 최종 하이브리드 추천 리스트
recommendations = {}
recommendations_content = {}  # 콘텐츠 기반 결과 저장 (비교용)
recommendations_cf = {}  # 협업 필터링 결과 저장 (비교용)
recommendations_category = {}  # 카테고리 기반 결과 저장 (비교용)
recommendations_popular = {}  # 인기도 기반 결과 저장 (비교용)

# 추가 k개의 추천 항목을 가져와 점수 부여
EXTRA_K = 20

for user_id in tqdm(sampled_user_ids, desc="하이브리드 추천 생성 중"):
    seen_items = user2seen[user_id]

    # 각 방식별 추천 결과 (상위 EXTRA_K개)
    content_recs = get_content_recommendations(user_id, seen_items, EXTRA_K)
    cf_recs = get_cf_recommendations(user_id, seen_items, EXTRA_K)
    category_recs = get_category_recommendations(user_id, seen_items, EXTRA_K)
    popular_recs = get_popular_recommendations(user_id, seen_items, EXTRA_K)

    # 기존 각 방식별 추천 저장 (비교용, 상위 TOP_N개)
    recommendations_content[user_id] = content_recs[:TOP_N]
    recommendations_cf[user_id] = cf_recs[:TOP_N]
    recommendations_category[user_id] = category_recs[:TOP_N]
    recommendations_popular[user_id] = popular_recs[:TOP_N]

    # 모든 추천 아이템 수집
    all_items = set()
    for items in [content_recs, cf_recs, category_recs, popular_recs]:
        all_items.update(items)

    # 각 아이템별 가중치 합산 점수 계산
    biz_score_map = {}
    for item in all_items:
        score = 0
        # 콘텐츠 기반 점수
        if item in content_recs:
            score += weights["content"] * (1 - content_recs.index(item) / EXTRA_K)
        # 협업 필터링 점수
        if item in cf_recs:
            score += weights["cf"] * (1 - cf_recs.index(item) / EXTRA_K)
        # 카테고리 기반 점수
        if item in category_recs:
            score += weights["category"] * (1 - category_recs.index(item) / EXTRA_K)
        # 인기도 기반 점수
        if item in popular_recs:
            score += weights["popularity"] * (1 - popular_recs.index(item) / EXTRA_K)

        biz_score_map[item] = score

    # 점수에 따라 정렬하여 최종 추천 생성
    hybrid_recs = sorted(biz_score_map.items(), key=lambda x: x[1], reverse=True)[
        :TOP_N
    ]
    recommendations[user_id] = [item[0] for item in hybrid_recs]


# #### 평가: Ground Truth 생성

# In[10]:

print("\n평가 준비 중...")

ground_truth = defaultdict(set)
for r in test_reviews:
    if r["user_id"] in sampled_user_ids:  # 샘플링된 유저만 평가
        ground_truth[r["user_id"]].add(r["business_id"])


# #### 평가 함수 정의

# In[11]:


def evaluate_recommendations(rec_dict, gt_dict, name=""):
    common_users = set(rec_dict.keys()) & set(gt_dict.keys())
    precision_list = []
    hit_rate_list = []
    ndcg_list = []

    for uid in common_users:
        # 추천 리스트
        pred = rec_dict[uid]
        pred_set = set(pred)
        # 실제 방문한 식당들
        actual = gt_dict[uid]

        # Precision@K
        hit = len(pred_set & actual)
        precision = hit / TOP_N
        precision_list.append(precision)

        # Hit Rate@K
        hit_rate = 1 if hit > 0 else 0
        hit_rate_list.append(hit_rate)

        # NDCG@K
        dcg = 0
        for i, item in enumerate(pred):
            if item in actual:
                dcg += 1.0 / np.log2(i + 2)

        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), TOP_N)))
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)

    print(f"📊 {name} 결과 (총 {len(common_users)}명 평가):")
    print(f"🎯 Precision@{TOP_N}: {np.mean(precision_list):.4f}")
    print(f"🎯 Hit Rate@{TOP_N}: {np.mean(hit_rate_list):.4f}")
    print(f"🎯 NDCG@{TOP_N}: {np.mean(ndcg_list):.4f}")
    print("-" * 50)

    return {
        "precision": np.mean(precision_list),
        "hit_rate": np.mean(hit_rate_list),
        "ndcg": np.mean(ndcg_list),
    }


# #### 각 방식별 성능 평가

# In[12]:

results = {}

print("\n===== 추천 시스템 성능 평가 =====\n")

# 인기도 기반 추천 평가
results["popularity"] = evaluate_recommendations(
    recommendations_popular, ground_truth, "인기도 기반 추천"
)

# 콘텐츠 기반 추천 평가
results["content"] = evaluate_recommendations(
    recommendations_content, ground_truth, "콘텐츠 기반 추천"
)

# 협업 필터링 추천 평가 (활성화된 경우)
if USE_CF:
    results["cf"] = evaluate_recommendations(
        recommendations_cf, ground_truth, "협업 필터링 추천"
    )

# 카테고리 기반 추천 평가 (활성화된 경우)
if USE_CATEGORY:
    results["category"] = evaluate_recommendations(
        recommendations_category, ground_truth, "카테고리 기반 추천"
    )

# 하이브리드 추천 평가
results["hybrid"] = evaluate_recommendations(
    recommendations, ground_truth, "하이브리드 추천"
)

# 최고 성능 방식 출력
best_method = max(results.items(), key=lambda x: x[1]["hit_rate"])
print(
    f"🏆 최고 성능 방식: {best_method[0]}, Hit Rate: {best_method[1]['hit_rate']:.4f}"
)
