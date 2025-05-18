#!/usr/bin/env python
# coding: utf-8

# #### import

# In[17]:


import json
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score
from tqdm import tqdm


# ### JSONL 파일에서 리뷰 로드

# In[18]:


TRAIN_FILE = "dataset/train_80_random.json"
TEST_FILE = "dataset/test_20_random.json"
TOP_N = 5


def load_reviews(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            obj["vector"] = np.array(obj["sentiment_vector"])
            data.append(obj)
    return data


train_reviews = load_reviews(TRAIN_FILE)
test_reviews = load_reviews(TEST_FILE)


# #### 훈련 데이터 → 벡터 평균 계산

# In[21]:


import json, numpy as np, pandas as pd
from collections import defaultdict
from datetime import datetime
from sklearn.preprocessing import normalize

# -------------------------------------------------
# 하이퍼 파라미터
# -------------------------------------------------
ASPECTS = ["food", "service", "price", "ambience", "location"]
HALF_LIFE = 180  # 최근성 half-life (일)
NEUT_THR = 0.80  # Neutral > 0.8 ⇒ 미언급으로 간주
TOP_N = 5  # 추천 개수


# -------------------------------------------------
# 1. 15-D 확률 → 5-D polarity
# -------------------------------------------------
# 1. 15-D 확률 → 5-D polarity  (Pos,Neu,Neg 순서용 버전)
def vec15_to_vec5(vec15):
    """
    vec15 = [pos, neu, neg] × 5   ← ❶ 우리 데이터는 이렇게 저장됨
    반환: 5-D polarity (pos-neg)  , Neutral > 0.8 → 0
    """
    out = []
    for i in range(5):
        pos, neu, neg = vec15[i * 3 : (i + 1) * 3]  # ← 앞뒤만 바꿔 줌
        if neu > NEUT_THR:  # 언급 안 된 aspect
            out.append(0.0)
        else:
            out.append(pos - neg)  # (-1, 1)
    return np.asarray(out, dtype=float)


# -------------------------------------------------
# 2. µ ∥ σ² 풀링 (최근성 가중)
# -------------------------------------------------
def recency_w(days):
    return np.exp(-np.log(2) * days / HALF_LIFE)


def agg_mu_sigma(rows):
    if not rows:  # 안전장치
        return np.zeros(10)
    now = datetime.now()
    V, W = [], []
    for v, ts in rows:
        V.append(v)
        W.append(recency_w((now - ts).days))
    V = np.vstack(V)  # (n,5)
    W = np.array(W)[:, None]  # (n,1)
    mu = (W * V).sum(0) / W.sum()
    var = (W * (V - mu) ** 2).sum(0) / W.sum()
    return np.concatenate([mu, var])  # 10-D


# -------------------------------------------------
# 3. 데이터 로드
# -------------------------------------------------
def load_reviews(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            obj["date"] = datetime.now()  # ★ 날짜 컬럼 없을 때 임시
            records.append(obj)
    return records


train_reviews = load_reviews("dataset/train_80_random.json")

# -------------------------------------------------
# 4. 임베딩 생성
# -------------------------------------------------
user_rows, biz_rows = defaultdict(list), defaultdict(list)
for r in train_reviews:
    vec15 = np.array(r["sentiment_vector"], float)
    vec5 = vec15_to_vec5(vec15)
    ts = r["date"]  # 실제 날짜 쓰면 더 정확
    user_rows[r["user_id"]].append((vec5, ts))
    biz_rows[r["business_id"]].append((vec5, ts))

user_embed = {u: agg_mu_sigma(rows) for u, rows in user_rows.items()}
biz_embed = {b: agg_mu_sigma(rows) for b, rows in biz_rows.items()}

# -------------------------------------------------
# 5. 정규화 후 유사도 행렬
# -------------------------------------------------
user_ids = list(user_embed)
biz_ids = list(biz_embed)

U = normalize(np.stack([user_embed[u] for u in user_ids]))
B = normalize(np.stack([biz_embed[b] for b in biz_ids]))
scores = U @ B.T

# -------------------------------------------------
# 6. 추천 생성 (재방문 허용)
# -------------------------------------------------
recommendations = {}
for i, uid in enumerate(user_ids):
    ranked_idx = np.argsort(scores[i])[::-1]
    recommendations[uid] = [biz_ids[j] for j in ranked_idx[:TOP_N]]

# 이후 ground-truth·평가 루프는 기존 코드 그대로 사용


# #### 정규화 후 유사도 계산

# In[20]:


user_ids = list(user_embed.keys())
biz_ids = list(biz_embed.keys())

user_matrix = normalize(np.stack([user_embed[u] for u in user_ids]))
biz_matrix = normalize(np.stack([biz_embed[b] for b in biz_ids]))

scores = np.dot(user_matrix, biz_matrix.T)


# #### 추천 리스트 생성

# In[8]:


user2seen = defaultdict(set)
for r in train_reviews:
    user2seen[r["user_id"]].add(r["business_id"])

recommendations = {}

for i, uid in enumerate(user_ids):
    user_score = scores[i]
    ranked_idx = np.argsort(user_score)[::-1]

    recs = []
    for j in ranked_idx:
        bid = biz_ids[j]
        # if bid not in user2seen[uid]:  # 이미 본 식당은 제외
        recs.append(bid)
        if len(recs) == TOP_N:
            break
    recommendations[uid] = recs


# #### 평가: Ground Truth 생성

# In[22]:


ground_truth = defaultdict(set)
for r in test_reviews:
    ground_truth[r["user_id"]].add(r["business_id"])


# #### Precision@K 계산

# In[23]:


common_users = set(recommendations.keys()) & set(ground_truth.keys())
precision_list = []

for uid in common_users:
    pred = set(recommendations[uid])
    actual = ground_truth[uid]
    hit = len(pred & actual)
    precision = hit / TOP_N
    precision_list.append(precision)


# #### 결과 출력

# In[24]:


print(f"📌 평가 대상 유저 수: {len(common_users)}")
print(f"🎯 Precision@{TOP_N}: {np.mean(precision_list):.4f}")


# In[ ]:


# In[ ]:


#

# In[12]:


print("train biz 수:", len(biz_embed))
hits_in_gt = 0
total_labels = 0
for u, items in ground_truth.items():
    total_labels += len(items)
    hits_in_gt += len([b for b in items if b in biz_embed])
print("GT 라벨 수:", total_labels, "그중 train 벡터 있는 라벨:", hits_in_gt)


# In[13]:


uid0 = next(iter(recommendations))
print("샘플 유저:", uid0)
print("추천 Top-10:", recommendations[uid0][:10])
print("GT 라벨:", ground_truth.get(uid0, set()))


# In[14]:


rng = np.random.default_rng(0)
rand_hit = 0
for uid in ground_truth:
    rand_preds = rng.choice(list(biz_embed.keys()), size=5, replace=False)
    if set(rand_preds) & ground_truth[uid]:
        rand_hit += 1
print("무작위 P@5 ≈", rand_hit / len(ground_truth))


# In[15]:


print("TOP_N =", TOP_N)


# In[ ]:


# In[16]:


import numpy as np, json, random, math

path = "dataset/train_80_random.json"
sample = json.loads(open(path).readline())

triplet = np.array(sample["sentiment_vector"][:3])  # food
print("food triplet:", triplet.round(3))

if triplet.argmax() == 0:
    print("순서 Pos,Neu,Neg  (뒤집힘)")
elif triplet.argmax() == 2:
    print("순서 Neg,Neu,Pos  (정상)")
else:
    print("Neutral이 최대 → 언급 안 됐거나 무감정")


# In[ ]:
