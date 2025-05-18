#!/usr/bin/env python
# coding: utf-8

# #### 감성분석 결과에 유저ID, 식당ID 추가

# In[3]:


import json
from tqdm import tqdm

# 파일 경로
full_data_file = "dataset/review_5up.json"
target_file = "dataset/review_5up_5aspect_3sentiment.jsonl"
output_file = "dataset/review_5up_5aspect_3sentiment_with_ids.jsonl"
# 전체 리뷰 데이터에서 review_id → (user_id, business_id) 맵 생성
id_map = {}

with open(full_data_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        rid = obj.get("review_id")
        if rid:
            id_map[rid] = {
                "user_id": obj.get("user_id"),
                "business_id": obj.get("business_id"),
            }

print(f" 전체 리뷰에서 매핑된 review_id 수: {len(id_map)}")

# 2. 대상 파일 읽고 user_id, business_id 추가
updated = []
missing = []

with open(target_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="🔄 ID 추가 중"):
        obj = json.loads(line)
        rid = obj.get("review_id")
        if rid in id_map:
            obj.update(id_map[rid])  # user_id, business_id 추가
            updated.append(obj)
        else:
            missing.append(obj)  # 매칭 안 되는 경우 따로 보관

# 3. 결과 저장
with open(output_file, "w", encoding="utf-8") as f:
    for obj in updated:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f" 저장 완료: 총 {len(updated)}건 → {output_file}")
if missing:
    print(f" 매칭 안된 리뷰 수: {len(missing)}")


# #### 추천형식에 맞게 변환

# In[11]:


import json

input_file = "dataset/review_5up_5aspect_3sentiment_with_ids.jsonl"
output_file = "dataset/review_5up_5aspect_3sentiment_vectorized_clean.json"


def sentiment_to_vector(sentiment_dict):
    aspects = ["food", "service", "price", "ambience", "location"]
    polarities = ["Negative", "Neutral", "Positive"]
    vector = []
    for asp in aspects:
        scores = sentiment_dict.get(asp, {}).get("scores", {})
        for pol in polarities:
            vector.append(scores.get(pol, 0.0))
    return vector


with open(input_file, "r", encoding="utf-8") as fin, open(
    output_file, "w", encoding="utf-8"
) as fout:

    for line in fin:
        obj = json.loads(line)

        # 벡터 생성
        sentiment_vec = sentiment_to_vector(obj.get("sentiment", {}))

        # 필요한 필드만 추출 및 재구성
        cleaned = {
            "review_id": obj.get("review_id"),
            "user_id": obj.get("user_id"),
            "business_id": obj.get("business_id"),
            "sentiment_vector": sentiment_vec,
        }

        fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

print(f"✅ 완료: text와 sentiment 제거 후 저장 → {output_file}")


# #### 유저별 평균 리뷰수 확인

# In[12]:


import json
from collections import defaultdict

# 파일 경로
INPUT_FILE = "dataset/review_5up_5aspect_3sentiment_vectorized_clean.json"

# 유저별 리뷰 수 저장
user_review_counts = defaultdict(int)

# 데이터 로딩
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        review = json.loads(line)
        user_id = review["user_id"]
        user_review_counts[user_id] += 1

# 통계 계산
total_users = len(user_review_counts)
total_reviews = sum(user_review_counts.values())
avg_reviews_per_user = total_reviews / total_users

print(f"📊 총 유저 수: {total_users}")
print(f"📝 총 리뷰 수: {total_reviews}")
print(f"📈 유저당 평균 리뷰 수: {avg_reviews_per_user:.2f}")


# #### 훈련 80% / 평가20% 로 분할

# In[1]:


import json
import math
from collections import defaultdict
from tqdm import tqdm

# 설정
INPUT_FILE = "dataset/review_5up_5aspect_3sentiment_vectorized_clean.json"
TRAIN_FILE = "dataset/train_80.json"
TEST_FILE = "dataset/test_20.json"
MIN_REVIEWS = 5

# 유저별 리뷰 모으기
user_reviews = defaultdict(list)

with open(INPUT_FILE, encoding="utf-8") as f:
    for line in f:
        review = json.loads(line)
        user_reviews[review["user_id"]].append(review)

# 분할
train_data, test_data = [], []

for uid, reviews in tqdm(user_reviews.items(), desc="20% 비율로 분할 중"):
    if len(reviews) < MIN_REVIEWS:
        continue

    # 시간순 정렬 ('date' 키가 없으면 'review_id' 등으로 대체 가능)
    sorted_reviews = sorted(reviews, key=lambda x: x.get("date", x.get("review_id")))

    k = max(1, math.ceil(len(reviews) * 0.2))  # 최소 1개는 평가용
    train_data.extend(sorted_reviews[:-k])
    test_data.extend(sorted_reviews[-k:])

# 저장
with open(TRAIN_FILE, "w", encoding="utf-8") as f:
    for r in train_data:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(TEST_FILE, "w", encoding="utf-8") as f:
    for r in test_data:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ 훈련 데이터 개수: {len(train_data)}")
print(f"✅ 평가 데이터 개수: {len(test_data)}")
print(f"📊 사용자 수: {len(user_reviews)}")


# In[3]:


import json


def count_unique_users(file_path):
    user_set = set()
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            review = json.loads(line)
            user_set.add(review["user_id"])
    return len(user_set)


train_file = "dataset/train_80.json"
test_file = "dataset/test_20.json"

train_user_count = count_unique_users(train_file)
test_user_count = count_unique_users(test_file)

print(f"👥 훈련 데이터의 유저 수: {train_user_count}")
print(f"🧪 평가 데이터의 유저 수: {test_user_count}")
