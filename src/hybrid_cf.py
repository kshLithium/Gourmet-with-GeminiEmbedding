from collections import defaultdict
import numpy as np
from similarity import cosine_similarity_dict, cosine_similarity_vec
from tqdm import tqdm

def build_maps(train_df):
    """
    사용자-아이템, 아이템-사용자, 아이템별 감성 벡터 평균 맵 생성
    
    Parameters:
        train_df (DataFrame): 훈련 데이터프레임 (user, biz, stars, sentiment_vector 포함)

    Returns:
        item_users (dict): 아이템 -> (사용자 -> 평점) 맵
        user_items (dict): 사용자 -> (아이템 -> 평점) 맵
        item_sentiment_avg (dict): 아이템 -> 감성 벡터 평균
    """
    item_users = defaultdict(dict)         # 아이템 기준 평점 정보
    user_items = defaultdict(dict)         # 사용자 기준 평점 정보
    item_sentiments = defaultdict(list)    # 아이템별 감성 벡터 모음

    for _, r in train_df.iterrows():
        item_users[r["biz"]][r["user"]] = r["stars"]
        user_items[r["user"]][r["biz"]] = r["stars"]
        item_sentiments[r["biz"]].append(r["sentiment_vector"])

    # 아이템별 감성 벡터 평균 계산
    item_sentiment_avg = {
        biz: np.mean(vectors, axis=0)
        for biz, vectors in item_sentiments.items()
    }

    return item_users, user_items, item_sentiment_avg


def precompute_hybrid_sims(item_users, item_sentiment_avg, alpha=0.8):
    """
    하이브리드 유사도 계산 (평점 유사도 + 감성 유사도 결합)
    
    Parameters:
        item_users (dict): 아이템 -> (사용자 -> 평점) 맵
        item_sentiment_avg (dict): 아이템 -> 감성 벡터 평균
        alpha (float): 평점 기반 유사도 가중치 (0.0~1.0)

    Returns:
        sims (dict): 아이템 쌍 -> 유사도 점수
    """
    items = list(item_users)
    sims = defaultdict(dict)

    for i, a in tqdm(enumerate(items), total=len(items), desc="하이브리드 유사도 계산"):
        for b in items[i + 1:]:
            # 평점 기반 유사도 (사용자 평점 유사도)
            rating_sim = cosine_similarity_dict(item_users[a], item_users[b])
            
            # 감성 벡터 유사도 (예: 음식/서비스/가격 등 측면별 평균 감성)
            sent_sim = cosine_similarity_vec(
                item_sentiment_avg.get(a, np.zeros(15)),
                item_sentiment_avg.get(b, np.zeros(15))
            )
            
            # 두 유사도의 가중 평균
            hybrid_sim = alpha * rating_sim + (1 - alpha) * sent_sim

            # 0보다 큰 유사도만 저장 (희소성 유지)
            if hybrid_sim > 0:
                sims[a][b] = hybrid_sim
                sims[b][a] = hybrid_sim

    return sims
