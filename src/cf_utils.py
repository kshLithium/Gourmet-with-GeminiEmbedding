from collections import defaultdict
from similarity import cosine_similarity_dict
import numpy as np
from tqdm import tqdm

def leave_one_out(ratings):
    """
    Leave-One-Out 방식으로 사용자별로 하나의 평가를 테스트용으로 분리하고 
    나머지를 학습용으로 사용하는 함수입니다.
    
    Parameters:
        ratings (DataFrame): 사용자-아이템 평점 데이터 ('user', 'biz', 'stars')
    
    Returns:
        train (DataFrame): 학습용 데이터
        test (dict): 테스트용 데이터 {user: test_item}
    """
    np.random.seed(42)  # 재현성 확보
    train_idx = []
    test = {}

    for u, grp in ratings.groupby("user"):
        idx = grp.index.values
        if len(idx) > 1:
            hold = np.random.choice(idx)  # 하나만 보류
            test[u] = ratings.loc[hold, "biz"]
            train_idx.extend([i for i in idx if i != hold])
        else:
            train_idx.extend(idx)  # 한 개뿐이면 그대로 train

    return ratings.loc[train_idx].reset_index(drop=True), test


def build_item_user_maps(ratings):
    """
    사용자-아이템 평점 데이터를 아이템 기준, 사용자 기준 dict로 재구성합니다.
    
    Parameters:
        ratings (DataFrame): 사용자-아이템 평점 데이터 ('user', 'biz', 'stars')
    
    Returns:
        item_users (dict): {item: {user: rating}}
        user_items (dict): {user: {item: rating}}
    """
    item_users = defaultdict(dict)
    user_items = defaultdict(dict)

    for _, r in ratings.iterrows():
        item_users[r["biz"]][r["user"]] = r["stars"]
        user_items[r["user"]][r["biz"]] = r["stars"]

    return item_users, user_items


def precompute_item_sim(item_users):
    """
    아이템 기반 협업 필터링을 위한 아이템 간 코사인 유사도 행렬을 미리 계산합니다.
    
    Parameters:
        item_users (dict): {item: {user: rating}}
    
    Returns:
        sims (dict): {item1: {item2: similarity}} 형태의 유사도 딕셔너리
    """
    items = list(item_users)
    sims = defaultdict(dict)

    for i, a in tqdm(enumerate(items), total=len(items), desc="아이템 유사도 계산"):
        for b in items[i + 1:]:  # 중복 계산 방지
            s = cosine_similarity_dict(item_users[a], item_users[b])
            if s > 0:
                sims[a][b] = s
                sims[b][a] = s  # 대칭

    return sims


def recommend(user, user_items, item_sims, n=5):
    """
    사용자에게 아직 평가하지 않은 아이템 중, 유사도와 기존 평점을 기반으로 상위 N개 추천.
    
    Parameters:
        user (str): 사용자 ID
        user_items (dict): {user: {item: rating}}
        item_sims (dict): {item1: {item2: similarity}}
        n (int): 추천할 아이템 수
    
    Returns:
        list: 추천된 아이템 ID 리스트
    """
    seen = set(user_items[user].keys())  # 이미 본 아이템 제외
    scores = defaultdict(float)

    for item, rating in user_items[user].items():
        for similar_item, sim in item_sims.get(item, {}).items():
            if similar_item not in seen:
                scores[similar_item] += sim * rating

    return [item for item, _ in sorted(scores.items(), key=lambda x: -x[1])[:n]]