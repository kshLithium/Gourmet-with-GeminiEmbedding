from data_loader import load_rating, load_rating_with_sentiment
from cf_utils import leave_one_out, build_item_user_maps, precompute_item_sim, recommend
from eval_metrics import precision_at_k, recall_at_k, ndcg_at_k
from tqdm import tqdm

def log(msg):
    """로그 메시지를 출력합니다 (구분 기호 포함)."""
    print(f"\n🟩 {msg}")

def run_experiment(
    rating_path,
    k=5,                     # 추천 개수 및 평가 기준
    min_ratings=5,           # 최소 사용자 리뷰 수 기준
    model_type="itemcf",     # 추천 모델 유형 ("itemcf" 또는 "hybrid")
    alpha=0.8,               # hybrid 모델에서 평점과 감성 가중치 비율
    verbose=True             # 중간 과정 출력 여부
):
    # 📌 1. 데이터 로드
    if verbose:
        log(f"데이터 로드 중: {rating_path}")
        
    if model_type == "hybrid":
        ratings = load_rating_with_sentiment(rating_path)  # 감성 벡터 포함
    elif model_type == "itemcf":
        ratings = load_rating(rating_path)  # 평점만 포함
    else:
        raise ValueError(f"❌ 지원하지 않는 model_type: {model_type}")

    # 📌 2. 최소 리뷰 수를 기준으로 사용자 필터링
    user_counts = ratings.groupby("user").size()
    valid_users = user_counts[user_counts >= min_ratings].index
    ratings = ratings[ratings["user"].isin(valid_users)]

    if verbose:
        print(f"총 {len(ratings):,}개 평점, {ratings['user'].nunique():,}명 사용자, {ratings['biz'].nunique():,}개 아이템")

    # 📌 3. 학습/테스트 데이터 분리 (Leave-One-Out)
    if verbose:
        log("학습/테스트 데이터 분리 중...")
    train, test = leave_one_out(ratings)
    if verbose:
        print(f"학습: {len(train):,}개, 테스트: {len(test):,}개")

    if len(test) == 0:
        raise ValueError("⚠️ 테스트 데이터가 없습니다. min_ratings 값을 낮춰보세요.")

    # 📌 4. 모델별 유사도 계산
    if model_type == "itemcf":
        if verbose:
            log("아이템-사용자 맵 구성 중...")
        item_users, user_items = build_item_user_maps(train)

        if verbose:
            log("아이템 유사도 계산 중 (item-based CF)...")
        item_sims = precompute_item_sim(item_users)

    elif model_type == "hybrid":
        from hybrid_cf import build_maps, precompute_hybrid_sims

        if verbose:
            log("감성 벡터 기반 맵 구성 중...")
        item_users, user_items, item_sentiment_avg = build_maps(train)

        if verbose:
            log(f"하이브리드 유사도 계산 중... (alpha={alpha})")
        item_sims = precompute_hybrid_sims(item_users, item_sentiment_avg, alpha=alpha)

    # 📌 5. 추천 수행
    if verbose:
        log(f"상위 {k}개 아이템 추천 중...")
    recommendations = {}
    for user in tqdm(test.keys(), desc="추천 중"):
        if user in user_items and len(user_items[user]) > 0:
            recommendations[user] = recommend(user, user_items, item_sims, n=k)

    # 📌 6. 추천 성능 평가
    if verbose:
        log("성능 평가 중...")
    p_at_k = precision_at_k(test, recommendations, k)
    r_at_k = recall_at_k(test, recommendations, k)
    n_at_k = ndcg_at_k(test, recommendations, k)

    if verbose:
        print(f"\n🎯 Precision@{k}: {p_at_k:.4f}")
        print(f"🎯 Recall@{k}:    {r_at_k:.4f}")
        print(f"🎯 NDCG@{k}:      {n_at_k:.4f}")

    return {
        f"precision@{k}": p_at_k,
        f"recall@{k}": r_at_k,
        f"ndcg@{k}": n_at_k
    }
