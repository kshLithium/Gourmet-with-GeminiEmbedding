import math

def precision_at_k(test, recs, k=5):
    """
    Precision@K 계산
    - 추천 상위 K개 중 정답 아이템이 포함된 비율을 측정
    
    Parameters:
        test (dict): 테스트 데이터 (user -> ground truth item)
        recs (dict): 추천 결과 (user -> 추천 아이템 리스트)
        k (int): 평가할 추천 개수
    
    Returns:
        float: Precision@K 값 (0~1)
    """
    hit_count = sum(1 for u, gt in test.items() if gt in recs.get(u, [])[:k])
    return hit_count / len(test)


def recall_at_k(test, recs, k=5):
    """
    Recall@K 계산
    - Leave-One-Out 기준으로 Precision과 동일한 값이 됨
    
    Parameters:
        test (dict): 테스트 데이터 (user -> ground truth item)
        recs (dict): 추천 결과 (user -> 추천 아이템 리스트)
        k (int): 평가할 추천 개수
    
    Returns:
        float: Recall@K 값 (0~1)
    """
    return precision_at_k(test, recs, k)  # ground truth 1개이므로 동일


def ndcg_at_k(test, recs, k=5):
    """
    NDCG@K 계산
    - 추천된 순서까지 고려한 정답 아이템의 점수(DCG)를 정규화 없이 평균
    
    Parameters:
        test (dict): 테스트 데이터 (user -> ground truth item)
        recs (dict): 추천 결과 (user -> 추천 아이템 리스트)
        k (int): 평가할 추천 개수
    
    Returns:
        float: NDCG@K 값 (0~1)
    """
    total = 0.0
    for u, gt in test.items():
        rec_list = recs.get(u, [])[:k]
        if gt in rec_list:
            idx = rec_list.index(gt)
            # DCG 점수: log 기반으로 순위가 낮을수록 감점
            total += 1 / math.log2(idx + 2)  # +2: 인덱스 0일 때 log2(2)=1
    return total / len(test)
