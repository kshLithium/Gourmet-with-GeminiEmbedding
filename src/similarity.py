import math
import numpy as np

def cosine_similarity_dict(a, b):
    """
    딕셔너리 기반 코사인 유사도 계산 (예: 사용자 평점 기반)
    
    Parameters:
        a, b (dict): {key: value} 형태의 벡터 (ex. user 평점 딕셔너리)

    Returns:
        float: 코사인 유사도 (0.0 ~ 1.0)
    """
    common = set(a) & set(b)
    if not common:
        return 0.0

    num = sum(a[u] * b[u] for u in common)
    den = (
        math.sqrt(sum(v * v for v in a.values())) *
        math.sqrt(sum(v * v for v in b.values())) + 1e-9
    )

    return num / den


def cosine_similarity_vec(a, b):
    """
    NumPy 벡터 기반 코사인 유사도 계산 (예: 감성 벡터 간 유사도)
    
    Parameters:
        a, b (np.ndarray): 실수 벡터

    Returns:
        float: 코사인 유사도
    """
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))