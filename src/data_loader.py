import json
import numpy as np
import pandas as pd

def load_rating(path):
    """
    JSONL 형식의 평점 데이터를 로드하여 DataFrame으로 반환합니다.
    
    각 줄은 다음 필드를 포함해야 합니다:
        - user_id: 사용자 ID
        - business_id: 아이템(가게) ID
        - stars: 평점 (정수 또는 실수)
    
    Parameters:
        path (str): JSONL 파일 경로
    
    Returns:
        pd.DataFrame: ['user', 'biz', 'stars'] 컬럼을 갖는 DataFrame
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)  # 한 줄씩 JSON 디코딩
            rows.append(
                {"user": d["user_id"], "biz": d["business_id"], "stars": d["stars"]}
            )
    return pd.DataFrame(rows)


def load_rating_with_sentiment(path):
    """
    감성 벡터(sentiment_vector)를 포함하는 평점 데이터를 로드합니다.
    
    각 줄은 다음 필드를 포함해야 합니다:
        - user_id: 사용자 ID
        - business_id: 아이템(가게) ID
        - stars: 평점
        - sentiment_vector: 감성 정보 벡터 (리스트 → numpy 배열)
    
    Parameters:
        path (str): JSONL 파일 경로
    
    Returns:
        pd.DataFrame: ['user', 'biz', 'stars', 'sentiment_vector'] 컬럼을 갖는 DataFrame
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({
                "user": d["user_id"],
                "biz": d["business_id"],
                "stars": d["stars"],
                "sentiment_vector": np.array(d["sentiment_vector"])  # 리스트 → ndarray
            })
    return pd.DataFrame(rows)
