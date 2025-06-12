import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import os
import pandas as pd
from typing import Dict, List, Tuple


from utils import mean_absolute_percentage_error
from models import ASRec

def train_model(model: ASRec, train_loader: DataLoader, val_loader: DataLoader,
                criterion: nn.Module, optimizer: optim.Optimizer,
                epochs: int, patience: int, min_delta: float, model_path: str, device: torch.device):
    """
    조기 종료(Early Stopping)를 사용하여 AS-Rec 모델을 학습

    인자:
        model (ASRec): AS-Rec 모델 인스턴스
        train_loader (DataLoader): 학습 세트용 DataLoader
        val_loader (DataLoader): 검증 세트용 DataLoader
        criterion (nn.Module): 손실 함수 (예: nn.MSELoss).
        optimizer (optim.Optimizer): 옵티마이저 (예: optim.Adam)
        epochs (int): 최대 학습 에포크 수
        patience (int): 조기 종료 전 개선을 기다릴 에포크 수
        min_delta (float): 개선으로 인정할 검증 RMSE의 최소 변화량
        model_path (str): 최적 모델 가중치를 저장할 경로
        device (torch.device): 학습을 실행할 장치 (cuda 또는 cpu)

    반환:
        None
    """
    best_val_rmse = float('inf')
    epochs_no_improve = 0

    print("\n 모델 학습 시작")
    for epoch in range(epochs):
        # 학습 단계
        model.train()
        total_train_loss = 0
        for user_ids, business_ids, sentiment_vectors, stars in train_loader:
            user_ids, business_ids, sentiment_vectors, stars = \
                user_ids.to(device), business_ids.to(device), sentiment_vectors.to(device), stars.to(device)

            optimizer.zero_grad()
            predictions = model(user_ids, business_ids, sentiment_vectors)
            loss = criterion(predictions, stars)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # 검증 단계
        model.eval()
        val_predictions = []
        val_true_ratings = []
        with torch.no_grad():
            for user_ids, business_ids, sentiment_vectors, stars in val_loader:
                user_ids, business_ids, sentiment_vectors, stars = \
                    user_ids.to(device), business_ids.to(device), sentiment_vectors.to(device), stars.to(device)

                predictions = model(user_ids, business_ids, sentiment_vectors)
                val_predictions.extend(predictions.cpu().tolist())
                val_true_ratings.extend(stars.cpu().tolist())

        current_val_rmse = np.sqrt(mean_squared_error(val_true_ratings, val_predictions))

        print(f"에포크 {epoch+1}/{epochs}, "
              f"학습 손실: {total_train_loss / len(train_loader):.4f}, "
              f"검증 RMSE: {current_val_rmse:.4f}")

        # 조기 종료 로직
        if current_val_rmse < best_val_rmse - min_delta:
            best_val_rmse = current_val_rmse
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"RMSE 개선됨. 모델 저장: {best_val_rmse:.4f}")
        else:
            epochs_no_improve += 1
            print(f"RMSE 개선되지 않음. ({epochs_no_improve}/{patience})")
            if epochs_no_improve == patience:
                print(f"조기 종료 - {patience} 에포크 동안 검증 RMSE 개선 없음.")
                break

def evaluate_model(model: ASRec, test_loader: DataLoader, device: torch.device, model_path: str = None) -> Dict[str, float]:
    """
    학습된 모델을 테스트 세트에서 평가

    인자:
        model (ASRec): AS-Rec 모델 인스턴스
        test_loader (DataLoader): 테스트 세트용 DataLoader
        device (torch.device): 평가를 실행할 장치 (cuda 또는 cpu)
        model_path (str, optional): 저장된 모델 가중치 경로. 제공되면, 평가 전에 이 가중치를 로드함

    반환:
        Dict[str, float]: 평가 지표(MSE, RMSE, MAE, MAPE)를 포함하는 딕셔너리
    """
    print("\n테스트 세트에서 모델 평가")
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"최적 모델 가중치를 '{model_path}'에서 로드했습니다.")
    elif model_path:
        print(f"최적 최종 모델 가중치 '{model_path}'를 찾을 수 없습니다. 현재 모델 상태로 테스트합니다.")


    model.eval()
    test_predictions = []
    true_ratings = []

    with torch.no_grad():
        for user_ids, business_ids, sentiment_vectors, stars in test_loader:
            user_ids, business_ids, sentiment_vectors, stars = \
                user_ids.to(device), business_ids.to(device), sentiment_vectors.to(device), stars.to(device)

            predictions = model(user_ids, business_ids, sentiment_vectors)
            test_predictions.extend(predictions.cpu().tolist())
            true_ratings.extend(stars.cpu().tolist())

    mse = mean_squared_error(true_ratings, test_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_ratings, test_predictions)
    mape = mean_absolute_percentage_error(true_ratings, test_predictions)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape
    }

    print(f"평균 제곱 오차 (MSE): {mse:.4f}")
    print(f"평균 제곱근 오차 (RMSE): {rmse:.4f}")
    print(f"평균 절대 오차 (MAE): {mae:.4f}")
    print(f"평균 절대 백분율 오차 (MAPE): {mape:.2f}%")

    return metrics

def recommend_topk_for_all_users(model: ASRec, df_processed: pd.DataFrame,
                                 user_encoder, business_encoder, k: int = 5, device: torch.device = 'cpu') -> Dict[str, List[str]]:
    """
    데이터셋의 모든 고유 사용자에 대해 Top-K 비즈니스 추천을 생성

    인자:
        model (ASRec): 학습된 AS-Rec 모델
        df_processed (pd.DataFrame): 사용자, 비즈니스, 평점, 감성 벡터를 포함하는 전처리된 DataFrame
        user_encoder: user_id 인코딩에 사용된 LabelEncoder
        business_encoder: business_id 인코딩에 사용된 LabelEncoder
        k (int): 각 사용자에 대해 생성할 상위 추천 수
        device (torch.device): 예측을 실행할 장치

    반환:
        Dict[str, List[str]]: 키는 원본 user_id이고 값은 추천된 원본 business_id 목록인 딕셔너리
    """
    model.eval()
    user_ids_unique = df_processed['user_id'].unique()
    business_ids_unique = df_processed['business_id'].unique()

    # 각 business_id에 대한 평균 sentiment_vector 계산
    sentiment_dict = df_processed.groupby('business_id')['sentiment_vector'].apply(
        lambda x: np.mean(x.tolist(), axis=0)
    ).to_dict()

    recommendations = {}

    # 고유한 사용자를 반복
    for user_id in user_ids_unique:
        encoded_user = user_encoder.transform([user_id])[0]

        # 현재 사용자가 이미 평점을 매긴 비즈니스
        rated_biz = df_processed[df_processed['user_id'] == user_id]['business_id'].unique()
        # 사용자가 평점을 매기지 않았고 감성 벡터를 사용할 수 있는 비즈니스
        unrated_biz = [b for b in business_ids_unique if b not in rated_biz and b in sentiment_dict]

        if not unrated_biz:
            recommendations[user_id] = []
            continue

        # 예측을 위한 텐서 준비
        user_tensor = torch.tensor([encoded_user] * len(unrated_biz), dtype=torch.long).to(device)
        biz_encoded = business_encoder.transform(unrated_biz)
        biz_tensor = torch.tensor(biz_encoded, dtype=torch.long).to(device)
        sentiment_list = [sentiment_dict[b] for b in unrated_biz]
        sentiment_tensor = torch.tensor(np.array(sentiment_list), dtype=torch.float).to(device)

        # 예측 수행
        with torch.no_grad():
            predicted_ratings = model(user_tensor, biz_tensor, sentiment_tensor)

        # Top-K 인덱스와 해당 비즈니스 ID 가져오기
        actual_k = min(k, len(predicted_ratings))
        if actual_k > 0:
            topk_indices = torch.topk(predicted_ratings, actual_k).indices.tolist()
            topk_business_ids = [unrated_biz[i] for i in topk_indices]
        else:
            topk_business_ids = []

        recommendations[user_id] = topk_business_ids

    return recommendations

