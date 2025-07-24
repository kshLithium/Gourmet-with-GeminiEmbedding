import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os


class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.user_ids = torch.tensor(df["user_encoded"].values, dtype=torch.long)
        self.business_ids = torch.tensor(
            df["business_encoded"].values, dtype=torch.long
        )

        # sliced_embeddings = [v[:768] for v in df["embedding"].tolist()]
        # self.embeddings = torch.tensor(np.array(sliced_embeddings), dtype=torch.float)

        self.embeddings = torch.tensor(
            np.array(df["embedding"].tolist()), dtype=torch.float
        )

        # self.sentiment_vectors = torch.tensor(np.array(df['sentiment_vector'].tolist()), dtype=torch.float)

        self.stars = torch.tensor(df["stars"].values, dtype=torch.float)

    def __len__(self):
        return len(self.stars)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.business_ids[idx],
            self.embeddings[idx],
            self.stars[idx],
        )


####


class ReviewNpyDataset(Dataset):
    def __init__(self, data_dir: str, data_type: str):
        """
        메모리 맵 모드를 사용하여 .npy 파일에서 데이터를 로드합니다.

        Args:
            data_dir (str): .npy 파일들이 저장된 디렉토리 경로 (e.g., 'Dataset/processed_npy')
            data_type (str): 불러올 데이터 종류 ('train', 'val', 'test')
        """
        # 메모리 맵 모드('r')로 NumPy 배열 파일을 엽니다.
        # 이렇게 하면 파일 전체가 RAM으로 로드되지 않습니다.
        self.user_ids = np.load(
            os.path.join(data_dir, f"{data_type}_users.npy"), mmap_mode="r"
        )
        self.business_ids = np.load(
            os.path.join(data_dir, f"{data_type}_businesses.npy"), mmap_mode="r"
        )
        self.embeddings = np.load(
            os.path.join(data_dir, f"{data_type}_embeddings.npy"), mmap_mode="r"
        )
        self.stars = np.load(
            os.path.join(data_dir, f"{data_type}_stars.npy"), mmap_mode="r"
        )

        # [핵심 수정] len() 대신 .shape[0]을 사용하여 배열의 실제 전체 길이를 가져옵니다.
        self.length = self.stars.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # NumPy 배열에서 데이터를 가져와 PyTorch 텐서로 변환하여 반환합니다.
        # .item()은 NumPy 스칼라를 파이썬 기본 타입으로 변환합니다.
        # from_numpy는 NumPy 배열과 메모리를 공유하여 더 효율적입니다.
        return (
            torch.tensor(self.user_ids[idx].item(), dtype=torch.long),
            torch.tensor(self.business_ids[idx].item(), dtype=torch.long),
            torch.from_numpy(self.embeddings[idx]),
            torch.tensor(self.stars[idx].item(), dtype=torch.float),
        )
