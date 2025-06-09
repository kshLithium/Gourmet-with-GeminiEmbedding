import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class ReviewDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.user_ids = torch.tensor(df['user_encoded'].values, dtype=torch.long)
        self.business_ids = torch.tensor(df['business_encoded'].values, dtype=torch.long)
        self.sentiment_vectors = torch.tensor(np.array(df['sentiment_vector'].tolist()), dtype=torch.float)
        self.stars = torch.tensor(df['stars'].values, dtype=torch.float)

    def __len__(self):
        return len(self.stars)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.business_ids[idx], self.sentiment_vectors[idx], self.stars[idx]

