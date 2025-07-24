import torch
import torch.nn as nn


class CustomerRestaurantInteractionModule(nn.Module):
    def __init__(
        self, num_users: int, num_businesses: int, embedding_dim: int, mlp_dims: list
    ):
        super(CustomerRestaurantInteractionModule, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.business_embedding = nn.Embedding(num_businesses, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2  # 사용자 임베딩 + 식당 임베딩
        for dim in mlp_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = mlp_dims[-1] if mlp_dims else embedding_dim * 2

    def forward(self, user_ids: torch.Tensor, business_ids: torch.Tensor):
        user_vec = self.user_embedding(user_ids)
        business_vec = self.business_embedding(business_ids)
        combined_vec = torch.cat((user_vec, business_vec), dim=1)
        interaction_features = self.mlp(combined_vec)
        return interaction_features


class ReviewEmbeddingModule(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dims: list):
        super(ReviewEmbeddingModule, self).__init__()
        layers = []
        input_dim = embedding_dim
        for dim in mlp_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = mlp_dims[-1] if mlp_dims else embedding_dim

    def forward(self, embeddings: torch.Tensor):
        embedding_features = self.mlp(embeddings)
        return embedding_features


class ASRec(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_businesses: int,
        embedding_dim: int,
        user_biz_mlp_dims: list,
        review_mlp_dims: list,  # 파라미터 이름 변경
        final_mlp_dims: list,
        gemini_embedding_dim: int,  # 파라미터 이름 변경
    ):
        super(ASRec, self).__init__()
        self.customer_restaurant_interaction_module = (
            CustomerRestaurantInteractionModule(
                num_users, num_businesses, embedding_dim, user_biz_mlp_dims
            )
        )

        self.review_embedding_module = ReviewEmbeddingModule(  # <-- 이름 변경
            gemini_embedding_dim, review_mlp_dims  # 파라미터 이름 변경
        )

        final_input_dim = (
            self.customer_restaurant_interaction_module.output_dim
            + self.review_embedding_module.output_dim  # <-- 이름 변경
        )

        layers = []
        input_dim = final_input_dim
        for dim in final_mlp_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        self.prediction_mlp = nn.Sequential(*layers)

    def forward(
        self,
        user_ids: torch.Tensor,
        business_ids: torch.Tensor,
        embeddings: torch.Tensor,
    ):
        user_biz_features = self.customer_restaurant_interaction_module(
            user_ids, business_ids
        )
        embedding_features = self.review_embedding_module(embeddings)  # <-- 이름 변경
        combined_features = torch.cat(
            (user_biz_features, embedding_features), dim=1  # <-- 이름 변경
        )
        predicted_rating = self.prediction_mlp(combined_features)
        return predicted_rating.squeeze()
