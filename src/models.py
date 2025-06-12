import torch
import torch.nn as nn

class CustomerRestaurantInteractionModule(nn.Module):
    def __init__(self, num_users: int, num_businesses: int, embedding_dim: int, mlp_dims: list):
        super(CustomerRestaurantInteractionModule, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.business_embedding = nn.Embedding(num_businesses, embedding_dim)

        layers = []
        input_dim = embedding_dim * 2 # 사용자 임베딩 + 식당 임베딩
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

class ReviewAspectModule(nn.Module):
    def __init__(self, sentiment_vector_dim: int, aspect_mlp_dims: list):
        super(ReviewAspectModule, self).__init__()
        layers = []
        input_dim = sentiment_vector_dim
        for dim in aspect_mlp_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = aspect_mlp_dims[-1] if aspect_mlp_dims else sentiment_vector_dim

    def forward(self, sentiment_vectors: torch.Tensor):
        aspect_features = self.mlp(sentiment_vectors)
        return aspect_features

class ASRec(nn.Module):
    def __init__(self, num_users: int, num_businesses: int, embedding_dim: int,
                 user_biz_mlp_dims: list, aspect_mlp_dims: list, final_mlp_dims: list,
                 sentiment_vector_dim: int):
        super(ASRec, self).__init__()
        self.customer_restaurant_interaction_module = CustomerRestaurantInteractionModule(
            num_users, num_businesses, embedding_dim, user_biz_mlp_dims
        )
        self.review_aspect_module = ReviewAspectModule(
            sentiment_vector_dim, aspect_mlp_dims
        )

        final_input_dim = self.customer_restaurant_interaction_module.output_dim + \
                          self.review_aspect_module.output_dim

        layers = []
        input_dim = final_input_dim
        for dim in final_mlp_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1)) # 최종 출력은 1차원 평점
        self.prediction_mlp = nn.Sequential(*layers)

    def forward(self, user_ids: torch.Tensor, business_ids: torch.Tensor, sentiment_vectors: torch.Tensor):
        user_biz_features = self.customer_restaurant_interaction_module(user_ids, business_ids)
        aspect_features = self.review_aspect_module(sentiment_vectors)
        combined_features = torch.cat((user_biz_features, aspect_features), dim=1)
        predicted_rating = self.prediction_mlp(combined_features)
        return predicted_rating.squeeze() # 평점 반환

