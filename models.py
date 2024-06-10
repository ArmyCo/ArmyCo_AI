import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TourEmbedding(nn.Module):
    def __init__(self, model_card: str, emb_path="./location_embedding.pth"):
        super(TourEmbedding, self).__init__()
        self.location_embed = torch.load(emb_path)
        self.llm = AutoModel.from_pretrained(model_card)

    def forward(self, location, reviews):
        """
        :param location: (batch_size, )
        :param reviews:  (batch_size, review_num, seq_len)
        :return: tourist attraction feature (batch_size, feature_dim)
        """
        batch_size, review_num, seq_len = reviews.shape

        review_outputs = self.llm(reviews["input_ids"].view(-1, seq_len),
                                  reviews["attention_mask"].view(-1, seq_len)).last_hidden_states[:, 0, :]  # cls token

        review_outputs = review_outputs.view(batch_size, review_num, -1)
        review_outputs = torch.mean(review_outputs, dim=1)
        location = self.location_embed(location)

        return torch.cat([location, review_outputs], dim=1)


class SimBasedRecsys:
    def __init__(self, user_idx, idx_attraction):
        self.tour_emb = torch.load("./tour_embedding.pth")
        self.location_emb = torch.load("./location_embedding.pth")

        self.user_idx = user_idx  # {user_name: idx}
        self.idx_attraction = idx_attraction  # {idx: attraction_name}

    def recommendation(self, user_location, k=3):
        """
        :param user_location: user_info
        :param k: the number of attractions to recommend
        :return: the result of recommendation
        """

        user_feature = self.location_emb[user_location]

        sim_matrix = torch.cosine_similarity(self.tour_emb, user_feature)

        top_k_recommendation = torch.topk(sim_matrix, k=k).indices

        return [self.idx_attraction[idx] for idx in top_k_recommendation]




