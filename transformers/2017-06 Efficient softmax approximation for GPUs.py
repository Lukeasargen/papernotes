
"""
Code from link below
https://github.com/huggingface/transformers/tree/main/src/transformers/models/transfo_xl
"""

import torch
import torch.nn as nn

from transformers.models.transfo_xl.modeling_transfo_xl import AdaptiveEmbedding
from transformers.models.transfo_xl.modeling_transfo_xl_utilities import ProjectedAdaptiveLogSoftmax

class SampleModel(nn.Module):
    def __init__(self, n_token, d_embed, d_model, cutoffs, div_val=1):
        super().__init__()
        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val)
        self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, cutoffs, div_val)

    def forward(self, x, labels=None):
        embeddings = self.word_emb(x)
        tokens = embeddings  # Transformer here
        softmax_output = self.crit(tokens, labels)
        print(f"{softmax_output.shape}")
        return embeddings

def main():
    n_token = 267735
    d_embed = 1024
    d_model = 1024  # Same as d_proj in AdaptiveEmbedding
    cutoffs = [20000, 40000, 200000]
    div_val = 4

    batch = 2
    seq_len = 4
    token_ids = torch.randint(low=0, high=n_token, size=(batch, seq_len))

    # model = SampleModel(n_token, d_embed, d_model, cutoffs, div_val)
    # print(model)
    # output = model(token_ids)
    # print(f"{output.shape=}")

    adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(in_features=d_model, n_classes=n_token, cutoffs=cutoffs, div_value=div_val)
    print(adaptive_softmax)

if __name__ == "__main__":
    main()