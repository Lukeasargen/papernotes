
"""
Code from link below
https://github.com/huggingface/transformers/tree/main/src/transformers/models/transfo_xl
"""

import torch
import torch.nn as nn


class AdaptiveEmbedding(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, sample_softmax=False):
        super().__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.div_val = div_val
        self.d_proj = d_proj
        self.emb_scale = d_proj**0.5
        self.cutoffs = cutoffs + [n_token]  # These are all the right side of the sections
        self.cutoff_ends = [0] + self.cutoffs  # Now each pair is a section
        self.emb_layers = nn.ModuleList()
        self.emb_projs = nn.ParameterList()  # FloatTensor creates nearly zero tensors
        if div_val == 1:
            # Without division, it's only a simple embedding layer
            self.emb_layers.append(nn.Embedding(n_token, d_embed, sparse=sample_softmax > 0))
            if d_proj != d_embed:
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_embed)))
        else:
            # Create a different embedding for each cutoff section
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                d_emb_i = d_embed // (div_val**i)
                self.emb_layers.append(nn.Embedding(r_idx - l_idx, d_emb_i))
                self.emb_projs.append(nn.Parameter(torch.FloatTensor(d_proj, d_emb_i)))

    def forward(self, inp):
        if self.div_val == 1:
            # Index a simple Embedding with the input Ids
            embed = self.emb_layers[0](inp)
            if self.d_proj != self.d_embed:
                embed = nn.functional.linear(embed, self.emb_projs[0])
        else:
            param = next(self.parameters())  # Used to get the dtype
            # Flatten the original ids to a 1D vector
            inp_flat = inp.view(-1)
            # Allocate a tensor for the output, size is sequence length by model dimension
            emb_flat = torch.zeros([inp_flat.size(0), self.d_proj], dtype=param.dtype, device=param.device)
            # Loop over each section, checking for tokens in the section
            for i in range(len(self.cutoffs)):
                l_idx, r_idx = self.cutoff_ends[i], self.cutoff_ends[i + 1]
                # convert ids to boolean, represents if it's inside the section
                mask_i = (inp_flat >= l_idx) & (inp_flat < r_idx)
                # Find the indices in the section
                indices_i = mask_i.nonzero().squeeze()
                if indices_i.numel() == 0:
                    # Skip if there are no token ids
                    continue
                # SHIFT THE IDS by the left section end
                inp_i = inp_flat.index_select(0, indices_i) - l_idx
                # Get the embeddings
                emb_i = self.emb_layers[i](inp_i)
                # Project the embeddings to the model dimension
                emb_i = nn.functional.linear(emb_i, self.emb_projs[i])
                # Place the embeddings in the output tensor using the position indices
                emb_flat.index_copy_(0, indices_i, emb_i)
            # un flatten the ids
            embed_shape = inp.size() + (self.d_proj,)
            embed = emb_flat.view(embed_shape)
        embed.mul_(self.emb_scale)
        return embed


class ProjectedAdaptiveLogSoftmax(nn.Module):
    def __init__(self, n_token, d_embed, d_proj, cutoffs, div_val=1, keep_order=False):
        super().__init__()
        self.n_token = n_token
        self.d_embed = d_embed
        self.d_proj = d_proj
        self.div_val = div_val
        self.cutoffs = cutoffs + [n_token]
        self.cutoff_ends = [0] + self.cutoffs
        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters


class SampleModel(nn.Module):
    def __init__(self, n_token, d_embed, d_model, cutoffs, div_val=1):
        super().__init__()
        self.word_emb = AdaptiveEmbedding(n_token, d_embed, d_model, cutoffs, div_val)
        self.crit = ProjectedAdaptiveLogSoftmax(n_token, d_embed, d_model, cutoffs, div_val)

    def forward(self, x):
        embeddings = self.word_emb(x)
        # transformer here
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

    model = SampleModel(n_token, d_embed, d_model, cutoffs, div_val)
    # print(model)
    embeddings = model(token_ids)
    print(f"{embeddings.shape=}")


if __name__ == "__main__":
    main()