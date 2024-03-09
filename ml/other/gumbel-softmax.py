import torch
from torch import nn
from torch.nn import functional as F


def gumbel_distribution(shape, eps=1e-11, device=None):
    u = torch.rand(*shape, device=device)
    return -torch.log(-torch.log(u + eps))


class GumbelSoftmax(nn.Module):
    def __init__(self):
        super(GumbelSoftmax, self).__init__()

    def forward(self, logits, temperature=1.0, hard_sample=False):
        log_probs = F.log_softmax(logits, dim=-1)
        y = log_probs + gumbel_distribution(shape=logits.size(), device=logits.device)
        y_soft = F.softmax(y/temperature, dim=-1)

        if not hard_sample:
            return y_soft

        _, idx = y_soft.max(dim=-1)
        y_hard = torch.zeros_like(y_soft).view(-1, y_soft.size(-1))
        y_hard.scatter_(1, idx.view(-1, 1), 1)
        y_hard = y_hard.view(y_soft.size())
        # Straight-through https://pytorch.org/docs/1.12/generated/torch.nn.functional.gumbel_softmax.html
        y_hard = (y_hard - y_soft).detach() + y_soft
        return y_hard


def main():
    batch = 1
    dims = 4
    temperature = 1.0
    hard_sample = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    inputs = torch.rand(batch, dims, device=device)
    inputs[:,0] = 2
    print(f"{inputs=}")
    print(f" probs={F.softmax(inputs, dim=-1)}")

    gumbel_softmax = GumbelSoftmax()
    for i in range(10):
        sample = gumbel_softmax(inputs, temperature=temperature, hard_sample=hard_sample)
        print(f"{sample=}")

if __name__ == "__main__":
    main()
