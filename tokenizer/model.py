import torch
import torch.nn as nn
import torch.nn.functional as F

def LN(x: torch.Tensor, eps: float = 1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std


class TiedTranspose(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.linear.bias is None
        return F.linear(x, self.linear.weight.t(), None)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight.t()

    @property
    def bias(self) -> torch.Tensor:
        return self.linear.bias

class SparseAutoencoder(nn.Module):
    def __init__(self, dim, topk, num_codes, tied = False, normalize = True):
        super(SparseAutoencoder, self).__init__()
        self.dim = dim
        self.encoder = nn.Linear(dim, num_codes, bias=False)
        self.decoder = TiedTranspose(self.encoder)
        self.topk = topk
        self.normalize = normalize
    
    def preprocess(self, x: torch.Tensor):
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)
    
    def encode(self, x: torch.Tensor):
        x, info = self.preprocess(x)
        self.encoder.weight.data = F.normalize(self.encoder.weight.data, p=2, dim=-1)
        latents_pre_act = F.linear(
            x, self.encoder.weight, bias=None
        )
        return latents_pre_act, info
    
    def decode(self, latents: torch.Tensor, info):
        ret = self.decoder(latents)
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret

    def forward(self, x):
        latents_pre_act, info = self.encode(x)
        topk_values, topk_indices = torch.topk(latents_pre_act, self.topk)
        mask = torch.zeros_like(latents_pre_act)
        mask.scatter_(1, topk_indices, 1.0)
        x_hat = self.decode(latents_pre_act * mask, info)
        return x_hat, latents_pre_act
    



