import torch
import torch.nn as nn
import torch.nn.functional as F

def LN(x: torch.Tensor, eps: float = 1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, mu, std
def topk_activation(x, k=256):
    topk_values, topk_indices = torch.topk(x, k, dim=1)
    sparse_x = torch.zeros_like(x)
    sparse_x.scatter_(1, topk_indices, topk_values)
    return sparse_x



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
    def __init__(self, txt_dim, clusters, hidden_dim, activation = 'topk', topk = 256, tied = False, normalize = True):
        super(SparseAutoencoder, self).__init__()

        self.txt_dim = txt_dim

        self.txt_encoder = nn.Linear(txt_dim, hidden_dim, bias=False)
        print(f"Tied: {tied}")
        if tied:
            self.txt_decoder = TiedTranspose(self.txt_encoder)
        else:
            self.txt_decoder = nn.Linear(hidden_dim, txt_dim, bias=False)
            self.txt_decoder.weight.data = self.txt_encoder.weight.data.t()

        print(f"Tied: {tied}")


        self.topk = topk
        self.normalize = normalize

        self.act = activation
    
    def preprocess(self, x: torch.Tensor):
        if not self.normalize:
            return x, dict()
        x, mu, std = LN(x)
        return x, dict(mu=mu, std=std)
    
    def txt_encode(self, x: torch.Tensor):
        x, info = self.preprocess(x)
        self.txt_encoder.weight.data = F.normalize(self.txt_encoder.weight.data, p=2, dim=-1)
        latents_pre_act = F.linear(
            x, self.txt_encoder.weight, bias=None
        )
        return latents_pre_act, info
    
    
    def txt_decode(self, latents: torch.Tensor, info):
        ret = self.txt_decoder(latents)
        if self.normalize:
            assert info is not None
            ret = ret * info["std"] + info["mu"]
        return ret


    def forward(self, txt_x):
        txt_latents_pre_act, txt_info = self.txt_encode(txt_x)

        latents_pre_act = txt_latents_pre_act

        topk_values, topk_indices = torch.topk(latents_pre_act, self.topk)
        mask = torch.zeros_like(latents_pre_act)
        mask.scatter_(1, topk_indices, 1.0)


        txt_recons = self.txt_decode(txt_latents_pre_act * mask, txt_info)


        return txt_recons, latents_pre_act
    



