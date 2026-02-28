import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/sunny2109/SAFMN
# 论文：https://arxiv.org/pdf/2302.13800
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=1):
        super().__init__()
        self.n_levels = n_levels
        chunk_dim = dim // n_levels

        # Spatial Weighting
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # # Feature Aggregation
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        xc = x.chunk(self.n_levels, dim=1)
        out = []
        for i in range(self.n_levels):
            if i > 0:
                p_size = (h // 2 ** i, w // 2 ** i)
                s = F.adaptive_max_pool2d(xc[i], p_size)
                s = self.mfr[i](s)
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                s = self.mfr[i](xc[i])
            print(s.shape)
            out.append(s)


        out = self.aggr(torch.cat(out, dim=1))
        out = self.act(out) * x
        print(out.shape)
        return out


if __name__ == '__main__':
    input = torch.randn(1, 3, 64, 64)  # 输入b c h w

    block = SAFM(dim=3)
    output = block(input)
    print(output.size())