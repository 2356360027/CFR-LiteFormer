import math
import torch
from torch import nn
from einops import rearrange, repeat
from typing import Optional, Callable
from timm.models.layers import DropPath, trunc_normal_
from inspect import isfunction
from functools import partial
import torch.nn.functional as F
from functools import partial
from typing import List
from torch import Tensor
import einops
import itertools
torch.autograd.set_grad_enabled(True)
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m



class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=8, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))

# building block modules
class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        return x

class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 norm_layer,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x



class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        # self.block1=depthwise_projection(in_features=dim,
        #                      out_features=dim_out,
        #                      kernel_size=(3,3),
        #                      padding=1,
        #                      groups=norm_groups,
        #                      pointwise=True)

        # self.block1 = MLPBlock(
        #     dim=dim,
        #     n_div=4,
        #     mlp_ratio=2,
        #     drop_path=0.,
        #     layer_scale_init_value=0,
        #     norm_layer=nn.BatchNorm2d,
        #     act_layer=partial(nn.ReLU, inplace=True),
        #     pconv_fw_type='split_cat'
        # )
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        # self.block2=depthwise_projection(in_features=dim_out,
        #                      out_features=dim_out,
        #                      kernel_size=(3,3),
        #                      padding=1,
        #                      groups=norm_groups,
        #                      pointwise=True)
        # self.block2 = MLPBlock(
        #     dim=dim_out,
        #     n_div=4,
        #     mlp_ratio=2,
        #     drop_path=0.,
        #     layer_scale_init_value=0,
        #     norm_layer=nn.BatchNorm2d,
        #     act_layer=partial(nn.ReLU, inplace=True),
        #     pconv_fw_type='split_cat'
        # )
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp) and exists(time_emb):
            h += self.mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)
class depthwise_conv_block(nn.Module):
    def __init__(self,
                in_features,
                out_features,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                groups=None,
                norm_type='bn',
                activation=True,
                use_bias=True,
                pointwise=True,
                ):
        super().__init__()
        self.pointwise = pointwise
        self.norm = norm_type
        self.act = activation
        self.depthwise = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features if pointwise else out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=use_bias)
        if pointwise:
            self.pointwise = nn.Conv2d(in_features,
                                        out_features,
                                        kernel_size=(1, 1),
                                        stride=(1, 1),
                                        padding=(0, 0),
                                        dilation=(1, 1),
                                        bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise:
            x = self.pointwise(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class depthwise_projection(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 groups,
                 kernel_size=(1, 1),
                 padding=(0, 0),
                 norm_type=None,
                 activation=False,
                 pointwise=False) -> None:
        super().__init__()

        self.proj = depthwise_conv_block(in_features=in_features,
                                         out_features=out_features,
                                         kernel_size=kernel_size,
                                         padding=padding,
                                         groups=groups,
                                         pointwise=pointwise,
                                         norm_type=norm_type,
                                         activation=activation)

    def forward(self, x):
        # P = int(x.shape[1] ** 0.5)
        # x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P)
        x = self.proj(x)
        # x = einops.rearrange(x, 'B C H W -> B (H W) C')
        return x


class CrossAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, n_div=12, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        # self.norm = nn.GroupNorm(norm_groups, in_channel)
        # self.q = nn.Conv2d(in_channel, in_channel//n_div, 1, bias=False)
        # self.kv = nn.Conv2d(in_channel, in_channel//n_div * 2, 1, bias=False)
        # self.out = nn.Conv2d(in_channel, in_channel//n_div, 1)
        self.out = nn.Conv2d(in_channel//n_div, in_channel, 3, padding=1)
        #
        self.q = depthwise_projection(in_features=in_channel,
                                              out_features=in_channel//n_div,
                                              groups=in_channel//n_div)
        self.kv = depthwise_projection(in_features=in_channel,
                                              out_features= in_channel//n_div * 2,
                                              groups=in_channel//n_div * 2)
        dim=in_channel//n_div * 2
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)


    def forward(self, x, xe, n_div=12):
        batch, channel, height, width = x.shape
        n_head = self.n_head
        head_dim = channel // (n_head * n_div)
        # x = self.norm(x)
        # xe = self.norm(xe)
        query = self.q(x).view(batch, n_head, head_dim * 1, height, width)
        # kv = self.kv(xe).view(batch, n_head, head_dim * 2, xe.shape[-2], xe.shape[-1])
        kv = self.kv(xe)
        x_3x3 = self.conv_3x3(kv).view(batch, n_head, -1, height, width)
        x_5x5 = self.conv_5x5(kv).view(batch, n_head, -1, height, width)
        x_7x7 = self.conv_7x7(kv).view(batch, n_head, -1, height, width)
        key_3, value_3 = x_3x3.chunk(2, dim=2)
        key_5, value_5 = x_5x5.chunk(2, dim=2)
        key_7, value_7 = x_7x7.chunk(2, dim=2)

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key_3).contiguous() / math.sqrt(channel // n_div)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, xe.shape[-2], xe.shape[-1])
        out_3 = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value_3).contiguous()

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key_5).contiguous() / math.sqrt(channel // n_div)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, xe.shape[-2], xe.shape[-1])
        out_5 = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value_5).contiguous()

        attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key_7).contiguous() / math.sqrt(channel // n_div)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, xe.shape[-2], xe.shape[-1])
        out_7 = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value_7).contiguous()

        # key, value = kv.chunk(2, dim=2)
        # attn = torch.einsum("bnchw, bncyx -> bnhwyx", query, key).contiguous() / math.sqrt(channel//n_div)
        # attn = attn.view(batch, n_head, height, width, -1)
        # attn = torch.softmax(attn, -1)
        # attn = attn.view(batch, n_head, height, width, xe.shape[-2], xe.shape[-1])
        # out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out=out_3+out_5+out_7

        out = self.out(out.view(batch, channel//n_div, height, width))

        return out + x


class FusionConv(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv, self).__init__()
        dim = int(out_channels // factor)
        # self.attention=CrossAttention(in_channels)
        self.down = nn.Conv2d(in_channels*2, dim, kernel_size=1, stride=1)
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7x7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3)
        self.spatial_attention = SpatialAttentionModule()
        self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        # self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x1, x2):
        x_fused = torch.cat([x1, x2], dim=1)
        x_fused = self.down(x_fused)
        # x_fused = self.attention(x1, x2)
        # x_fused = self.down(x_fused)

        x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)
        # x_fused_s = x_fused * self.spatial_attention(x_fused)

        x_out = self.up(x_fused_s + x_fused_c)

        return x_out


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)

    @torch.no_grad()
    def fuse(self):
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert (m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert (m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1, 1, 1, 1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, time_emb_dim=None,kernel_size=3, stride=1,  use_hs=1):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, oup)
        ) if exists(time_emb_dim) else None

        self.identity = stride == 1 and inp == oup
        # assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                # SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                # SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x,time_emb=None):
        h=self.token_mixer(x)
        if exists(self.mlp) and exists(time_emb):
            h += self.mlp(time_emb)[:, :, None, None]
        return self.channel_mixer(h)


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)

    def forward(self, x1, x2,last=False):
        # # x2 是从低到高，x4是从高到低的设计，x2传递语义信息，x4传递边缘问题特征补充
        # x_1_2_fusion = self.fusion_1x2(x1, x2)
        # x_1_4_fusion = self.fusion_1x4(x1, x4)
        # x_fused = x_1_2_fusion + x_1_4_fusion
        x_fused = self.fusion_conv(x1, x2)
        return x_fused





class ResnetBlocks(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0):
        super().__init__()
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)

    def forward(self, x, time_emb=None):
        if exists(time_emb):
            x = self.res_block(x, time_emb)
        else:
            x = self.res_block(x, None)
        return x

class EncoderNet(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=8,
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128
    ):
        super().__init__()


        time_dim = None
        self.time_mlp = None


        num_mults = len(channel_mults)
        channel_mult = inner_channel * channel_mults[-1]
        pre_channel = channel_mult // (2**4)
        downs = [nn.Conv2d(in_channel, pre_channel, kernel_size=3, padding=1)]
        for _ in range(num_mults):
            for _ in range(0, res_blocks-1):
                downs.append(ResnetBlocks(
                    pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=min(norm_groups, pre_channel//8),
                    dropout=dropout))

            if num_mults == 1:
                downs.append(nn.PixelUnshuffle(4))

                pre_channel = pre_channel * (2**4)
            else:
                downs.append(nn.PixelUnshuffle(2))

                pre_channel = pre_channel * (2**2)
        self.downs = nn.ModuleList(downs)
        # self.fusions = nn.ModuleList(fusions)



    def forward(self, x):


        for layer in self.downs:

                x = layer(x)
        print(x.shape)
        return x


class DecoderNet(nn.Module):
    def __init__(
        self,
        in_channel=3,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=8,
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=128
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        fusions = []
        num_mults = len(channel_mults)
        channel_mult = inner_channel * channel_mults[-1]
        pre_channel = channel_mult // (2**4)
        downs = [nn.Conv2d(in_channel, pre_channel, kernel_size=3, padding=1)]
        for _ in range(num_mults):
            for _ in range(0, res_blocks-1):
                downs.append(ResnetBlocks(
                    pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=min(norm_groups, pre_channel//8),
                    dropout=dropout))

            if num_mults == 1:
                downs.append(nn.PixelUnshuffle(4))
                for _ in range(0, res_blocks-1):
                    fusions.append(Upsample(pre_channel))
                pre_channel = pre_channel * (2**4)
            else:
                downs.append(nn.PixelUnshuffle(2))
                for _ in range(0, res_blocks-1):
                    fusions.append(Upsample(pre_channel))
                pre_channel = pre_channel * (2**2)
        self.downs = nn.ModuleList(downs)
        # self.fusions = nn.ModuleList(fusions)

        mids = []
        pre_channel = channel_mult
        for _ in range(0, attn_res):
            mids.append(ResnetBlocks(
                pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=min(norm_groups, pre_channel//8),
                dropout=dropout))
            # mids.append(RepViTBlock(
            #         pre_channel, pre_channel, pre_channel,time_emb_dim=time_dim))
            mids.append(CrossAttention(pre_channel, norm_groups=min(norm_groups, pre_channel//8)))
            # mids.append(MSAA(pre_channel, pre_channel))

        self.mid = nn.ModuleList(mids)

        ups = []
        for _ in reversed(range(num_mults)):
            for _ in range(0, res_blocks):
                ups.append(ResnetBlocks(
                    pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=min(norm_groups, pre_channel//8),
                    dropout=dropout))

            if num_mults == 1:
                ups.append(nn.PixelShuffle(4))
                pre_channel = pre_channel // (2**4)
            else:
                ups.append(nn.PixelShuffle(2))
                pre_channel = pre_channel // (2**2)
        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=min(norm_groups, pre_channel//8))


    def forward(self, x, xe, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        n = 0
        for layer in self.downs:
            if isinstance(layer, ResnetBlocks):
            # if isinstance(layer,RepViTBlock ):
                x = layer(x, t)
                xe = layer(xe, None)

                # x = x + nn.functional.interpolate(xe, scale_factor=8, mode='bilinear', align_corners=False)
                # x = x + self.fusions[n](xe)
                x = x + xe
                n = n + 1
            else:
                x = layer(x)
                xe = layer(xe)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocks):
            # if isinstance(layer, RepViTBlock):
                x = layer(x, t)
                xe = layer(xe, None)

            else:
                x = layer(x, xe)

        for layer in self.ups:
            # if isinstance(layer, ResnetBlocks):
            if isinstance(layer, ResnetBlocks):
                # x = layer(x, t)
                x = layer(x,t)
            else:
                x = layer(x)

        return self.final_conv(x)


if __name__ == '__main__':
    net = EncoderNet(
        in_channel=3,
        out_channel=3,
        inner_channel=32,
        channel_mults=[12, 12],
        norm_groups=32,
        res_blocks=2,
        dropout=0.1,
        image_size=256
    )
    x = torch.randn(1,3,256,256)

    out = net(x)
    # from thop import profile
    # from thop import clever_format
    # flops, params = profile(net, inputs=(x, xe, t))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)
    print('done')