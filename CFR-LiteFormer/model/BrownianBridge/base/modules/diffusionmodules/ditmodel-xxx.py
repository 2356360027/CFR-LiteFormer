# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from einops import rearrange, repeat
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from einops import einops
from timm.models.vision_transformer import PatchEmbed, Mlp,Attention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torchvision.models.mobilenetv2 import ConvBNReLU


def modulate(x, shift, scale):
    # print(scale.unsqueeze(1).shape)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        # print(freqs.shape)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

#
# class LabelEmbedder(nn.Module):
#     """
#     Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
#     """
#     def __init__(self, num_classes, hidden_size, dropout_prob):
#         super().__init__()
#         use_cfg_embedding = dropout_prob > 0
#         self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
#         self.num_classes = num_classes
#         self.dropout_prob = dropout_prob
#
#     def token_drop(self, labels, force_drop_ids=None):
#         """
#         Drops labels to enable classifier-free guidance.
#         """
#         if force_drop_ids is None:
#             drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
#         else:
#             drop_ids = force_drop_ids == 1
#         labels = torch.where(drop_ids, self.num_classes, labels)
#         return labels
#
#     def forward(self, labels, train, force_drop_ids=None):
#         use_dropout = self.dropout_prob > 0
#         if (train and use_dropout) or (force_drop_ids is not None):
#             labels = self.token_drop(labels, force_drop_ids)
#         embeddings = self.embedding_table(labels)
#         return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################
#
# class AgentAttention(nn.Module):
#     def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
#                  sr_ratio=1, agent_num=49, **kwargs):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#
#         self.dim = dim
#         self.num_patches = num_patches
#         window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
#         self.window_size = window_size
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)
#
#         self.agent_num = agent_num
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, H, W):
#         b, n, c = x.shape
#         num_heads = self.num_heads
#         head_dim = c // num_heads
#         q = self.q(x)
#
#         if self.sr_ratio > 1:
#             x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
#             x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
#         else:
#             kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
#         k, v = kv[0], kv[1]
#
#         agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
#         q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
#
#         kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
#         position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(b, n, c)
#         v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
#         if self.sr_ratio > 1:
#             v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
#         x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
# class FocusedLinearAttention(nn.Module):
#     def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
#                  focusing_factor=3, kernel_size=5):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)
#
#         self.focusing_factor = focusing_factor
#         self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
#                              groups=head_dim, padding=kernel_size // 2)
#         self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
#         self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
#         print('Linear Attention sr_ratio{} f{} kernel{}'.
#               format(sr_ratio, focusing_factor, kernel_size))
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         q = self.q(x)
#
#         if self.sr_ratio > 1:
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
#         else:
#             kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
#         k, v = kv[0], kv[1]
#         n = k.shape[1]
#
#         k = k + self.positional_encoding
#         focusing_factor = self.focusing_factor
#         kernel_function = nn.ReLU()
#         scale = nn.Softplus()(self.scale)
#         q = kernel_function(q) + 1e-6
#         k = kernel_function(k) + 1e-6
#         q = q / scale
#         k = k / scale
#         q_norm = q.norm(dim=-1, keepdim=True)
#         k_norm = k.norm(dim=-1, keepdim=True)
#         q = q ** focusing_factor
#         k = k ** focusing_factor
#         q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
#         k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
#
#         q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
#         k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
#         v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
#
#         z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
#         # print("Z",z.shape)
#         kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
#         x = q @ kv * z
#         # print("x.shape", x.shape)
#         # print(" q @ kv",( q @ kv).shape)
#
#         if self.sr_ratio > 1:
#             v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)
#         x = x.transpose(1, 2).reshape(B, N, C)
#         # print("x.shape", x.shape)
#         v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x

class FocusedLinearAttention(nn.Module):
    def \
            __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale_factor = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.w_g = nn.Parameter(torch.randn(dim, 1))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x)
        k= kv
        n = k.shape[1]
        # print(k.shape)

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        # print("q",q.shape)
        # print("k",k.shape)

        query_weight = q @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1

        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1

        G = torch.sum(A * q, dim=1)  # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=k.shape[1]
        )  # BxNxD
        # x = self.proj(G * k) + q  # BxNxD
        x = (G * k) # BxNxD

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
        # v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
        # print("q", q.shape)
        # print("k", k.shape)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        # z=z.reshape(B, C, -1).permute(0, 2, 1)
        # k=k.reshape(B, C, -1).permute(0, 2, 1)


        x=x.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        x=x*z
        x = x.transpose(1, 2).reshape(B, N, C)
        # q=q.reshape(B, C, N).permute(0, 2, 1)
        # q = q.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        #
        #
        # x=x+self.dwc(q).reshape(B, C, N).permute(0, 2, 1)

        # if self.sr_ratio > 1:
        #     v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)
        x = x.transpose(1, 2).reshape(B, N, C)

        # print(k.shape)
        # k = k.reshape(B, C, N).permute(0, 2, 1)
        # t = k[:,:1, :]
        # k = k[:, 1:, :]
        # x=x[:,1:,:]
        # print(x.shape)
        # print(k.shape)
        k = k.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        x = x + self.dwc(k).reshape(B, C, N).permute(0, 2, 1)
        # x=torch.cat([t, x], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


# class FocusedLinearAttention(nn.Module):
#     def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
#                  focusing_factor=3, kernel_size=5):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#
#         self.scale_factor = head_dim ** -0.5
#
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv = nn.Linear(dim, dim , bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.w_g = nn.Parameter(torch.randn(dim, 1))
#
#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)
#
#         self.focusing_factor = focusing_factor
#         self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
#                              groups=head_dim, padding=kernel_size // 2)
#         self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
#         self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
#         print('Linear Attention sr_ratio{} f{} kernel{}'.
#               format(sr_ratio, focusing_factor, kernel_size))
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         q = self.q(x)
#
#         if self.sr_ratio > 1:
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
#         else:
#             kv = self.kv(x)
#         k= kv
#         n = k.shape[1]
#         # print(k.shape)
#
#         k = k + self.positional_encoding
#         focusing_factor = self.focusing_factor
#         kernel_function = nn.ReLU()
#         scale = nn.Softplus()(self.scale)
#         q = kernel_function(q) + 1e-6
#         k = kernel_function(k) + 1e-6
#         q = q / scale
#         k = k / scale
#         q_norm = q.norm(dim=-1, keepdim=True)
#         k_norm = k.norm(dim=-1, keepdim=True)
#         q = q ** focusing_factor
#         k = k ** focusing_factor
#         q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
#         k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
#         # print("q",q.shape)
#         # print("k",k.shape)
#
#         query_weight = q @ self.w_g  # BxNx1 (BxNxD @ Dx1)
#         A = query_weight * self.scale_factor  # BxNx1
#
#         A = torch.nn.functional.normalize(A, dim=1)  # BxNx1
#
#         G = torch.sum(A * q, dim=1)  # BxD
#
#         G = einops.repeat(
#             G, "b d -> b repeat d", repeat=k.shape[1]
#         )  # BxNxD
#         # x = self.proj(G * k) + q  # BxNxD
#         x = (G * k) # BxNxD
#
#         q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
#         k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
#         # v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
#         # print("q", q.shape)
#         # print("k", k.shape)
#
#         z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
#         # z=z.reshape(B, C, -1).permute(0, 2, 1)
#         # k=k.reshape(B, C, -1).permute(0, 2, 1)
#
#
#         x=x.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
#
#         x=x*z
#         x = x.transpose(1, 2).reshape(B, N, C)
#         # q=q.reshape(B, C, N).permute(0, 2, 1)
#         # q = q.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
#         #
#         #
#         # x=x+self.dwc(q).reshape(B, C, N).permute(0, 2, 1)
#
#         # if self.sr_ratio > 1:
#         #     v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)
#         x = x.transpose(1, 2).reshape(B, N, C)
#         # k = k.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
#         # x = x + self.dwc(k).reshape(B, C, N).permute(0, 2, 1)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x

def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*b, n, c)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class tFocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale_factor = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.w_g = nn.Parameter(torch.randn(dim, 1))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)

        kv = self.kv(x)
        k= kv
        n = k.shape[1]
        # print(k.shape)
        query_weight = q @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1

        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1

        G = torch.sum(A * q, dim=1)  # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=k.shape[1]
        )  # BxNxD
        x = (G * k)  # BxNxD

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        # print("q",q.shape)
        # print("k",k.shape)


        # x = (G * k) # BxNxD

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)


        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)


        x=x.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        x=x*z
        x = x.transpose(1, 2).reshape(B, N, C)

        # q = q.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        #
        #
        # x=x+self.dwc(q).reshape(B, C, N).permute(0, 2, 1)

        # if self.sr_ratio > 1:
        #     v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)
        # x = x.transpose(1, 2).reshape(B, N, C)
        # q = q.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        # x = x + self.dwc(q).reshape(B, C, N).permute(0, 2, 1)
        # q = q.reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class sparseattention(nn.Module):

    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale_factor = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.w_g = nn.Parameter(torch.randn(dim, 1))

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, 256, dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x)
        k = kv
        n = k.shape[1]
        # print(k.shape)

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        # print("q",q.shape)
        # print("k",k.shape)

        query_weight = q @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        A = query_weight * self.scale_factor  # BxNx1

        A = torch.nn.functional.normalize(A, dim=1)  # BxNx1

        G = torch.sum(A * q, dim=1)  # BxD

        G = einops.repeat(
            G, "b d -> b repeat d", repeat=k.shape[1]
        )  # BxNxD

        x = (G * k)  # BxNxD

        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
        # v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3)
        # print("q", q.shape)
        # print("k", k.shape)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        # z=z.reshape(B, C, -1).permute(0, 2, 1)
        # k=k.reshape(B, C, -1).permute(0, 2, 1)

        x = x.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        x = x * z
        x = x.transpose(1, 2).reshape(B, N, C)
        # q = q.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        #
        #
        # x=x+self.dwc(q).reshape(B, C, N).permute(0, 2, 1)

        # if self.sr_ratio > 1:
        #     v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N, mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)
        # x = x.transpose(1, 2).reshape(B, N, C)
        k = k.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
        x = x + self.dwc(k).reshape(B, C, N).permute(0, 2, 1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AgentAttention(nn.Module):



    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=16, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim , bias=qkv_bias)
        # self.kv = nn.Linear(dim, dim , bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.w_g = nn.Parameter(torch.randn(dim , 1))

        self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm = nn.LayerNorm(dim)
        #
        # self.mapping=nn.Linear(dim , 1024)
        self.Proj = nn.Linear(dim, dim)


        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)

        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        # query = self.to_query(x)
        # key = self.to_key(x)
        #
        # query = torch.nn.functional.normalize(query, dim=-1)  # BxNxD
        # key = torch.nn.functional.normalize(key, dim=-1)  # BxNxD
        #
        # query_weight = query @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        # A = query_weight * self.scale_factor  # BxNx1
        # print("query_weight:", G.shape)
        #
        # A = torch.nn.functional.normalize(A, dim=1)  # BxNx1
        #
        # G = torch.sum(A * query, dim=1)  # BxD
        # print("G—1:", G.shape)
        #
        # G = einops.repeat(
        #     G, "b d -> b repeat d", repeat=key.shape[1]
        # )  # BxNxD
        # print("G:",G.shape)
        # print("key:", key.shape)
        #
        #
        # out = self.Proj(G * key) + query  # BxNxD
        # print("out:", out.shape)
        #
        # out = self.final(out)  # BxNxD
        #
        # return out
        b, n, c = x.shape
        # print(x.shape)
        # x=x[:,1:,:]
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)
        # print("q",q.shape)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            # kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
            kv = self.kv(x)
        k = kv
        # print("k", k.shape)
        # q= q[:,1:,:]
        # print(q.shape)
        # k = k[:, 1:, :]
        agent_tokens_q = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        agent_tokens_k = self.pool(k.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        # v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)


        # print(agent_tokens_q.shape)
        # print(self.w_g)


        agent_w_q=agent_tokens_q @ self.w_g
        # agent_w_1=q@self.w_g

        A=agent_w_q*self.scale
        G=torch.sum(A * agent_tokens_q, dim=1)
        # print("A", A.shape)
        # print("G", G.shape)
        # G_1 =einops.repeat(
        #     G, "b d -> b repeat d", repeat=q.shape[1]
        # )  # BxNxD
        G = einops.repeat(
            G, "b d -> b repeat d", repeat=agent_tokens_k.shape[1]
        )  # BxNxD

        # print("G", G.shape)
        # print("agent_tokens_k", agent_tokens_k.shape)
        # print("agent_tokens_q", agent_tokens_q.shape)
        agent_new=self.Proj(G * agent_tokens_k) + agent_tokens_q
        # print("agent_new", agent_new.shape)
        # attn=q*agent_new
        # print("attn", attn.shape)
        agent_new = agent_new.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens_k=agent_tokens_k.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        # print("agent_new", agent_new.shape)
        # print("agent_tokens_q", agent_tokens_q.shape)
        # agent_all=agent_tokens_k+agent_tokens_q
        # # k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        q_attn = self.softmax((q * self.scale) @ agent_new.transpose(-2, -1) )
        # print("q:",q.shape)
        # print("agent_token_k",agent_tokens_k.shape)
        q_attn = self.attn_drop(q_attn)
        # print("q_attn", q_attn.shape)
        attn=q_attn@agent_tokens_k
        # print(attn.shape)





        # agent_k=self.softmax(agent_new @ k.transpose(-2, -1))
        # print("agent_k", agent_k.shape)
        #
        # q_attn = self.softmax((q * self.scale) @ agent_new.transpose(-2, -1) )
        # q_attn = self.attn_drop(q_attn)
        # print("q_attn", q_attn.shape)
        #
        # agent_v =agent_k @ v
        # attn=q_attn @ agent_v



        # print("agent_v", agent_v.shape)
        # print("attn", attn.shape)


        # agent_tokens_k=agent_tokens_k.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        # q_attn = self.softmax((q * self.scale) @ agent_tokens_k.transpose(-2, -1))
        # print("q_attn", q_attn.shape)
        #
        #
        # attn_out=q_attn @ attn
        # print("attn_out", attn_out.shape)




        x = attn.transpose(1, 2).reshape(b, n, c)
        # print("x",x.shape)
        # k = k.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
        # if self.sr_ratio > 1:
        #     v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        # x = x + self.dwc(k).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.Proj(x)
        x = self.proj_drop(x)
        return x
        # b, n, c = x.shape
        # num_heads = self.num_heads
        # head_dim = c // num_heads
        # q = self.q(x)
        # print("q:", q.shape)
        #
        # query_weight = q @ self.w_g  # BxNx1 (BxNxD @ Dx1)
        # A = query_weight * self.scale  # BxNx1
        # print("A:", A.shape)
        #
        # A = torch.nn.functional.normalize(A, dim=1)  # BxNx1
        #
        # print("A:", A.shape)
        # G = torch.sum(A * q, dim=1)  # BxD
        #
        #
        # k = self.kv(x).reshape(b, -1, 1, c).permute(2, 0, 1, 3)
        # print("k:", k.shape)
        # print("G:", G.shape)
        #
        #
        # agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        # q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        # k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        # agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
        # print("agent_tokens:", G.shape)
        #
        # kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        # position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        # position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # position_bias = position_bias1 + position_bias2
        #
        #
        # agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        # agent_attn = self.attn_drop(agent_attn)
        #
        #
        # print("agent_token", agent_tokens.shape, "k", k.shape,"position_bias", position_bias.shape)
        #
        # agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        # agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        # agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        # agent_bias = agent_bias1 + agent_bias2
        # print("agent_attn", agent_attn.shape,"q", q.shape,"agent_bias", agent_bias.shape)
        #
        #
        #
        # x = self.Proj( q @ agent_attn) + q+agent_bias
        #
        # q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        # q_attn = self.attn_drop(q_attn)
        # x = q_attn @ agent_v
        #
        # x = x.transpose(1, 2).reshape(b, n, c)
        # # v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
        # # if self.sr_ratio > 1:
        # #     v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        # x = x + self.dwc(agent_attn).permute(0, 2, 3, 1).reshape(b, n, c)
        #
        # x = self.proj(x)
        # x = self.proj_drop(x)
        # return x
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
        # self.channel_attention = ChannelAttentionModule(dim)
        self.up = nn.Conv2d(dim, out_channels, kernel_size=1, stride=1)
        # self.down_2 = nn.Conv2d(in_channels, dim, kernel_size=1, stride=1)

    def forward(self, x1, x2,H,W):
        x1=x1.permute(0,2,1).reshape(1,384,H,W)
        x2 = x2.permute(0, 2, 1).reshape(1, 384, H, W)
        x_fused = torch.cat([x1, x2], dim=1)
        x_fused = self.down(x_fused)
        # x_fused = self.attention(x1, x2)
        # x_fused = self.down(x_fused)

        # x_fused_c = x_fused * self.channel_attention(x_fused)
        x_3x3 = self.conv_3x3(x_fused)
        x_5x5 = self.conv_5x5(x_fused)
        x_7x7 = self.conv_7x7(x_fused)
        x_fused_s = x_3x3 + x_5x5 + x_7x7
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)
        # x_fused_s = x_fused * self.spatial_attention(x_fused)

        x_out = self.up(x_fused_s)

        return x_out
class MSAA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MSAA, self).__init__()
        self.fusion_conv = FusionConv(in_channels, out_channels)

    def forward(self, x1, x2,H,W,Wlast=False):
        # # x2 是从低到高，x4是从高到低的设计，x2传递语义信息，x4传递边缘问题特征补充
        # x_1_2_fusion = self.fusion_1x2(x1, x2)
        # x_1_4_fusion = self.fusion_1x4(x1, x4)
        # x_fused = x_1_2_fusion + x_1_4_fusion
        x_fused = self.fusion_conv(x1, x2,H,W)
        return x_fused

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B, L, C = x.shape
#
#         qkv = self.qkv(x)
#         if ATTENTION_MODE == 'flash':
#             qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads).float()
#             q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
#             x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
#             x = einops.rearrange(x, 'B H L D -> B L (H D)')
#         elif ATTENTION_MODE == 'xformers':
#             qkv = einops.rearrange(qkv, 'B L (K H D) -> K B L H D', K=3, H=self.num_heads)
#             q, k, v = qkv[0], qkv[1], qkv[2]  # B L H D
#             x = xformers.ops.memory_efficient_attention(q, k, v)
#             x = einops.rearrange(x, 'B L H D -> B L (H D)', H=self.num_heads)
#         elif ATTENTION_MODE == 'math':
#             qkv = einops.rearrange(qkv, 'B L (K H D) -> K B H L D', K=3, H=self.num_heads)
#             q, k, v = qkv[0], qkv[1], qkv[2]  # B H L D
#             attn = (q @ k.transpose(-2, -1)) * self.scale
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = (attn @ v).transpose(1, 2).reshape(B, L, C)
#         else:
#             raise NotImplemented
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
## 获取轻量的自高斯注意力
class LSGAttention(nn.Module):
    def __init__(self, dim, att_inputsize, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.att_inputsize = att_inputsize[0]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)  # 线性层
        self.attn_drop = nn.Dropout(attn_drop)  # 随机丢弃一些线性神经元，防止过拟合
        self.proj = nn.Linear(dim, dim)  # 线性层
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        totalpixel = self.att_inputsize * self.att_inputsize
        gauss_coords_h = torch.arange(totalpixel) - int((totalpixel - 1) / 2)
        gauss_coords_w = torch.arange(totalpixel) - int((totalpixel - 1) / 2)
        gauss_x, gauss_y = torch.meshgrid([gauss_coords_h, gauss_coords_w])
        sigma = 10
        gauss_pos_index = torch.exp(torch.true_divide(-(gauss_x ** 2 + gauss_y ** 2), (2 * sigma ** 2)))  # 二维高斯函数
        self.register_buffer("gauss_pos_index", gauss_pos_index)
        self.token_wA = nn.Parameter(torch.empty(1, self.att_inputsize * self.att_inputsize, dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, dim, dim),
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        # print(B_, N, C)
        wa = repeat(self.token_wA, '() n d -> b n d', b=B_)  # wa (bs 4 64)
        wa = rearrange(wa, 'b h w -> b w h')  # Transpose # wa (bs 64 4)
        A = torch.einsum('bij,bjk->bik', x, wa)  # A (bs 81 4)
        A = rearrange(A, 'b h w -> b w h')  # Transpose # A (bs 4 81)
        A = A.softmax(dim=-1)
        VV = repeat(self.token_wV, '() n d -> b n d', b=B_)  # VV(bs,64,64)
        VV = torch.einsum('bij,bjk->bik', x, VV)  # VV(bs,81,64)
        x = torch.einsum('bij,bjk->bik', A, VV)  # T(bs,4,64)

        absolute_pos_bias = self.gauss_pos_index.unsqueeze(0)  # 获取高斯绝对位置信息
        # print(self.att_inputsize)
        q = self.qkv(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 分支头q进行线性变换
        k = x.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # 分支头k，v直接输入一个x
        v = x.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = q * self.scale  # 除以根号d，对注意力权重进行缩放，以确保数值的稳定性
        attn = (q @ k.transpose(-2, -1))  # 矩阵乘法，计算相似性矩阵
        attn = attn + absolute_pos_bias.unsqueeze(0)  # 融合高斯绝对位置信息
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)  # softmax函数，进行归一化处理
        attn = self.attn_drop(attn)  # 随机丢弃一些线性神经元，防止过拟合
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # 融合注意力
        x = self.proj(x)  # 最后再通过一个线性层
        x = self.proj_drop(x)  # 随机丢弃，防止过拟合
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, num_patches,skip=False, down_factor=4,mlp_ratio=4.0, downsampler=None, down_shortcut=False,**block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.attn = sparseattention(
        #     hidden_size,hidden_size,num_heads=num_heads,)
        self.attn= FocusedLinearAttention(
                hidden_size, num_patches,
                num_heads=num_heads, qkv_bias=True, qk_scale=None,
                attn_drop=0., proj_drop=0., sr_ratio=1,
                focusing_factor=3, kernel_size=5)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        # print(num_patches)
        # self.attn = LSGAttention(
        #     hidden_size, att_inputsize=(attn_size,attn_size), num_heads=num_heads,
        #     qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)

        # self.attn = AgentAttention(
        #         hidden_size, num_patches,
        #         num_heads=num_heads, qkv_bias=False, qk_scale=None,
        #         attn_drop=0., proj_drop=0, sr_ratio=1,
        #         agent_num=16)
        # self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        # self.skip_linear =nn.Linear(2*hidden_size,hidden_size) if skip else None
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # downsampler
        # downsampler_ker_size = int(downsampler[-1])
        # downsampler_padding = (int(downsampler[-1]) - 1) // 2
        # self.downsampler = DownSampler(hidden_size, hidden_size, kernel_size=downsampler_ker_size, stride=1,
        #                                padding=downsampler_padding, groups=hidden_size, down_factor=down_factor,
        #                                downsampler=downsampler, down_shortcut=down_shortcut)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU()
        # self.mlp1 =MSCFFN_step1(hidden_size)
        # self.mlp2=MSCFFN_step2(hidden_size)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim)
        # self.mrfp = MRFP(hidden_size, hidden_features=mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6* hidden_size, bias=True)
        )

    def forward(self, x, c,H,W,skip=None):
        # print(c.shape)
        # print("adaln输入的参数")

        # gate_msa, gate_mlp= self.adaLN_modulation(c).chunk(2, dim=1)
        if skip is not None:
            # print(".....")
            # x = self.mrfp(skip, H, W)
            # x_select1, x_select2, x_select3 = x[:, :H * W * 4, :], x[:, H * W * 4:H * W * 4 + H * W, :], x[:,
            #                                                                                              H * W * 4 + H * W:,
            #                                                                                              :]
            # skip = torch.cat([x_select1, x_select2 + x, x_select3], dim=1)

            # x = self.cti_tov(query=x, reference_points=deform_inputs[0],
            #                  feat=c, spatial_shapes=deform_inputs[1],
            #                  level_start_index=deform_inputs[2], H=H, W=W)
            # x = self.skip_linear(torch.cat([x, skip], dim=-1))
            x=x+skip
            # print(x.shape)
            # print(skip.shape)
            # x = self.skip_linear(x,skip,H,W)
            # x=x.reshape(1,384,-1).permute(0,2,1)
        # print(shift_mlp.shape, gate_mlpshape, gate_mlp.shape)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        # x = x + self.attn(self.norm1(x),H,W)
        # x = x + self.mlp(self.norm2(x))
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa),H,W)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # self.adaLN_modulation = nn.Sequential(
        #     nn.SiLU(),
        #     nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        # )

    def forward(self, x, c):
        # shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class MultiScaleDWConv(nn.Module):
    def __init__(self, dim, scale=(1, 3, 5, 7)):
        super().__init__()
        self.scale = scale
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(scale)):
            if i == 0:
                channels = dim - dim // len(scale) * (len(scale) - 1)
            else:
                channels = dim // len(scale)
            conv = nn.Conv2d(channels, channels,
                             kernel_size=scale[i],
                             padding=scale[i] // 2,
                             groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x



#
# class Mlp(nn.Module):  ### MS-FFN
#     """
#     Mlp implemented by with 1x1 convolutions.
#
#     Input: Tensor with shape [B, C, H, W].
#     Output: Tensor with shape [B, C, H, W].
#     Args:
#         in_features (int): Dimension of input features.
#         hidden_features (int): Dimension of hidden features.
#         out_features (int): Dimension of output features.
#         act_cfg (dict): The config dict for activation between pointwise
#             convolution. Defaults to ``dict(type='GELU')``.
#         drop (float): Dropout rate. Defaults to 0.0.
#     """
#
#     def __init__(self,
#                  in_features,
#                  hidden_features=None,
#                  out_features=None,
#                  act_cfg=dict(type='GELU'),
#                  drop=0, ):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Sequential(
#             nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=False),
#             build_activation_layer(act_cfg),
#             nn.BatchNorm2d(hidden_features),
#         )
#         self.dwconv = MultiScaleDWConv(hidden_features)
#         self.act = build_activation_layer(act_cfg)
#         self.norm = nn.BatchNorm2d(hidden_features)
#         self.fc2 = nn.Sequential(
#             nn.Conv2d(hidden_features, in_features, kernel_size=1, bias=False),
#             nn.BatchNorm2d(in_features),
#         )
#         self.drop = nn.Dropout(drop)
#
#     def forward(self, x,h,w):
#         # print(x.shape)
#         b, n, c = x.shape
#         x=x.view(b,c,h,w)
#
#         x = self.fc1(x)
#
#         x = self.dwconv(x) + x
#         x = self.norm(self.act(x))
#
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         x=x.view(b,n,c)
#
#         return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.h=input_size//patch_size
        self.w=input_size//patch_size
        self.hidden_size = hidden_size
        print(self.h, self.w)
        self.extras=1



        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # self.y_conv=nn.Conv2d(in_channels, hidden_size, 1)
        # self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = (input_size // patch_size) ** 2
        print("num_patch",num_patches)

        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        #
        # self.blocks = nn.ModuleList([
        #     DiTBlock(hidden_size, num_heads, num_patches,mlp_ratio=mlp_ratio) for _ in range(depth)
        # ])

        self.in_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads,num_patches, mlp_ratio=mlp_ratio)
            for _ in range(depth // 2)])

        self.mid_block =  DiTBlock(hidden_size, num_heads, num_patches,mlp_ratio=mlp_ratio)

        self.out_blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads,num_patches,skip=True,mlp_ratio=mlp_ratio)
            for _ in range(depth // 2)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        # self.last_layer = nn.Conv2d(self.in_channels, self.in_channels, 3, padding=1)
        self.initialize_weights()
        # trunc_normal_(self.pos_embed, std=.02)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int((self.x_embedder.num_patches+1) ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        # nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        #
        # # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.in_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.mid_block.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.mid_block.adaLN_modulation[-1].bias, 0)
        for block in self.out_blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        # print(map.shape)

        # print(time_token.shape)
        #
        # print(x.shape)
        x=self.x_embedder(x)+self.pos_embed
        # print(t.shape)
        t = self.t_embedder(t)



        # t= t.unsqueeze(dim=1)
        # x = torch.cat((t, x), dim=1)



        # print(x.shape)

        # (N, T, D), where T = H * W / patch_size ** 2
        # y=self.x_embedder(cond)
        # y=y.sum(dim=1)
        # y=y.view(y.shape[0], -1)
        # t = self.t_embedder(t)
        # (N, D)
        # t=t.unsqueeze(0).expand(-1,1024,-1)

        # print(t.shape)
        # y = self.y_embedder(y, self.training)    # (N, D)
        c = t                             # (N, D)
        # c=t
        # x = self.downsampler(x)
        skips = []
        for blk in self.in_blocks:
            x = blk(x,c,self.h,self.w)
            skips.append(x)

        x = self.mid_block(x,c,self.h,self.w)

        for blk in self.out_blocks:
            x = blk(x,c,self.h,self.w ,skips.pop())
        # for block in self.blocks:
        #     x = block(x, c,self.h,self.w)                      # (N, T, D)
        # x = x[:, self.extras:, :]
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        # x=self.last_layer(x)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)
def DiT_m_4(**kwargs):
    return DiT(depth=12, hidden_size=512, patch_size=4, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_d_2(**kwargs):
    return DiT(depth=12, hidden_size=512, patch_size=2, num_heads=8, **kwargs)

def DiT_S_d_4(**kwargs):
    return DiT(depth=12, hidden_size=512, patch_size=4, num_heads=8, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
    'DiT-S/D/4':  DiT_S_d_4,'DiT-m/4':  DiT_m_4,'DiT-S/D/2':  DiT_S_d_2
}
