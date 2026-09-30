import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module, Parameter, init
# ==============================================================================
# 1. 替代梯度与脉冲神经元 (Surrogate Gradient & LIF Neuron)
# ==============================================================================
class Surrogate_BP_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        # temp = torch.abs(1 - torch.abs(torch.arcsin(input))) < 0.7
        temp = (1 / 2.5) * torch.sign(abs(input) < 2.5)
        return grad_input * temp.float()
class LIFNode(nn.Module):
    def __init__(self, tau=1.0, detach_reset=True, backend='cupy', v_threshold=1.0):
        super().__init__()
        self.spike_fn = Surrogate_BP_Function.apply
        self.tau = nn.Parameter(torch.tensor([tau], dtype=torch.float))
        self.v_threshold = nn.Parameter(torch.tensor([1.0]))
        self.detach_reset = detach_reset

    def forward(self, x_seq: torch.Tensor):
        spike_seq = []
        mem = torch.zeros_like(x_seq[0])
        decay = torch.sigmoid(self.tau)
        for t in range(x_seq.shape[0]):
            mem = decay * mem + x_seq[t]
            # 2. 脉冲发放：当膜电位超过阈值时发放脉冲
            mem_thr = mem - self.v_threshold
            x = self.spike_fn(mem_thr)
            if self.detach_reset:
                mem = mem - (x * self.v_threshold).detach()
            else:
                mem = mem - x * self.v_threshold
            spike_seq.append(x.unsqueeze(0))
        # 拼接时间维度，返回 [T, B, C, H, W] 的脉冲张量
        spike_seq = torch.cat(spike_seq, 0)
        return spike_seq

class ComplexDropout(nn.Module):
    def __init__(self, p=0.1, mode='spatial'):
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x_r, x_i):
        if not self.training or self.p == 0.0:
            return x_r, x_i

        if self.mode == 'spatial':
            # 针对 [T, B, N, D]，在 T 和 D 维度广播，随机 Drop 掉整条时间线上的部分 Token
            mask_shape = (1, x_r.shape[1], x_r.shape[2], 1)
        else:
            mask_shape = x_r.shape

        mask = (torch.rand(mask_shape, device=x_r.device) >= self.p).to(x_r.dtype) / (1.0 - self.p)
        return x_r * mask, x_i * mask

class ComplexSpikingLinear(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.fc_r = nn.Linear(dim, dim, bias=False)
        self.fc_i = nn.Linear(dim, dim, bias=False)
        self.bn_r = nn.BatchNorm1d(dim)
        self.bn_i = nn.BatchNorm1d(dim)

        # 独立的 LIF 神经元处理实部和虚部
        self.lif_r = LIFNode()
        self.lif_i = LIFNode()

    def _bn_forward(self, x, bn):
        TB, N, D = x.shape
        return bn(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x_r, x_i):
        T, B, N, D = x_r.shape
        x_r_flat = x_r.flatten(0, 1)
        x_i_flat = x_i.flatten(0, 1)

        # 遵循复数乘法: (Wr + iWi) * (Xr + iXi) = (WrXr - WiXi) + i(WrXi + WiXr)
        out_r = self.fc_r(x_r_flat) - self.fc_i(x_i_flat)
        out_i = self.fc_r(x_i_flat) + self.fc_i(x_r_flat)

        out_r = self._bn_forward(out_r, self.bn_r).reshape(T, B, N, D)
        out_i = self._bn_forward(out_i, self.bn_i).reshape(T, B, N, D)

        return self.lif_r(out_r), self.lif_i(out_i)

class SpikingPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, time_steps=4):
        super().__init__()
        self.time_steps = time_steps
        self.embed_dim = embed_dim

        # 第一层空间特征映射
        self.proj1 = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        # 用真正的脉冲神经元替代 ReLU
        self.lif1 = LIFNode()

        self.proj2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        # 用真正的脉冲神经元替代 ReLU
        self.lif2 = LIFNode()

        # 将平均池化替换为最大池化
        self.pool = nn.AdaptiveMaxPool2d((4, 4))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        x1 = self.bn1(self.proj1(x.flatten(0, 1)))  # [B, embed_dim//2, H, W]

        spike_seq1 = self.lif1(x1.reshape(self.time_steps, B, self.embed_dim // 2, H, W))  # 经过 LIF1 激发为纯二值脉冲 [T, B, embed_dim//2, H, W]

        x2 = self.bn2(self.proj2(spike_seq1.flatten(0, 1)))

        spike_seq2 = self.lif2(x2.reshape(self.time_steps, B, self.embed_dim, H, W))

        x2_pool = self.pool(spike_seq2.flatten(0, 1))  # 空间归约 [T*B, embed_dim, 4, 4]

        x2_seq = x2_pool.reshape(self.time_steps, B,self.embed_dim, 4, 4)  # [T, B, embed_dim, 4, 4]
        # x2_seq = x2_pool.reshape(self.time_steps, B,self.embed_dim, 6, 6)  # [T, B, embed_dim, 4, 4]

        tokens = x2_seq.flatten(3)  # [T, B, D, 16]
        # 交换维度让 D 处于最后
        tokens = tokens.permute(0, 1, 3, 2)  # [T, B, 16, D]
        return tokens

class SpikingPatchEmbed_lidar(nn.Module):
    def __init__(self, in_channels, embed_dim, time_steps=4):
        super().__init__()
        self.time_steps = time_steps
        self.embed_dim = embed_dim

        # 第一层空间特征映射
        self.proj1 = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim // 2)
        # 用真正的脉冲神经元替代 ReLU
        self.lif1 = LIFNode()

        self.proj2 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dim)
        # 用真正的脉冲神经元替代 ReLU
        self.lif2 = LIFNode()

        # 将平均池化替换为最大池化
        self.pool = nn.AdaptiveMaxPool2d((4, 4))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        x1 = self.bn1(self.proj1(x.flatten(0, 1)))  # [B, embed_dim//2, H, W]
        spike_seq1 = self.lif1(x1.reshape(self.time_steps, B, self.embed_dim // 2, H, W))  # 经过 LIF1 激发为纯二值脉冲 [T, B, embed_dim//2, H, W]

        x2 = self.bn2(self.proj2(spike_seq1.flatten(0, 1)))
        spike_seq2 = self.lif2(x2.reshape(self.time_steps, B, self.embed_dim, H, W))

        x2_pool = self.pool(spike_seq2.flatten(0, 1))  # 空间归约 [T*B, embed_dim, 4, 4]

        x2_seq = x2_pool.reshape(self.time_steps, B,self.embed_dim, 4, 4)  # [T, B, embed_dim, 4, 4]

        tokens = x2_seq.flatten(3)  # [T, B, D, 16]
        # tokens = spike_seq3.flatten(3)  # [T, B, D, 16]
        # 交换维度让 D 处于最后
        tokens = tokens.permute(0, 1, 3, 2)  # [T, B, 16, D]

        return tokens
class ComplexSpikeDrivenSelfAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # self.head_dim = dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = ComplexSpikingLinear(dim)
        self.k_proj = ComplexSpikingLinear(dim)
        self.v_proj = ComplexSpikingLinear(dim)

        self.attn_lif_r = LIFNode()
        self.attn_lif_i = LIFNode()

        self.proj = ComplexSpikingLinear(dim)

    def forward(self, x_r, x_i):
        T, B, N, D = x_r.shape

        # 1. 生成复数 Q, K, V
        q_r, q_i = self.q_proj(x_r, x_i)
        k_r, k_i = self.k_proj(x_r, x_i)
        v_r, v_i = self.v_proj(x_r, x_i)

        # 重塑为多头结构
        reshape_fn = lambda x: x.reshape(T, B, N, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        q_r, q_i = reshape_fn(q_r), reshape_fn(q_i)
        k_r, k_i = reshape_fn(k_r), reshape_fn(k_i)
        v_r, v_i = reshape_fn(v_r), reshape_fn(v_i)

        # 2. 复数点积注意力 (Q * K^*)
        # K^* (共轭转置): 实部转置，虚部转置并取反
        k_r_t = k_r.transpose(-2, -1)
        k_i_t = k_i.transpose(-2, -1)

        attn_r = (torch.matmul(q_r, k_r_t) + torch.matmul(q_i, k_i_t)) * self.scale
        attn_i = (torch.matmul(q_i, k_r_t) - torch.matmul(q_r, k_i_t)) * self.scale

        # 激发注意力脉冲
        attn_r_spikes = self.attn_lif_r(attn_r.flatten(0, 1).reshape(T, B * self.num_heads, N, N)).reshape(T, B,
                                                                                                           self.num_heads,
                                                                                                           N, N)
        attn_i_spikes = self.attn_lif_i(attn_i.flatten(0, 1).reshape(T, B * self.num_heads, N, N)).reshape(T, B,
                                                                                                           self.num_heads,
                                                                                                           N, N)

        # 3. 注意力矩阵与 V 相乘 (Attn * V)
        out_r = torch.matmul(attn_r_spikes, v_r) - torch.matmul(attn_i_spikes, v_i)
        out_i = torch.matmul(attn_r_spikes, v_i) + torch.matmul(attn_i_spikes, v_r)

        out_r = out_r.permute(0, 1, 3, 2, 4).reshape(T, B, N, D)
        out_i = out_i.permute(0, 1, 3, 2, 4).reshape(T, B, N, D)

        # 4. 输出投影
        return self.proj(out_r, out_i)


class ComplexSpikingMLP(nn.Module):

    def __init__(self, dim, mlp_ratio):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        # --- 第一层复数全连接 ---
        self.fc1_r = nn.Linear(dim, hidden_dim, bias=False)
        self.fc1_i = nn.Linear(dim, hidden_dim, bias=False)
        self.bn1_r = nn.BatchNorm1d(hidden_dim)
        self.bn1_i = nn.BatchNorm1d(hidden_dim)
        self.lif1_r = LIFNode()
        self.lif1_i = LIFNode()

        # --- 第二层复数全连接 ---
        self.fc2_r = nn.Linear(hidden_dim, dim, bias=False)
        self.fc2_i = nn.Linear(hidden_dim, dim, bias=False)
        self.bn2_r = nn.BatchNorm1d(dim)
        self.bn2_i = nn.BatchNorm1d(dim)
        self.lif2_r = LIFNode()
        self.lif2_i = LIFNode()

        self.drop = nn.Dropout(0.1)

    def _bn_forward(self, x, bn):
        TB, N, D = x.shape
        x_flat = x.transpose(1, 2).reshape(TB, D, N)
        return bn(x_flat).reshape(TB, D, N).transpose(1, 2)

    def forward(self, x_r, x_i):
        T, B, N, D = x_r.shape
        x_r_flat = x_r.flatten(0, 1)
        x_i_flat = x_i.flatten(0, 1)

        # 1. 经过第一层复数映射与 LIF
        h_r = self.fc1_r(x_r_flat) - self.fc1_i(x_i_flat)
        h_i = self.fc1_r(x_i_flat) + self.fc1_i(x_r_flat)

        h_r_spikes = self.lif1_r(self._bn_forward(h_r, self.bn1_r).reshape(T, B, N, -1))
        h_i_spikes = self.lif1_i(self._bn_forward(h_i, self.bn1_i).reshape(T, B, N, -1))

        # 2. 经过第二层复数映射与 LIF
        h_r_flat = h_r_spikes.flatten(0, 1)
        h_i_flat = h_i_spikes.flatten(0, 1)

        out_r = self.fc2_r(h_r_flat) - self.fc2_i(h_i_flat)
        out_i = self.fc2_r(h_i_flat) + self.fc2_i(h_r_flat)

        out_r_spikes = self.lif2_r(self._bn_forward(out_r, self.bn2_r).reshape(T, B, N, D))
        out_i_spikes = self.lif2_i(self._bn_forward(out_i, self.bn2_i).reshape(T, B, N, D))

        return out_r_spikes, out_i_spikes


class ComplexSpikformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        # 依赖上一轮提供的 ComplexSpikeDrivenSelfAttention
        self.attn = ComplexSpikeDrivenSelfAttention(dim, num_heads)
        self.mlp = ComplexSpikingMLP(dim, mlp_ratio)

    def forward(self, x_r, x_i):
        # 1. 脉冲域复数残差连接 (注意力层)
        attn_r, attn_i = self.attn(x_r, x_i)
        x_r = x_r + attn_r
        x_i = x_i + attn_i

        # 2. 脉冲域复数残差连接 (MLP层)
        mlp_r, mlp_i = self.mlp(x_r, x_i)
        x_r = x_r + mlp_r
        x_i = x_i + mlp_i

        return x_r, x_i

class SpikingProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, proj_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(proj_dim)
        self.lif1 = LIFNode()

        self.fc2 = nn.Linear(proj_dim, in_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.lif2 = LIFNode()

    def forward(self, x_seq):
        # 输入 x_seq 形状: [T, B, D]
        T, B, D = x_seq.shape
        x_flat = x_seq.flatten(0, 1)  # 合并 T 和 B，准备进入 Linear 和 BN -> [T*B, D]

        # 第一层脉冲全连接
        x1 = self.bn1(self.fc1(x_flat))
        x1_spike = self.lif1(x1.reshape(T, B, -1))  # [T, B, D]

        # 第二层脉冲全连接
        x2_flat = x1_spike.flatten(0, 1)
        x2 = self.bn2(self.fc2(x2_flat)).reshape(T, B, -1)

        return x2
class ComplexSpikingClassificationHead(nn.Module):

    def __init__(self, fused_dim, hidden_dim, num_classes):
        super().__init__()

        # 第一层：复数线性映射 (fused_dim -> hidden_dim)
        self.fc1_r = nn.Linear(fused_dim, hidden_dim, bias=False)
        self.fc1_i = nn.Linear(fused_dim, hidden_dim, bias=False)
        self.bn_r = nn.BatchNorm1d(hidden_dim)
        self.bn_i = nn.BatchNorm1d(hidden_dim)

        # 独立的复数 LIF 神经元
        self.lif_r = LIFNode(tau=1.0)
        self.lif_i = LIFNode(tau=1.0)

        # 第二层：复数线性分类映射 (hidden_dim -> num_classes)
        self.fc2_r = nn.Linear(hidden_dim, num_classes, bias=False)
        self.fc2_i = nn.Linear(hidden_dim, num_classes, bias=False)

    def _bn_forward(self, x, bn):
        T, B, D = x.shape
        x_flat = x.reshape(T * B, D)
        return bn(x_flat).reshape(T, B, D)

    def forward(self, x_r, x_i):
        # 输入 x_r, x_i 形状: [T, B, fused_dim]
        T, B, _ = x_r.shape

        h_r = self.fc1_r(x_r) - self.fc1_i(x_i)
        h_i = self.fc1_r(x_i) + self.fc1_i(x_r)

        # 批归一化与脉冲激发
        h_r_spikes = self.lif_r(self._bn_forward(h_r, self.bn_r))
        h_i_spikes = self.lif_i(self._bn_forward(h_i, self.bn_i))

        # 类别复数映射
        o_r = self.fc2_r(h_r_spikes) - self.fc2_i(h_i_spikes)
        o_i = self.fc2_r(h_i_spikes) + self.fc2_i(h_r_spikes)

        # 计算复数模长，完成从复数空间到实数概率空间的“测量”
        # 加上 1e-6 避免在原点处梯度 NaN 的问题。可加可不加
        logits_complex_mag = torch.sqrt(o_r ** 2 + o_i ** 2 + 1e-6)  # [T, B, num_classes]

        # 时间维度上的发放率聚合
        return logits_complex_mag.mean(dim=0)

class ComplexMultimodalSpikformer(nn.Module):
    def __init__(self,hsi_channels=40,lidar_channels=1,num_classes=7,embed_dim=64,num_heads=1,depth=1,time_steps=4):
        super().__init__()
        self.time_steps = time_steps
        self.embed_dim = embed_dim

        # LiDAR的嵌入维度必须与HSI对齐，以便构成复数
        self.hsi_embed = SpikingPatchEmbed(hsi_channels, embed_dim, self.time_steps)
        self.lidar_embed = SpikingPatchEmbed_lidar(lidar_channels, embed_dim, self.time_steps)

        self.hsi_proj = SpikingProjectionHead(self.embed_dim, self.embed_dim*2)
        self.lidar_proj = SpikingProjectionHead(self.embed_dim, self.embed_dim*2)

        # self.complex_encoder = nn.ModuleList([
        #     ComplexSpikformerBlock(embed_dim, num_heads=num_heads, mlp_ratio=2.0) # mlp_ratio Houston2018 1 Augsburg 1, Trento Houston 2013 Berlin 2
        #     for _ in range(num_heads)
        # ])
        self.complex_encoder = nn.ModuleList([
            ComplexSpikformerBlock(
                embed_dim,
                num_heads=num_heads,
                mlp_ratio=1.0
            )
            for _ in range(depth)
        ])

        # 分类头：取复数的模长 (Amplitude) 或将实部虚部拼接进行降维
        hidden_dim = embed_dim
        # hidden_dim = embed_dim
        self.cls_head = ComplexSpikingClassificationHead(embed_dim, hidden_dim, num_classes)

    def forward(self, hsi_patch, lidar_patch):
        # 提取 Patch 嵌入特征: [T, B, N, D]
        x_r = self.hsi_embed(hsi_patch)
        x_i = self.lidar_embed(lidar_patch)

        # 提取用于对比学习的全局特征/ Token 均值特征: [T, B, D]
        # (N 维度为空间 Token 数，平均池化后作为该样本在当前时间步的代表特征)
        CL_r = x_r.mean(dim=2)
        CL_i = x_i.mean(dim=2)

        # 经过投影头得到对比学习特征 z: [T, B, D_proj]
        z_hsi_seq = self.hsi_proj(CL_r)
        z_lidar_seq = self.lidar_proj(CL_i)

        # 经过 Complex Spikformer Block (自注意力机制)
        for block in self.complex_encoder:
            x_r, x_i = block(x_r, x_i)

        # 空间维度归约: [T, B, N, D] -> [T, B, D]
        x_r_out = x_r.mean(dim=2)
        x_i_out = x_i.mean(dim=2)

        # 分类头得到 Logits: [B, num_classes]
        logits = self.cls_head(x_r_out, x_i_out)

        if self.training:
            # 训练阶段同时返回分类预测和对比特征
            return logits, z_hsi_seq, z_lidar_seq
        else:
            # 推理阶段只返回 Logits
            return logits



