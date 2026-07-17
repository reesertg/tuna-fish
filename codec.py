"""
fish-speech DAC (Descript Audio Codec) 神经网络音频编解码器

该模块实现了一个端到端的音频压缩系统，主要包含：
1. Encoder: 将原始音频波形编码为低维潜在表示
2. Quantizer: 对潜在表示进行残差向量量化 (RVQ)
3. Decoder: 从量化后的潜在表示重建音频波形

特点：
- 支持因果卷积，适用于流式处理
- 集成 Transformer 模块增强序列建模能力
- 使用 Snake 激活函数提升音频生成质量
- 支持语义和声学的分层量化

主要改动：
1. 引用优化：去掉除 PyTorch 和标准库之外的引用
2. 结构重组：按功能模块分组，添加清晰的分隔注释
3. 删除重复：移除重复定义的函数
4. 统一格式：标准化缩进和空行
5. 添加注释：为每个类和关键函数添加中文文档字符串

模块结构：
├── 基础工具函数
├── 卷积层封装（因果卷积、权重归一化）
├── 激活函数（Snake）
├── ConvNeXt 模块
├── 向量量化模块（VQ、RVQ、下采样RVQ）
├── Transformer 组件
├── 编码器/解码器组件
├── 模型基类（保存/加载）
├── DAC 主模型
├── 模型实例化函数
└── 模型推理样例

"""

import math
import typing as tp
from dataclasses import dataclass
from typing import List, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations


# ==============================================================================
#                              基础工具函数
# ==============================================================================

def find_multiple(n: int, k: int) -> int:
    """找到大于等于 n 且是 k 的倍数的最小整数"""
    if n % k == 0:
        return n
    return n + k - (n % k)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """移除 1D 张量的填充"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left:end]


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """计算卷积需要的额外填充，确保输出长度正确"""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "zeros",
    value: float = 0.0,
):
    """
    1D 填充函数，支持 reflect 模式处理短序列
    对于 reflect 模式，当序列过短时先补零再反射
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0
    
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def init_weights(m):
    """卷积层权重初始化"""
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ==============================================================================
#                              卷积层封装
# ==============================================================================

def WNConv1d(*args, **kwargs):
    """带权重归一化的 1D 卷积层"""
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    """带权重归一化的 1D 转置卷积层"""
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class CausalConvNet(nn.Module):
    """
    因果卷积网络
    
    通过左填充实现因果性：当前时刻只能看到过去的信息
    适用于流式处理和自回归生成
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        stride=1,
        groups=1,
        padding=None,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1  # 有效卷积核大小
        self.dilation = dilation
        self.padding = self.kernel_size - self.stride  # 因果填充量

    def forward(self, x):
        pad = self.padding
        extra_padding = get_extra_padding_for_conv1d(
            x, self.kernel_size, self.stride, pad
        )
        x = pad1d(x, (pad, extra_padding), mode="constant", value=0)
        return self.conv(x).contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


class CausalTransConvNet(nn.Module):
    """
    因果转置卷积网络（上采样）
    通过移除右侧填充保持因果性
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, dilation=1, stride=1, padding=None
    ):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.stride = stride
        self.kernel_size = kernel_size

    def forward(self, x):
        x = self.conv(x)
        pad = self.kernel_size - self.stride
        padding_right = math.ceil(pad)
        padding_left = pad - padding_right
        x = unpad1d(x, (padding_left, padding_right))
        return x.contiguous()

    def weight_norm(self, name="weight", dim=0):
        self.conv = weight_norm(self.conv, name=name, dim=dim)
        return self

    def remove_weight_norm(self):
        self.conv = remove_parametrizations(self.conv)
        return self


def CausalWNConv1d(*args, **kwargs):
    """因果卷积 + 权重归一化"""
    return CausalConvNet(*args, **kwargs).weight_norm()


def CausalWNConvTranspose1d(*args, **kwargs):
    """因果转置卷积 + 权重归一化"""
    return CausalTransConvNet(*args, **kwargs).weight_norm()


# ==============================================================================
#                              激活函数
# ==============================================================================

@torch.compile
def snake(x, alpha):
    """
    Snake 激活函数（JIT 编译加速）
    
    公式: x + (1/alpha) * sin(alpha * x)^2
    特别适合音频生成任务，能够捕获周期性特征
    """
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    """Snake 激活函数模块，每个通道有独立的可学习频率参数"""
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


# ==============================================================================
#                              ConvNeXt 模块
# ==============================================================================

class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt 块
    
    结构: DwConv -> LayerNorm -> Linear -> GELU -> Linear -> Scale -> Residual
    用于增强特征提取能力
    """
    def __init__(
        self,
        dim: int,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()
        self.dwconv = CausalConvNet(
            dim, dim, kernel_size=kernel_size, groups=dim, dilation=dilation,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, int(mlp_ratio * dim))
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, apply_residual: bool = True):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        if apply_residual:
            x = input + x
        return x


# ==============================================================================
#                              向量量化模块
# ==============================================================================

@dataclass
class VQResult:
    """向量量化结果数据类"""
    z: torch.Tensor              # 量化后的连续表示
    codes: torch.Tensor          # 码本索引
    latents: torch.Tensor        # 量化前的潜在表示
    codebook_loss: torch.Tensor  # 码本损失
    commitment_loss: torch.Tensor  # 承诺损失
    semantic_distill_z: torch.Tensor | None = None


class VectorQuantize(nn.Module):
    """
    向量量化模块
    
    特性：
    1. Factorized codes: 在低维空间进行最近邻查找，提高码本利用率
    2. L2-normalized codes: 使用余弦相似度代替欧氏距离，提高训练稳定性
    """
    def __init__(self, input_dim: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.in_proj = WNConv1d(input_dim, codebook_dim, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, input_dim, kernel_size=1)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        """
        量化输入张量
        
        Args:
            z: 输入张量 [B, D, T]
            
        Returns:
            z_q: 量化后的表示
            commitment_loss: 承诺损失
            codebook_loss: 码本损失
            indices: 码本索引
            z_e: 投影后的潜在表示
        """
        z_e = self.in_proj(z)
        z_q, indices = self.decode_latents(z_e)

        commitment_loss = F.mse_loss(z_e, z_q.detach(), reduction="none").mean([1, 2])
        codebook_loss = F.mse_loss(z_q, z_e.detach(), reduction="none").mean([1, 2])

        # 直通估计器：前向传播用量化值，反向传播绕过量化
        z_q = z_e + (z_q - z_e).detach()
        z_q = self.out_proj(z_q)

        return z_q, commitment_loss, codebook_loss, indices, z_e

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id):
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents):
        """将潜在表示解码为量化表示"""
        encodings = latents.transpose(1, 2).reshape(-1, latents.shape[1])
        codebook = self.codebook.weight

        # L2 归一化（余弦相似度）
        encodings = F.normalize(encodings)
        codebook = F.normalize(codebook)

        # 计算欧氏距离
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1].reshape(latents.shape[0], -1)
        z_q = self.decode_code(indices)
        return z_q, indices


class ResidualVectorQuantize(nn.Module):
    """
    残差向量量化 (RVQ)
    
    多个量化器级联，每个量化器处理前一个的残差
    来源：SoundStream (https://arxiv.org/abs/2107.03312)
    """
    def __init__(
        self,
        input_dim: int = 512,
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        codebook_dim: Union[int, list] = 8,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        if isinstance(codebook_dim, int):
            codebook_dim = [codebook_dim for _ in range(n_codebooks)]

        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebook_size = codebook_size
        self.quantizer_dropout = quantizer_dropout

        self.quantizers = nn.ModuleList([
            VectorQuantize(input_dim, codebook_size, codebook_dim[i])
            for i in range(n_codebooks)
        ])

    def forward(self, z, n_quantizers: int = None):
        """
        对输入进行残差向量量化
        
        Args:
            z: 输入张量 [B, D, T]
            n_quantizers: 使用的量化器数量
            
        Returns:
            z_q, codes, latents, commitment_loss, codebook_loss
        """
        z_q = 0
        residual = z
        commitment_loss = 0
        codebook_loss = 0
        codebook_indices = []
        latents = []

        if n_quantizers is None:
            n_quantizers = self.n_codebooks
            
        # 训练时随机丢弃部分量化器
        if self.training:
            n_quantizers = torch.ones((z.shape[0],)) * self.n_codebooks + 1
            dropout = torch.randint(1, self.n_codebooks + 1, (z.shape[0],))
            n_dropout = int(z.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(z.device)

        for i, quantizer in enumerate(self.quantizers):
            if not self.training and i >= n_quantizers:
                break

            z_q_i, commitment_loss_i, codebook_loss_i, indices_i, z_e_i = quantizer(residual)

            mask = torch.full((z.shape[0],), fill_value=i, device=z.device) < n_quantizers
            z_q = z_q + z_q_i * mask[:, None, None]
            residual = residual - z_q_i

            commitment_loss += (commitment_loss_i * mask).mean()
            codebook_loss += (codebook_loss_i * mask).mean()
            codebook_indices.append(indices_i)
            latents.append(z_e_i)

        codes = torch.stack(codebook_indices, dim=1)
        latents = torch.cat(latents, dim=1)
        return z_q, codes, latents, commitment_loss, codebook_loss

    def from_codes(self, codes: torch.Tensor):
        """从码本索引重建连续表示"""
        z_q = 0.0
        z_p = []
        n_codebooks = codes.shape[1]
        for i in range(n_codebooks):
            z_p_i = self.quantizers[i].decode_code(codes[:, i, :])
            z_p.append(z_p_i)
            z_q_i = self.quantizers[i].out_proj(z_p_i)
            z_q = z_q + z_q_i
        return z_q, torch.cat(z_p, dim=1), codes


class DownsampleResidualVectorQuantize(nn.Module):
    """
    下采样残差向量量化
    
    在量化前进行下采样，减少序列长度，提高压缩率
    支持语义和声学的分层量化
    """
    def __init__(
        self,
        input_dim: int = 1024,
        n_codebooks: int = 9,
        codebook_dim: int = 8,
        quantizer_dropout: float = 0.5,
        codebook_size: int = 1024,
        semantic_codebook_size: int = 4096,
        downsample_factor: tuple[int] = (2, 2),
        downsample_dims: tuple[int] | None = None,
        pre_module: nn.Module | None = None,
        post_module: nn.Module | None = None,
        semantic_predictor_module: nn.Module | None = None,
    ):
        super().__init__()

        if downsample_dims is None:
            downsample_dims = [input_dim for _ in range(len(downsample_factor))]

        all_dims = (input_dim,) + tuple(downsample_dims)

        # 语义量化器（第一层，捕获高层语义信息）
        self.semantic_quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=1,
            codebook_size=semantic_codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=0.0,
        )

        # 声学量化器（后续层，捕获细节信息）
        self.quantizer = ResidualVectorQuantize(
            input_dim=input_dim,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
        )

        self.downsample_factor = downsample_factor
        self.downsample_dims = downsample_dims

        # 下采样模块
        self.downsample = nn.Sequential(*[
            nn.Sequential(
                CausalConvNet(all_dims[idx], all_dims[idx + 1], kernel_size=factor, stride=factor),
                ConvNeXtBlock(dim=all_dims[idx + 1]),
            )
            for idx, factor in enumerate(downsample_factor)
        ])

        # 上采样模块
        self.upsample = nn.Sequential(*[
            nn.Sequential(
                CausalTransConvNet(all_dims[idx + 1], all_dims[idx], kernel_size=factor, stride=factor),
                ConvNeXtBlock(dim=all_dims[idx]),
            )
            for idx, factor in reversed(list(enumerate(downsample_factor)))
        ])

        self.apply(self._init_weights)
        self.pre_module = pre_module if pre_module is not None else nn.Identity()
        self.post_module = post_module if post_module is not None else nn.Identity()
        self.semantic_predictor_module = (
            semantic_predictor_module if semantic_predictor_module is not None else nn.Identity()
        )

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, z, n_quantizers: int = None, semantic_len: torch.Tensor = None, **kwargs):
        original_shape = z.shape
        if semantic_len is None:
            semantic_len = torch.LongTensor([z.shape[-1]])

        # 下采样
        z = self.downsample(z)
        z = self.pre_module(z)

        # 语义量化
        semantic_z, semantic_codes, semantic_latents, semantic_commitment_loss, semantic_codebook_loss = \
            self.semantic_quantizer(z)

        # 残差量化
        residual_z = z - semantic_z
        residual_z, codes, latents, commitment_loss, codebook_loss = \
            self.quantizer(residual_z, n_quantizers=n_quantizers)

        z = semantic_z + residual_z
        commitment_loss = commitment_loss + semantic_commitment_loss
        codebook_loss = codebook_loss + semantic_codebook_loss
        codes = torch.cat([semantic_codes, codes], dim=1)
        latents = torch.cat([semantic_latents, latents], dim=1)

        z = self.post_module(z)
        z = self.upsample(z)

        # 对齐到原始长度
        diff = original_shape[-1] - z.shape[-1]
        if diff > 0:
            z = F.pad(z, (abs(diff), 0))
        elif diff < 0:
            z = z[..., abs(diff):]

        return VQResult(
            z=z,
            codes=codes,
            latents=latents,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
        )

    def decode(self, indices: torch.Tensor):
        """从码本索引解码"""
        indices[:, 0] = torch.clamp(indices[:, 0], max=self.semantic_quantizer.codebook_size - 1)
        indices[:, 1:] = torch.clamp(indices[:, 1:], max=self.quantizer.codebook_size - 1)

        z_q_semantic = self.semantic_quantizer.from_codes(indices[:, :1])[0]
        z_q_residual = self.quantizer.from_codes(indices[:, 1:])[0]
        z_q = z_q_semantic + z_q_residual
        z_q = self.post_module(z_q)
        z_q = self.upsample(z_q)
        return z_q


# ==============================================================================
#                              Transformer 组件
# ==============================================================================

@dataclass
class ModelArgs:
    """Transformer 模型参数配置"""
    block_size: int = 8192           # 最大序列长度
    n_layer: int = 8                 # Transformer 层数
    n_head: int = 8                  # 注意力头数
    dim: int = 512                   # 模型维度
    intermediate_size: int = 1536    # FFN 中间层维度
    n_local_heads: int = -1          # KV 头数（GQA）
    head_dim: int = 64               # 每个头的维度
    rope_base: float = 10000         # RoPE 基数
    norm_eps: float = 1e-5           # LayerNorm epsilon
    dropout_rate: float = 0.1        # Dropout 概率
    attn_dropout_rate: float = 0.1   # 注意力 Dropout
    channels_first: bool = True      # 通道优先格式
    pos_embed_type: str = "rope"     # 位置编码类型
    max_relative_position: int = 128 # 相对位置最大距离
    window_size: int = 512           # 窗口注意力大小

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        assert self.pos_embed_type in ["rope", "conformer"]


class KVCache(nn.Module):
    """KV 缓存，用于自回归推理加速"""
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]
        self.k_cache[:, :, input_pos] = k_val
        self.v_cache[:, :, input_pos] = v_val
        return (
            self.k_cache[:, :, : input_pos.max() + 1, :],
            self.v_cache[:, :, : input_pos.max() + 1, :],
        )

    def clear_cache(self, prompt_len):
        self.k_cache[:, :, prompt_len:, :] = 0
        self.v_cache[:, :, prompt_len:, :] = 0


class RMSNorm(nn.Module):
    """RMS 归一化，比 LayerNorm 更高效"""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerScale(nn.Module):
    """层缩放，稳定深层网络训练"""
    def __init__(self, dim: int, init_values: Union[float, Tensor] = 1e-2, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class FeedForward(nn.Module):
    """SwiGLU 前馈网络"""
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    """预计算 RoPE 频率"""
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    """应用旋转位置编码"""
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack([
        xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
        xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], -1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class Attention(nn.Module):
    """多头注意力机制，支持 RoPE 和 GQA"""
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.head_dim * config.n_head, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.attn_dropout_rate = config.attn_dropout_rate
        self.pos_embed_type = config.pos_embed_type

        if self.pos_embed_type == "conformer":
            self.max_relative_position = config.max_relative_position
            num_pos_embeddings = 2 * config.max_relative_position + 1
            self.rel_pos_embeddings = nn.Parameter(torch.zeros(num_pos_embeddings, self.head_dim))
            nn.init.normal_(self.rel_pos_embeddings, mean=0.0, std=0.02)

    def _compute_conformer_pos_scores(self, q: Tensor, seqlen: int) -> Tensor:
        """计算 Conformer 风格相对位置分数"""
        positions = torch.arange(seqlen, device=q.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions = torch.clamp(
            relative_positions + self.max_relative_position, 0, 2 * self.max_relative_position
        )
        rel_embeddings = self.rel_pos_embeddings[relative_positions]
        q = q.transpose(1, 2)
        rel_logits = torch.matmul(q, rel_embeddings.transpose(-2, -1))
        return rel_logits.transpose(1, 2)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape
        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([kv_size, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.pos_embed_type == "rope":
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)

        if self.pos_embed_type == "conformer":
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            rel_scores = self._compute_conformer_pos_scores(q, seqlen)
            scores = scores + rel_scores
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            if self.attn_dropout_rate > 0 and self.training:
                attn = F.dropout(attn, p=self.attn_dropout_rate)
            y = torch.matmul(attn, v)
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout_rate if self.training else 0.0,
                attn_mask=mask,
            )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.head_dim * self.n_head)
        return self.wo(y)


class TransformerBlock(nn.Module):
    """Transformer 块（Pre-Norm + LayerScale）"""
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.attention_layer_scale = LayerScale(config.dim, inplace=True)
        self.ffn_layer_scale = LayerScale(config.dim, inplace=True)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention_layer_scale(
            self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        )
        out = h + self.ffn_layer_scale(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    """标准 Transformer 编码器"""
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        if config.pos_embed_type == "rope": # encoder 86Hz, quantizer 21Hz
            freqs_cis = precompute_freqs_cis(32768, self.config.head_dim, self.config.rope_base)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        else:
            self.register_buffer("freqs_cis", None)

        # causal_mask = torch.tril(torch.ones(32768, 32768, dtype=torch.bool))
        causal_mask = None
        self.register_buffer("causal_mask", causal_mask, persistent=False)

        self.max_batch_size = -1
        self.max_seq_length = -1
        self.use_kv_cache = False

    def setup_caches(self, max_batch_size, max_seq_length):
        """设置 KV 缓存用于推理"""
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.norm.weight.dtype
        device = self.norm.weight.device

        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype
            ).to(device)
        self.use_kv_cache = True

    def forward(self, x: Tensor, input_pos: Optional[Tensor] = None, mask: Optional[Tensor] = None) -> Tensor:
        if self.config.pos_embed_type == "rope":
            assert self.freqs_cis is not None
            freqs_cis = self.freqs_cis[input_pos]
        else:
            freqs_cis = None

        if mask is None:
            if not self.training and self.use_kv_cache:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., : input_pos.max() + 1]
            else:
                mask = self.causal_mask[None, None, input_pos]
                mask = mask[..., input_pos]

        for layer in self.layers:
            x = layer(x, input_pos, freqs_cis, mask)
        return self.norm(x)


class WindowLimitedTransformer(Transformer):
    """窗口受限 Transformer，降低计算复杂度"""
    def __init__(
        self,
        config: ModelArgs,
        input_dim: int = 512,
        window_size: Optional[int] = None,
        causal: bool = True,
        look_ahead_conv: nn.Module = None,
    ):
        super().__init__(config)
        self.window_size = window_size
        self.causal = causal
        self.channels_first = config.channels_first
        self.look_ahead_conv = look_ahead_conv if look_ahead_conv is not None else nn.Identity()
        self.input_proj = nn.Linear(input_dim, config.dim) if input_dim != config.dim else nn.Identity()
        self.output_proj = nn.Linear(config.dim, input_dim) if input_dim != config.dim else nn.Identity()

    def make_window_limited_mask(self, max_length: int, x_lens: Optional[Tensor] = None) -> Tensor:
        """生成窗口受限的注意力掩码"""
        if self.causal:
            mask = torch.tril(torch.ones(max_length, max_length))
            row_indices = torch.arange(max_length).view(-1, 1)
            window_size = self.window_size or max_length
            valid_range = (row_indices - window_size + 1).clamp(min=0)
            column_indices = torch.arange(max_length)
            mask = (column_indices >= valid_range) & mask.bool()
        else:
            raise NotImplementedError
        return mask.bool()[None, None]

    def make_mask(self, max_length: int, x_lens: Optional[Tensor] = None) -> Tensor:
        if self.causal:
            mask = torch.tril(torch.ones(max_length, max_length))
        else:
            mask = torch.ones(max_length, max_length)
        return mask.bool()[None, None]

    def forward(self, x: Tensor, x_lens: Optional[Tensor] = None) -> Tensor:
        if self.channels_first:
            x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.look_ahead_conv(x)
        input_pos = torch.arange(x.shape[1], device=x.device)
        
        max_length = x.shape[1]
        if self.window_size is not None:
            mask = self.make_window_limited_mask(max_length, x_lens)
        else:
            mask = self.make_mask(max_length, x_lens)
        mask = mask.to(x.device)
        
        x = super().forward(x, input_pos, mask)
        x = self.output_proj(x)
        if self.channels_first:
            x = x.transpose(1, 2)
        return x


# ==============================================================================
#                              编码器/解码器组件
# ==============================================================================

class ResidualUnit(nn.Module):
    """残差单元，使用膨胀卷积增大感受野"""
    def __init__(self, dim: int = 16, dilation: int = 1, causal: bool = False):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            conv_class(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            conv_class(dim, dim, kernel_size=1),
        )
        self.causal = causal

    def forward(self, x):
        y = self.block(x)
        pad = x.shape[-1] - y.shape[-1]
        if pad > 0:
            if self.causal:
                x = x[..., :-pad]
            else:
                x = x[..., pad // 2: -pad // 2]
        return x + y


class EncoderBlock(nn.Module):
    """编码器块：残差单元 + 下采样 + 可选 Transformer"""
    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        
        transformer_module = nn.Identity() if n_t_layer == 0 else WindowLimitedTransformer(
            causal=causal,
            input_dim=dim,
            window_size=getattr(transformer_general_config, "window_size", 512),
            config=transformer_general_config(
                n_layer=n_t_layer, n_head=dim // 64, dim=dim, intermediate_size=dim * 3
            ),
        )
        
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1, causal=causal),
            ResidualUnit(dim // 2, dilation=3, causal=causal),
            ResidualUnit(dim // 2, dilation=9, causal=causal),
            Snake1d(dim // 2),
            conv_class(dim // 2, dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
            transformer_module,
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    """完整编码器：多级下采样压缩音频"""
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 4, 8, 8],
        d_latent: int = 64,
        n_transformer_layers: list = [0, 0, 4, 4],
        transformer_general_config: ModelArgs = None,
        causal: bool = False,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        self.block = [conv_class(1, d_model, kernel_size=7, padding=3)]

        for stride, n_t_layer in zip(strides, n_transformer_layers):
            d_model *= 2
            self.block += [EncoderBlock(
                d_model, stride=stride, causal=causal,
                n_t_layer=n_t_layer, transformer_general_config=transformer_general_config
            )]

        self.block += [Snake1d(d_model), conv_class(d_model, d_latent, kernel_size=3, padding=1)]
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """解码器块：上采样 + 残差单元"""
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        causal: bool = False,
        n_t_layer: int = 0,
        transformer_general_config=None,
    ):
        super().__init__()
        conv_trans_class = CausalWNConvTranspose1d if causal else WNConvTranspose1d
        
        self.block = nn.Sequential(
            Snake1d(input_dim),
            conv_trans_class(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2)),
            ResidualUnit(output_dim, dilation=1, causal=causal),
            ResidualUnit(output_dim, dilation=3, causal=causal),
            ResidualUnit(output_dim, dilation=9, causal=causal),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    """完整解码器：多级上采样重建音频"""
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        d_out: int = 1,
        causal: bool = False,
        n_transformer_layers: list = [0, 0, 0, 0],
        transformer_general_config=None,
    ):
        super().__init__()
        conv_class = CausalWNConv1d if causal else WNConv1d
        layers = [conv_class(input_channel, channels, kernel_size=7, padding=3)]

        for i, (stride, n_t_layer) in enumerate(zip(rates, n_transformer_layers)):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers += [DecoderBlock(
                input_dim, output_dim, stride, causal=causal,
                n_t_layer=n_t_layer, transformer_general_config=transformer_general_config
            )]

        layers += [Snake1d(output_dim), conv_class(output_dim, d_out, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ==============================================================================
#                              DAC 主模型
# ==============================================================================

class DAC(nn.Module):
    """
    DAC 神经网络音频编解码器
    
    Args:
        encoder_dim: 编码器初始通道数
        encoder_rates: 编码器下采样率列表
        latent_dim: 潜在空间维度
        decoder_dim: 解码器初始通道数
        decoder_rates: 解码器上采样率列表
        quantizer: 向量量化器模块
        sample_rate: 音频采样率
        causal: 是否使用因果卷积
    """
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        quantizer: torch.nn.Module = None,
        sample_rate: int = 44100,
        causal: bool = True,
        encoder_transformer_layers: List[int] = [0, 0, 0, 0],
        decoder_transformer_layers: List[int] = [0, 0, 0, 0],
        overwrite_decoder: torch.nn.Module = None,
        transformer_general_config=None,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = torch.tensor(encoder_rates).prod().item()

        self.encoder = Encoder(
            encoder_dim, encoder_rates, latent_dim, causal=causal,
            n_transformer_layers=encoder_transformer_layers,
            transformer_general_config=transformer_general_config,
        )

        self.quantizer = quantizer

        if overwrite_decoder is not None:
            self.decoder = overwrite_decoder
        else:
            self.decoder = Decoder(
                latent_dim, decoder_dim, decoder_rates, causal=causal,
                n_transformer_layers=decoder_transformer_layers,
                transformer_general_config=transformer_general_config,
            )

        self.apply(init_weights)
        self.delay = self.get_delay()
        self.frame_length = self.hop_length * 4

    def get_output_length(self, input_length):
        """计算输出长度"""
        L = input_length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d, k, s = layer.dilation[0], layer.kernel_size[0], layer.stride[0]
                if isinstance(layer, nn.Conv1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                else:
                    L = (L - 1) * s + d * (k - 1) + 1
                L = math.floor(L)
        return L

    def get_delay(self):
        """计算模型延迟"""
        l_out = self.get_output_length(0)
        L = l_out
        layers = [layer for layer in self.modules() if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d))]
        
        for layer in reversed(layers):
            d, k, s = layer.dilation[0], layer.kernel_size[0], layer.stride[0]
            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            else:
                L = (L - 1) * s + d * (k - 1) + 1
            L = math.ceil(L)
        return (L - l_out) // 2

    def preprocess(self, audio_data, sample_rate):
        """预处理：填充到 hop_length 的倍数"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return nn.functional.pad(audio_data, (0, right_pad))

    def encode(self, audio_data: torch.Tensor, audio_lengths: torch.Tensor = None, n_quantizers: int = None, **kwargs):
        """编码音频为量化码本索引"""
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.frame_length) * self.frame_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        if audio_lengths is None:
            audio_lengths = torch.LongTensor([length + right_pad]).to(audio_data.device)

        z = self.encoder(audio_data)
        vq_results = self.quantizer(z, n_quantizers, **kwargs)
        indices = vq_results.codes
        indices_lens = torch.ceil(audio_lengths / self.frame_length).long()
        return indices, indices_lens

    def from_indices(self, indices: torch.Tensor):
        """从码本索引重建音频"""
        z = self.quantizer.decode(indices)
        return self.decoder(z)

    def decode(self, z: torch.Tensor):
        """从潜在表示解码音频"""
        return self.decoder(z)

    def forward(self, audio_data: torch.Tensor, sample_rate: int = None, n_quantizers: int = None, **kwargs):
        """完整前向传播：编码 -> 量化 -> 解码"""
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)
        vq_results = self.encode(audio_data, n_quantizers=n_quantizers, **kwargs)
        z = vq_results[0] if isinstance(vq_results, tuple) else vq_results.z
        x = self.decode(z)
        return x[..., :length], vq_results


# ==============================================================================
#                              模型实例化
# ==============================================================================

def get_model():
    """创建 DAC 模型实例"""
    model = DAC(
        encoder_transformer_layers=[0, 0, 0, 4],
        decoder_transformer_layers=[4, 0, 0, 0],
        transformer_general_config=ModelArgs,
        quantizer=DownsampleResidualVectorQuantize(
            post_module=WindowLimitedTransformer(
                config=ModelArgs(block_size=2048, n_head=16, dim=1024, intermediate_size=3072),
                input_dim=1024, window_size=128
            ),
            pre_module=WindowLimitedTransformer(
                config=ModelArgs(block_size=2048, n_head=16, dim=1024, intermediate_size=3072),
                input_dim=1024, window_size=128
            )
        )
    )
    return model


# ==============================================================================
#                              模型推理
# ==============================================================================

if __name__=="__main__":
    import torch
    import pathlib,sys
    import tqdm,soundfile,soxr
    # from codec import get_model

    class Codec:
        def __init__(self,rank=0,codec_path="./s2-pro/codec.pth",DTYPE=torch.float16,compile=False):
            self.SAMPLE_RATE,self.DEVICE,self.DTYPE=44100,rank,DTYPE
            self.fishcodec=get_model().eval().to(dtype=self.DTYPE).to(device=self.DEVICE)
            self.fishcodec.load_state_dict(torch.load(codec_path,map_location="cpu"),strict=False)
            if compile: 
                self.fishcodec.encode=torch.compile(self.fishcodec.encode)
                self.fishcodec.decode=torch.compile(self.fishcodec.decode)

        def encode(self,wave_path=None,code_path=None,wave=None):
            if wave_path!=None:
                wave,sr=soundfile.read(wave_path,always_2d=True) # soundfile + soxr, remove torchaudio 
                if wave.shape[1]>1: wave=wave.mean(axis=1,keepdims=True)
                if sr!=self.SAMPLE_RATE: wave=soxr.resample(wave,sr,self.SAMPLE_RATE) 
                wave=torch.FloatTensor(wave).T.to(device=self.DEVICE,dtype=self.DTYPE)
            elif wave==None: return None

            with torch.inference_mode():
                indices,feature_lengths=self.fishcodec.encode(wave[None])
                code=indices[:1,:,:feature_lengths[0]]

            if code_path!=None: open(code_path,"wb").write(code.cpu().to(torch.int16).numpy().tobytes())
            else: return code

        def decode(self,code_path=None,wave_path=None,code=None):
            if code_path!=None: code=torch.frombuffer(bytearray(open(code_path,"rb").read()),dtype=torch.int16).view(1,10,-1).to(torch.int64).to(0)
            elif code==None: return None

            with torch.inference_mode():
                z=self.fishcodec.quantizer.decode(code)
                audio=self.fishcodec.decoder(z)[0,0].float()

            if wave_path!=None: soundfile.write(wave_path,(audio.cpu().clamp(-1.0,1.0)*32767)[:,None].to(torch.int16).numpy(),self.SAMPLE_RATE)
            else: return audio

    encodec=Codec(codec_path="./s2-pro/codec.pth")
    if len(sys.argv)>1 and sys.argv[1]=="encode":
        for wave_path in tqdm.tqdm(list([p for p in pathlib.Path("./examples/prompt").glob("*") if p.suffix[1:].upper() in soundfile._formats])):
            encodec.encode(wave_path,wave_path.with_suffix(".code"))
    else:
        for code_path in tqdm.tqdm(list(pathlib.Path("./examples/generate").glob("*.code"))):
            encodec.decode(code_path,code_path.with_suffix(".wav"))

