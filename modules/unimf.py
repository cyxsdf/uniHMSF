import torch
from torch import nn
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.masked_multihead_attention import MultiheadAttention
import math


class MultiScaleCNN(nn.Module):
    """多尺度卷积模块，提取不同尺度的局部特征"""

    def __init__(self, in_channels, embed_dim, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.kernel_sizes = kernel_sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, embed_dim, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.proj = nn.Linear(len(kernel_sizes) * embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: (batch, seq_len, in_channels) -> (batch, in_channels, seq_len)
        x = x.transpose(1, 2)
        # 多尺度特征提取
        features = []
        for conv in self.convs:
            feat = conv(x)  # (batch, embed_dim, seq_len)
            feat = F.relu(feat)
            features.append(feat.transpose(1, 2))  # (batch, seq_len, embed_dim)

        # 特征融合
        combined = torch.cat(features, dim=-1)  # (batch, seq_len, len(kernels)*embed_dim)
        combined = self.proj(combined)  # (batch, seq_len, embed_dim)
        return self.dropout(combined)


class CrossScaleAttention(nn.Module):
    """跨尺度注意力机制，增强不同尺度特征的信息交互"""

    def __init__(self, embed_dim, num_heads, scale_count):
        super().__init__()
        self.scale_count = scale_count
        self.attns = nn.ModuleList([
            MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                lens=(0, 0, 0),  # 临时占位，实际使用时会调整
                modalities='all'
            ) for _ in range(scale_count)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(scale_count)])

    def forward(self, scales):
        # scales: list of (seq_len, batch, embed_dim)
        updated_scales = []
        for i in range(self.scale_count):
            query = scales[i]
            # 融合其他尺度的信息
            combined = []
            for j in range(self.scale_count):
                if i != j:
                    key = value = scales[j]
                    attn_out, _ = self.attns[i](query, key, value)
                    combined.append(attn_out)

            # 与自身残差连接
            if combined:
                combined = torch.stack(combined).mean(dim=0)
                updated = self.norms[i](query + combined)
            else:
                updated = query
            updated_scales.append(updated)
        return updated_scales


class HierarchicalTransformer(nn.Module):
    """层次化Transformer，对不同尺度特征进行融合"""

    def __init__(self, embed_dim, num_heads, layers, scale_count):
        super().__init__()
        self.scale_count = scale_count
        # 每个尺度的Transformer层
        self.scale_layers = nn.ModuleList([
            nn.ModuleList([
                MultimodalTransformerEncoderLayer(
                    embed_dim=embed_dim,
                    lens=(0, 0, 0),  # 临时占位
                    modalities='single',
                    num_heads=num_heads
                ) for _ in range(layers)
            ]) for _ in range(scale_count)
        ])
        # 跨尺度注意力
        self.cross_scale_attn = CrossScaleAttention(embed_dim, num_heads, scale_count)
        # 最终融合层
        self.fusion_proj = nn.Linear(scale_count * embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, scales):
        # 先在每个尺度内部进行自注意力
        for i in range(self.scale_count):
            x = scales[i]
            for layer in self.scale_layers[i]:
                x = layer(x)
            scales[i] = x

        # 跨尺度注意力融合
        scales = self.cross_scale_attn(scales)

        # 最终特征融合
        batch_size = scales[0].size(1)
        seq_len = scales[0].size(0)

        # 调整维度并拼接所有尺度特征
        fused = torch.cat([s.unsqueeze(0) for s in scales], dim=0)  # (scale, seq, batch, dim)
        fused = fused.permute(2, 1, 0, 3)  # (batch, seq, scale, dim)
        fused = fused.reshape(batch_size, seq_len, -1)  # (batch, seq, scale*dim)

        # 投影到目标维度
        fused = self.fusion_proj(fused)  # (batch, seq, dim)
        fused = self.layer_norm(fused)
        return fused.transpose(0, 1)  # (seq, batch, dim)


class MultimodalTransformerEncoder(nn.Module):
    """修改后的多模态Transformer编码器，集成层次化多尺度建模"""

    def __init__(self, embed_dim, num_heads, layers, lens, modalities, attn_dropout=0.0, relu_dropout=0.0,
                 res_dropout=0.0, embed_dropout=0.0, attn_mask=False, embed_positions=None,
                 scale_count=3):  # 新增尺度数量参数
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.scale_count = scale_count

        if embed_positions is not None:
            self.embed_scale = math.sqrt(embed_dim)
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        else:
            self.embed_scale = 1
            self.embed_positions = embed_positions

        self.attn_mask = attn_mask

        # 替换GRU为层次化Transformer
        self.hierarchical_transformer = HierarchicalTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            layers=layers,
            scale_count=scale_count
        )

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k=None, x_in_v=None):
        """
        Args:
            x_in (FloatTensor): 包含多尺度特征的列表，每个元素形状为 `(src_len, batch, embed_dim)`
        """
        # 如果输入不是多尺度特征列表，则创建单尺度特征
        if not isinstance(x_in, list):
            x_in = [x_in] * self.scale_count

        # 对每个尺度添加位置嵌入
        scaled_features = []
        for x in x_in:
            scaled_x = self.embed_scale * x
            if self.embed_positions is not None:
                scaled_x += self.embed_positions(x.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            scaled_features.append(scaled_x)

        # 层次化Transformer处理
        x = self.hierarchical_transformer(scaled_features)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class MultimodalTransformerEncoderLayer(nn.Module):
    """保持原有编码器层结构，用于层次化Transformer内部"""

    def __init__(self, embed_dim, lens, modalities, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            lens=lens,
            modalities=modalities,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = Linear(self.embed_dim, 4 * self.embed_dim)
        self.fc2 = Linear(4 * self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)


def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1 + abs(dim2 - dim1))
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


if __name__ == '__main__':
    encoder = MultimodalTransformerEncoder(300, 4, 2, [], 'all', scale_count=3)
    x1 = torch.rand(20, 2, 300)
    x2 = torch.rand(15, 2, 300)
    x3 = torch.rand(10, 2, 300)
    print(encoder([x1, x2, x3]).shape)