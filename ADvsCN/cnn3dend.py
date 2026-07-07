import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math


class h_sigmoid(nn.Module):
	def __init__(self, inplace=True):
		super(h_sigmoid, self).__init__()
		self.relu = nn.ReLU6(inplace=inplace)

	def forward(self, x):
		return self.relu(x + 3) / 6


class h_swish(nn.Module):
	def __init__(self, inplace=True):
		super(h_swish, self).__init__()
		self.sigmoid = h_sigmoid(inplace=inplace)

	def forward(self, x):
		return x * self.sigmoid(x)


class CoordAtt(nn.Module):
	def __init__(self, inp, oup, reduction=32):
		super(CoordAtt, self).__init__()
		self.pool_d = nn.AdaptiveAvgPool3d((None, None, 1))
		self.pool_h = nn.AdaptiveAvgPool3d((None, 1, None))
		self.pool_w = nn.AdaptiveAvgPool3d((1, None, None))

		mip = max(8, inp // reduction)

		self.conv1 = nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0)
		self.bn1 = nn.BatchNorm3d(mip)
		self.act = h_swish()

		self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
		self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)

	def forward(self, x):
		identity = x

		n, c, d, h, w = x.size()
		x_d = self.pool_d(x).squeeze(2)
		x_h = self.pool_h(x).squeeze(3)
		x_w = self.pool_w(x).permute(0, 1, 4, 3, 2).squeeze(2)

		# 确保所有张量在第4个维度上具有相同的大小
		assert x_d.size(3) == x_h.size(3) == x_w.size(3)

		y = torch.cat([x_d, x_h, x_w], dim=1)

		y = self.conv1(y)
		y = self.bn1(y)
		y = self.act(y)

		x_d, x_h, x_w = torch.split(y, [d, h, w], dim=2)

		a_h = self.conv_h(x_h).sigmoid()
		a_w = self.conv_w(x_w).sigmoid()

		out = identity * a_w * a_h

		return out


class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        self.head_dim = in_channels // heads
        self.heads = heads
        assert self.head_dim * heads == in_channels, "Incompatible number of heads and in_channels"

        self.values = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.keys = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.queries = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0,
                                 bias=False)
        self.fc_out = nn.Conv3d(self.heads * self.head_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.norm = nn.BatchNorm3d(in_channels)  # 归一化层
        self.activation = nn.ReLU()  # 激活函数

    def forward(self, x):
        N, C, D, H, W = x.size()
        residual = x  # 保存输入以用于残差连接
        # 添加空间位置向量嵌入的部分

        # Apply convolutions to values, keys, and queries
        values = self.values(x).view(N, self.heads, self.head_dim, D, H, W)
        keys = self.keys(x).view(N, self.heads, self.head_dim, D, H, W)
        queries = self.queries(x).view(N, self.heads, self.head_dim, D, H, W)

        # Permute dimensions for matrix multiplication
        values = values.permute(0, 1, 3, 4, 5, 2).contiguous()
        keys = keys.permute(0, 1, 3, 4, 5, 2).contiguous()
        queries = queries.permute(0, 1, 3, 4, 5, 2).contiguous()

        # Calculate attention scores
        energy = torch.einsum("nhdxyz,nhexyz->nhedxy", [queries, keys])
        attention = F.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.einsum("nhedxy,nhdxyz->nhexyz", [attention, values]).reshape(N, self.heads * self.head_dim, D, H, W)

        # Reshape and apply final convolution
        out = self.fc_out(out)
        out = self.norm(out)
        out = self.activation(out)
        out = out + residual

        return out


class DecoderSelfAttention(nn.Module):
    def __init__(self, in_channels, heads=4):
        super(DecoderSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.head_dim = in_channels // heads
        self.heads = heads
        assert self.head_dim * heads == in_channels, "Incompatible number of heads and in_channels"

        self.values = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.keys = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.queries = nn.Conv3d(in_channels, self.heads * self.head_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc_out = nn.Conv3d(self.heads * self.head_dim, in_channels, kernel_size=1, stride=1, padding=0)

        self.norm = nn.BatchNorm3d(in_channels)  # 归一化层
        self.activation = nn.ReLU()  # 激活函数

    def forward(self, x, encoder_out):
        N, C, D, H, W = x.size()
        residual = x  # 保存输入以用于残差连接
        # 添加空间位置向量嵌入的部分

        # Apply convolutions to values, keys, and queries
        values = self.values(encoder_out).view(N, self.heads, self.head_dim, D, H, W)
        keys = self.keys(encoder_out).view(N, self.heads, self.head_dim, D, H, W)
        queries = self.queries(x).view(N, self.heads, self.head_dim, D, H, W)

        # Permute dimensions for matrix multiplication
        values = values.permute(0, 1, 3, 4, 5, 2).contiguous()
        keys = keys.permute(0, 1, 3, 4, 5, 2).contiguous()
        queries = queries.permute(0, 1, 3, 4, 5, 2).contiguous()

        # Calculate attention scores
        energy = torch.einsum("nhdxyz,nhexyz->nhedxy", [queries, keys])
        attention = F.softmax(energy, dim=-1)

        # Apply attention to values
        out = torch.einsum("nhedxy,nhdxyz->nhexyz", [attention, values]).reshape(N, self.heads * self.head_dim, D, H, W)

        # Reshape and apply final convolution
        out = self.fc_out(out)
        out = self.norm(out)
        out = self.activation(out)
        out = out + residual

        return out


class SELayer(nn.Module):
	def __init__(self, channel, reduction=1):
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool3d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(channel // reduction, channel, bias=False),
			nn.Sigmoid(),
		)

	def forward(self, x):
		b, c, _, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1, 1)
		return x * y.expand_as(x)


class BasicConv(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
	             bn=True, bias=False):
		super(BasicConv, self).__init__()
		self.out_channels = out_planes
		self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
		                      dilation=dilation, groups=groups, bias=bias)
		self.bn = nn.BatchNorm3d(out_planes, momentum=0.01, affine=True) if bn else None
		self.relu = nn.ReLU() if relu else None

	def forward(self, x):
		x = self.conv(x)
		if self.bn is not None:
			x = self.bn(x)
		if self.relu is not None:
			x = self.relu(x)
		return x


class ChannelPool(nn.Module):
	def forward(self, x):
		return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
	def __init__(self):
		super(SpatialGate, self).__init__()
		kernel_size = 3
		self.compress = ChannelPool()
		self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

	def forward(self, x):
		x_compress = self.compress(x)
		x_out = self.spatial(x_compress)
		scale = torch.sigmoid(x_out)
		self.attention = scale
		return x * self.attention


class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x):
        if x.dim() == 4:
            # 2D positional encoding
            pe = torch.zeros(x.size(0), x.size(2), x.size(3), self.d_model)
            pe.requires_grad = False
            pos = torch.arange(0, x.size(2), dtype=torch.float).unsqueeze(0).unsqueeze(0)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
            pe[:, :, :, 0::2] = torch.sin(pos * div_term)
            pe[:, :, :, 1::2] = torch.cos(pos * div_term)
        elif x.dim() == 5:
            # 3D positional encoding
            pe = torch.zeros(x.size(0), x.size(2), x.size(3), x.size(4),self.d_model)
            pe.requires_grad = False
            #print(pe.shape)
            pos = torch.arange(0, self.d_model/2, dtype=torch.float).unsqueeze(0).unsqueeze(0)
            #print(pos.shape)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
            pe[:, :, :, :, 0::2] = torch.sin(pos * div_term)
            pe[:, :, :, :, 1::2] = torch.cos(pos * div_term)
        else:
            raise ValueError("Positional encoding input must have 4 or 5 dimensions")

        # 将 pe 移动到与 x 相同的设备上
        pe = pe.permute(0, 4, 1, 2, 3)
        pe = pe.to(x.device)
        return x + pe


class OrdinalEmbedding(nn.Module):
	def __init__(self, input_dim, embedding_dim):
		super(OrdinalEmbedding, self).__init__()
		self.fc = nn.Linear(input_dim, embedding_dim)

	def forward(self, x):
		return self.fc(x)


class MultiHeadAttention2D(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(MultiHeadAttention2D, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        # Linear transformations for Q, K, and V
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)

        # Linear transformation for output
        self.W_out = nn.Linear(input_dim, input_dim)

    def forward(self, q, k, v):
        batch_size = q.size(0)

        # Linear transformation
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # Splitting into multiple heads
        Q = Q.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate heads and apply final linear transformation
        attention_output = attention_output.transpose(0, 1).contiguous().view(batch_size, -1)
        output = self.W_out(attention_output)
        return output


class CognitiveAttentionModule(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(CognitiveAttentionModule, self).__init__()
        self.multihead_self_attention = MultiHeadAttention2D(input_dim, num_heads)
        self.multihead_cross_attention = MultiHeadAttention2D(input_dim, num_heads)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, input_q, input_k, input_v):
        # Self-attention
        self_attention_output = self.multihead_self_attention(input_q, input_q, input_q)

        # Cross-attention
        cross_attention_output = self.multihead_cross_attention(self_attention_output, input_k, input_v)

        # Layer normalization
        output = self.layer_norm(cross_attention_output + input_q)

        return output


class StandardViTBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,          # 嵌入维度（对应ViT的hidden_dim）
        num_heads: int = 8,      # 注意力头数，需满足 embed_dim % num_heads == 0
        mlp_ratio: float = 4.0,  # MLP升维比例（原版ViT用4倍）
        dropout: float = 0.0,    # 全局Dropout概率
        attn_dropout: float = 0.0# 注意力层Dropout概率
    ):
        super().__init__()
        # 1. 层归一化（Pre-LN架构，先LN再注意力/MLP）
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)  # ViT原版用eps=1e-6
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # 2. 多头自注意力（MSA）：对齐ViT原版参数
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,     # 输入格式[B, N, C]，适配视觉数据
            bias=True             # ViT原版用偏置
        )
        
        # 3. 前馈网络（MLP）：升维→GELU→Dropout→降维→Dropout
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),            # ViT原版用GELU（而非ReLU）
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # 4. Dropout（残差分支的Dropout，可选，原版ViT在训练时用）
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,        # 注意力掩码（如因果掩码，ViT一般不用）
        key_padding_mask: torch.Tensor = None  # Padding掩码（处理不等长patch序列）
    ) -> torch.Tensor:
        """
        标准ViT Block前向传播（Pre-LN架构）
        Args:
            x: 输入张量，维度 [B, N, C]（B=批次，N=patch数，C=嵌入维度）
            attn_mask: 注意力掩码，维度 [N, N] 或 [B*num_heads, N, N]
            key_padding_mask: Padding掩码，维度 [B, N]（True表示该位置是padding，需屏蔽）
        Returns:
            输出张量，维度 [B, N, C]（与输入维度一致）
        """
        # ========== 第一步：多头自注意力 + 残差连接 ==========
        # Pre-LN：先归一化，再做注意力
        x_norm1 = self.norm1(x)
        # 自注意力计算（q=k=v=x_norm1）
        attn_output, _ = self.attn(
            query=x_norm1,
            key=x_norm1,
            value=x_norm1,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False  # 无需返回注意力权重，提升效率
        )
        # 残差连接：原始输入 + Dropout(注意力输出)
        x = x + self.dropout(attn_output)
        
        # ========== 第二步：前馈网络 + 残差连接 ==========
        # Pre-LN：先归一化，再做MLP
        x_norm2 = self.norm2(x)
        # MLP计算
        mlp_output = self.mlp(x_norm2)
        # 残差连接：当前x + MLP输出
        x = x + mlp_output
        
        return x

class CrossAttentionFusionBlock(nn.Module):
    def __init__(self, dim=128, num_heads=4, mlp_ratio=2., dropout=0.0):
        super().__init__()
        # 1. Cross Attention 部分
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        
        # batch_first=True 让输入形状为 [Batch, Seq_len, Dim]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        self.dropout1 = nn.Dropout(dropout)

        # 2. Feed Forward Network (MLP) 部分
        self.norm_ffn = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(), # GELU通常比ReLU在Transformer中表现更好
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, cls_token, branches_list):
        """
        Args:
            cls_token: [Batch, 1, Dim] - 你的 CLS Token
            branches_list: list of [Batch, Dim] - 你的4个分支特征列表 [x1, x2, x3, x4]
        """
        # --- 1. 准备数据 ---
        # 将4个分支特征堆叠成序列: [Batch, 4, Dim]
        # stack dim=1 表示在时间步方向堆叠
        if isinstance(branches_list, list):
            kv_seq = torch.stack(branches_list, dim=1) 
        else:
            kv_seq = branches_list
        # --- 2. Cross Attention ---
        # Query 来自 CLS Token，Key/Value 来自 分支特征
        # Residual Connection: x = x + Attention(norm(x))
        
        q = self.norm_q(cls_token) # [B, 1, D]
        k = v = self.norm_kv(kv_seq) # [B, 4, D]
        
        # attn_output: [B, 1, D] (因为 Query 长度是 1)
        # attn_weights: [B, 1, 4] (可以看到模型更关注哪一个分支)
        attn_output, attn_weights = self.cross_attn(query=q, key=k, value=v)
        
        # 残差连接：将学到的信息加回 CLS Token
        x = cls_token + self.dropout1(attn_output)

        # --- 3. Feed Forward Network ---
        # Residual Connection: x = x + MLP(norm(x))
        x = x + self.mlp(self.norm_ffn(x))

        return x, attn_weights

class feature_Netin_tp4_multr_class(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		# CLS Token - 可学习的分类标记
		self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
		
		# Transformer Block - 用于特征交互
		# self.transformer_block = StandardViTBlock(embed_dim=128, num_heads=4, mlp_ratio=2, dropout=dropout, attn_dropout=dropout)
		self.fusion_block = CrossAttentionFusionBlock(dim=128, num_heads=4, mlp_ratio=2, dropout=dropout)
		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat2 = PositionalEncoding3D(128)
		self.ca2 = SelfAttention(128)
		self.de2 = DecoderSelfAttention(128)
		self.pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat3 = PositionalEncoding3D(128)
		self.ca3 = SelfAttention(128)
		self.de3 = DecoderSelfAttention(128)
		self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat4 = PositionalEncoding3D(128)
		self.ca4 = SelfAttention(128)
		self.de4 = DecoderSelfAttention(128)
		self.pool4 = nn.AdaptiveAvgPool3d((1, 1, 1))

	def forward(self, x):
		features_x = self.feature_extractor(x)
        
		features_x1 = self.locat1(features_x)  # 不需要位置向量的tr可以去掉这行
		map1 = self.ca1(features_x1)
		features_x1 = self.de1(features_x1, map1) * features_x1
		features_x1 = self.pool1(features_x1)
		features_x1 = features_x1.view(features_x1.shape[0], -1)

		features_x2 = self.locat2(features_x)  # 不需要位置向量的tr可以去掉这行
		map2 = self.ca2(features_x2)
		features_x2 = self.de2(features_x2, map2) * features_x2
		features_x2 = self.pool2(features_x2)
		features_x2 = features_x2.view(features_x2.shape[0], -1)

		features_x3 = self.locat3(features_x)  # 不需要位置向量的tr可以去掉这行
		map3 = self.ca3(features_x3)
		features_x3 = self.de3(features_x3, map3) * features_x3
		features_x3 = self.pool3(features_x3)
		features_x3 = features_x3.view(features_x3.shape[0], -1)

		features_x4 = self.locat4(features_x)  # 不需要位置向量的tr可以去掉这行
		map4 = self.ca4(features_x4)
		features_x4 = self.de4(features_x4, map4) * features_x4
		features_x4 = self.pool4(features_x4)
		features_x4 = features_x4.view(features_x4.shape[0], -1)

		# 扩展CLS Token到batch大小
		batch_size = features_x1.size(0)
		cls_token = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, 128]
		
		
		cls_feature, attn_weights = self.fusion_block(cls_token, [features_x1, features_x2, features_x3, features_x4])
		
		# 提取CLS Token（第一个位置）作为最终特征
		# cls_feature 目前是 [B, 1, 128]，需要去掉中间的维度输入分类器
		cls_feature = cls_feature.squeeze(1) # [B, 128]
		
		# 输入分类器
		logits1 = self.classifier1(cls_feature)

		return logits1, features_x, features_x1, features_x2, features_x3, features_x4

class feature_Netin_tp4_multr_class_concat_four_features(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128*4, 128*2),
            nn.LayerNorm(128*2),
			nn.ReLU(),
			nn.Linear(128*2, 2),
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat2 = PositionalEncoding3D(128)
		self.ca2 = SelfAttention(128)
		self.de2 = DecoderSelfAttention(128)
		self.pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat3 = PositionalEncoding3D(128)
		self.ca3 = SelfAttention(128)
		self.de3 = DecoderSelfAttention(128)
		self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat4 = PositionalEncoding3D(128)
		self.ca4 = SelfAttention(128)
		self.de4 = DecoderSelfAttention(128)
		self.pool4 = nn.AdaptiveAvgPool3d((1, 1, 1))

	def forward(self, x):
		features_x = self.feature_extractor(x)
        
		features_x1 = self.locat1(features_x)  # 不需要位置向量的tr可以去掉这行
		map1 = self.ca1(features_x1)
		features_x1 = self.de1(features_x1, map1) * features_x1
		features_x1 = self.pool1(features_x1)
		features_x1 = features_x1.view(features_x1.shape[0], -1)

		features_x2 = self.locat2(features_x)  # 不需要位置向量的tr可以去掉这行
		map2 = self.ca2(features_x2)
		features_x2 = self.de2(features_x2, map2) * features_x2
		features_x2 = self.pool2(features_x2)
		features_x2 = features_x2.view(features_x2.shape[0], -1)

		features_x3 = self.locat3(features_x)  # 不需要位置向量的tr可以去掉这行
		map3 = self.ca3(features_x3)
		features_x3 = self.de3(features_x3, map3) * features_x3
		features_x3 = self.pool3(features_x3)
		features_x3 = features_x3.view(features_x3.shape[0], -1)

		features_x4 = self.locat4(features_x)  # 不需要位置向量的tr可以去掉这行
		map4 = self.ca4(features_x4)
		features_x4 = self.de4(features_x4, map4) * features_x4
		features_x4 = self.pool4(features_x4)
		features_x4 = features_x4.view(features_x4.shape[0], -1)

		# 拼接特征
		cls_feature = torch.cat([features_x1, features_x2, features_x3, features_x4], dim=1)

		# 输入分类器
		logits1 = self.classifier1(cls_feature)

		return logits1, features_x, features_x1, features_x2, features_x3, features_x4

class feature_Netin_tp4_multr_score(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)
		self.classifier2 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)
		self.classifier3 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)

		self.classifier4 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat2 = PositionalEncoding3D(128)
		self.ca2 = SelfAttention(128)
		self.de2 = DecoderSelfAttention(128)
		self.pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat3 = PositionalEncoding3D(128)
		self.ca3 = SelfAttention(128)
		self.de3 = DecoderSelfAttention(128)
		self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat4 = PositionalEncoding3D(128)
		self.ca4 = SelfAttention(128)
		self.de4 = DecoderSelfAttention(128)
		self.pool4 = nn.AdaptiveAvgPool3d((1, 1, 1))

	def forward(self, x):
		features_x = self.feature_extractor(x)
        
		features_x1 = self.locat1(features_x)  # 不需要位置向量的tr可以去掉这行
		map1 = self.ca1(features_x1)
		features_x1 = self.de1(features_x1, map1) * features_x1
		features_x1 = self.pool1(features_x1)
		features_x1 = features_x1.view(features_x1.shape[0], -1)

		features_x2 = self.locat2(features_x)  # 不需要位置向量的tr可以去掉这行
		map2 = self.ca2(features_x2)
		features_x2 = self.de2(features_x2, map2) * features_x2
		features_x2 = self.pool2(features_x2)
		features_x2 = features_x2.view(features_x2.shape[0], -1)

		features_x3 = self.locat3(features_x)  # 不需要位置向量的tr可以去掉这行
		map3 = self.ca3(features_x3)
		features_x3 = self.de3(features_x3, map3) * features_x3
		features_x3 = self.pool3(features_x3)
		features_x3 = features_x3.view(features_x3.shape[0], -1)

		features_x4 = self.locat4(features_x)  # 不需要位置向量的tr可以去掉这行
		map4 = self.ca4(features_x4)
		features_x4 = self.de4(features_x4, map4) * features_x4
		features_x4 = self.pool4(features_x4)
		features_x4 = features_x4.view(features_x4.shape[0], -1)

		logits1 = self.classifier1(features_x1)
		logits2 = self.classifier2(features_x2)
		logits3 = self.classifier3(features_x3)
		logits4 = self.classifier4(features_x4)

		return logits1, logits2, logits3, logits4, features_x, features_x1, features_x2, features_x3, features_x4

class feature_Netin_tp4_multr_score_fuseToOne_distill_cls(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)
		self.classifier2 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)
		self.classifier3 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)

		self.classifier4 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat2 = PositionalEncoding3D(128)
		self.ca2 = SelfAttention(128)
		self.de2 = DecoderSelfAttention(128)
		self.pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat3 = PositionalEncoding3D(128)
		self.ca3 = SelfAttention(128)
		self.de3 = DecoderSelfAttention(128)
		self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat4 = PositionalEncoding3D(128)
		self.ca4 = SelfAttention(128)
		self.de4 = DecoderSelfAttention(128)
		self.pool4 = nn.AdaptiveAvgPool3d((1, 1, 1))
		
		# CLS Token - 可学习的分类标记
		self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
		
		# Transformer Block - 用于特征交互
		self.fusion_block = CrossAttentionFusionBlock(dim=128, num_heads=4, mlp_ratio=2, dropout=dropout)

	def forward(self, x):
		features_x = self.feature_extractor(x)
        
		features_x1 = self.locat1(features_x)  # 不需要位置向量的tr可以去掉这行
		map1 = self.ca1(features_x1)
		features_x1 = self.de1(features_x1, map1) * features_x1
		features_x1 = self.pool1(features_x1)
		features_x1 = features_x1.view(features_x1.shape[0], -1)

		features_x2 = self.locat2(features_x)  # 不需要位置向量的tr可以去掉这行
		map2 = self.ca2(features_x2)
		features_x2 = self.de2(features_x2, map2) * features_x2
		features_x2 = self.pool2(features_x2)
		features_x2 = features_x2.view(features_x2.shape[0], -1)

		features_x3 = self.locat3(features_x)  # 不需要位置向量的tr可以去掉这行
		map3 = self.ca3(features_x3)
		features_x3 = self.de3(features_x3, map3) * features_x3
		features_x3 = self.pool3(features_x3)
		features_x3 = features_x3.view(features_x3.shape[0], -1)

		features_x4 = self.locat4(features_x)  # 不需要位置向量的tr可以去掉这行
		map4 = self.ca4(features_x4)
		features_x4 = self.de4(features_x4, map4) * features_x4
		features_x4 = self.pool4(features_x4)
		features_x4 = features_x4.view(features_x4.shape[0], -1)

		logits1 = self.classifier1(features_x1)
		logits2 = self.classifier2(features_x2)
		logits3 = self.classifier3(features_x3)
		logits4 = self.classifier4(features_x4)
		# 将四个特征连同cls_token拼接起来，然后送入transformer_block,最后提取出cls_token的特征
		# 扩展CLS Token到batch大小
		batch_size = features_x1.size(0)
		cls_token = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, 128]

		cls_feature, attn_weights = self.fusion_block(cls_token, [features_x1, features_x2, features_x3, features_x4])
		cls_feature = cls_feature.squeeze(1)

		return logits1, logits2, logits3, logits4, cls_feature

class feature_Netin_tp4_multr_class_and_score(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		
		# Transformer Block - 用于特征交互
		# self.transformer_block = StandardViTBlock(embed_dim=128, num_heads=4, mlp_ratio=2, dropout=dropout, attn_dropout=dropout)
		self.fusion_block = CrossAttentionFusionBlock(dim=128, num_heads=4, mlp_ratio=2, dropout=dropout)
		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))

	def forward(self, x, scor_features:list):
		features_x = self.feature_extractor(x)
        
		features_x1 = self.locat1(features_x)  # 不需要位置向量的tr可以去掉这行
		map1 = self.ca1(features_x1)
		features_x1 = self.de1(features_x1, map1) * features_x1
		features_x1 = self.pool1(features_x1)
		features_x = features_x1.view(features_x1.shape[0], -1)
		features_x = features_x.unsqueeze(1)
		# print(f"features_x shape: {features_x.shape}")
		cls_feature, attn_weights = self.fusion_block(features_x, scor_features)
		
		# 提取CLS Token（第一个位置）作为最终特征
		# cls_feature 目前是 [B, 1, 128]，需要去掉中间的维度输入分类器
		cls_feature = cls_feature.squeeze(1) # [B, 128]
		
		# 输入分类器
		logits1 = self.classifier1(cls_feature)

		return logits1, features_x1

class feature_Netin_tp4_multr_class_score_attention_baseline(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))

	def forward(self, x):
		features_x = self.feature_extractor(x)
        
		features_x1 = self.locat1(features_x)
		map1 = self.ca1(features_x1)
		features_x1 = self.de1(features_x1, map1) * features_x1
		features_x1 = self.pool1(features_x1)
		features_x = features_x1.view(features_x1.shape[0], -1)
		
		# 输入分类器
		logits1 = self.classifier1(features_x)

		return logits1

class CNN3D(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)

	def forward(self, x):
		features_x = self.feature_extractor(x)
		features_x1 = self.pool1(features_x)
		features_x1 = features_x1.view(features_x1.shape[0], -1)
		logits1 = self.classifier1(features_x1)

		return logits1

class feature_Netin_tp4_multr_score_attention(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)

		self.score_predict = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(6*8*6, 128),
            nn.LayerNorm(128),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))


	def forward(self, x, scores):
		scores = scores.unsqueeze(dim=-1)
		features_x = self.feature_extractor(x) # torch.Size([2, 128, 6, 7, 6])

		predicted_score = self.score_predict(features_x.view(features_x.shape[0], features_x.shape[1], -1)) # torch.Size([2, 128])
		predicted_score = predicted_score.squeeze(dim=-1)
		score_attention_weight = torch.sigmoid(1 - torch.abs(predicted_score - scores)) # torch.Size([2, 128])
		
		weight_expanded = score_attention_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # 2. 逐通道加权（广播机制自动将权重应用到每个空间位置）
		features_weighted = features_x * weight_expanded  # [2, 128, 6, 7, 6]

		features_weighted = self.locat1(features_weighted)  # 不需要位置向量的tr可以去掉这行
		map1 = self.ca1(features_weighted)
		features_weighted = self.de1(features_weighted, map1) * features_weighted
		features_weighted = self.pool1(features_weighted)
		features_weighted = features_weighted.view(features_weighted.shape[0], -1)
		# 输入分类器
		logits1 = self.classifier1(features_weighted)

		return logits1, features_x, features_weighted, score_attention_weight

class feature_Netin_tp4_multr_score_attention_selfDisdill(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)

		self.score_predict = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(6*8*6, 128),
            nn.LayerNorm(128),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()
		)

		self.score_attention_predict = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(6*8*6, 128),
            nn.LayerNorm(128),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat2 = PositionalEncoding3D(128)
		self.ca2 = SelfAttention(128)
		self.pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))
				
		self.fc = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 256),
            nn.LayerNorm(256),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(256, 128),
            nn.LayerNorm(128),
			nn.ReLU(),
			nn.Sigmoid()
		)

	def forward(self, x, scores, a=0):
		scores = scores.unsqueeze(dim=-1)
		features_x = self.feature_extractor(x) # torch.Size([2, 128, 6, 7, 6])

		attention_predict = self.score_attention_predict(features_x.view(features_x.shape[0], features_x.shape[1], -1)) # torch.Size([2, 128])
		attention_predict = attention_predict.squeeze(dim=-1)

		predicted_score = self.score_predict(features_x.view(features_x.shape[0], features_x.shape[1], -1)) # torch.Size([2, 128])
		predicted_score = predicted_score.squeeze(dim=-1)
		score_attention_weight = 1 - torch.abs(predicted_score - scores) # torch.Size([2, 128])
		
		weight_expanded = score_attention_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # 2. 逐通道加权（广播机制自动将权重应用到每个空间位置）
		features_weighted = features_x * weight_expanded  # [2, 128, 6, 7, 6]

		features_weighted = self.locat1(features_weighted)  # 不需要位置向量的tr可以去掉这行
		map1 = self.ca1(features_weighted)
		features_weighted = self.de1(features_weighted, map1) * features_weighted
		features_weighted = self.pool1(features_weighted)
		features_weighted = features_weighted.view(features_weighted.shape[0], -1)
		# 输入分类器
		logits1 = self.classifier1(features_weighted)

		return logits1, attention_predict, score_attention_weight

class feature_Netin_tp4_multr_score_attention_fakeScores(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)

		self.score_predict = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(6*8*6, 128),
            nn.LayerNorm(128),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat4 = PositionalEncoding3D(128)
		self.ca4 = SelfAttention(128)
		self.de4 = DecoderSelfAttention(128)
		self.pool4 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.classifier4 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1)
		)

	def forward(self, x):

		features_x = self.feature_extractor(x) # torch.Size([2, 128, 6, 7, 6])

		features_x4 = self.locat4(features_x.detach)  # 不需要位置向量的tr可以去掉这行
		map4 = self.ca4(features_x4)
		features_x4 = self.de4(features_x4, map4) * features_x4
		features_x4 = self.pool4(features_x4)
		features_x4 = features_x4.view(features_x4.shape[0], -1)

		scores = self.classifier4(features_x4)

		# fake_scores = scores.unsqueeze(dim=-1)
		predicted_score = self.score_predict(features_x.view(features_x.shape[0], features_x.shape[1], -1)) # torch.Size([2, 128])
		predicted_score = predicted_score.squeeze(dim=-1)

		diff = predicted_score - scores.detach()  # [2, 128]
		abs_diff = torch.abs(diff)       # 取绝对值，只关注差值大小，不关注正负
		score_attention_weight = torch.exp(-abs_diff)

		# score_attention_weight = torch.sigmoid(1 - torch.abs(predicted_score - scores)) # torch.Size([2, 128])
		
		weight_expanded = score_attention_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # 2. 逐通道加权（广播机制自动将权重应用到每个空间位置）
		features_weighted = features_x * weight_expanded  # [2, 128, 6, 8, 6]

		features_weighted = self.locat1(features_weighted)  # 不需要位置向量的tr可以去掉这行
		map1 = self.ca1(features_weighted)
		features_weighted = self.de1(features_weighted, map1) * features_weighted
		features_weighted = self.pool1(features_weighted)
		features_weighted = features_weighted.view(features_weighted.shape[0], -1)
		# 输入分类器
		logits1 = self.classifier1(features_weighted)

		return logits1, scores

class feature_Netin_tp4_multr_score_attention_fakeScoresSuper(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)

		self.score_predict = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(6*8*6, 128),
            nn.LayerNorm(128),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat4 = PositionalEncoding3D(128)
		self.ca4 = SelfAttention(128)
		self.de4 = DecoderSelfAttention(128)
		self.pool4 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.classifier4 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1)
		)

	def forward(self, x, scores):
		scores = scores.unsqueeze(dim=-1)
		features_x = self.feature_extractor(x) # torch.Size([2, 128, 6, 7, 6])

		features_x4 = self.locat4(features_x)  # 不需要位置向量的tr可以去掉这行
		map4 = self.ca4(features_x4)
		features_x4 = self.de4(features_x4, map4) * features_x4
		features_x4 = self.pool4(features_x4)
		features_x4 = features_x4.view(features_x4.shape[0], -1)

		sP= self.classifier4(features_x4)

		predicted_score = self.score_predict(features_x.view(features_x.shape[0], features_x.shape[1], -1)) # torch.Size([2, 128])
		predicted_score = predicted_score.squeeze(dim=-1)
		score_attention_weight = torch.sigmoid(1 - torch.abs(predicted_score - scores)) # torch.Size([2, 128])
		
		weight_expanded = score_attention_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # 2. 逐通道加权（广播机制自动将权重应用到每个空间位置）
		features_weighted = features_x * weight_expanded  # [2, 128, 6, 7, 6]

		features_weighted = self.locat1(features_weighted)  # 不需要位置向量的tr可以去掉这行
		map1 = self.ca1(features_weighted)
		features_weighted = self.de1(features_weighted, map1) * features_weighted
		features_weighted = self.pool1(features_weighted)
		features_weighted = features_weighted.view(features_weighted.shape[0], -1)
		# 输入分类器
		logits1 = self.classifier1(features_weighted)

		return logits1, sP

class feature_Netin_tp4_multr_score_attention_vitblock(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)
		
		self.fc = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(288, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 1),
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.locat2 = PositionalEncoding3D(128)
		self.ca2 = SelfAttention(128)
		self.pool2 = nn.AdaptiveAvgPool3d((1, 1, 1))

		self.cls_token = nn.Parameter(torch.zeros(1, 1, 288))
		
		# Transformer Block - 用于特征交互
		# self.transformer_block = StandardViTBlock(embed_dim=128, num_heads=4, mlp_ratio=2, dropout=dropout, attn_dropout=dropout)
		self.fusion_block = CrossAttentionFusionBlock(dim=288, num_heads=4, mlp_ratio=2, dropout=dropout)

	def forward(self, x):
		features_x = self.feature_extractor(x) # torch.Size([2, 128, 6, 8, 6])

		features_x_position = self.locat2(features_x)
		score_f = self.ca2(features_x_position)
		score_f = score_f.view(score_f.shape[0], score_f.shape[1], -1)
		# 扩展CLS Token到batch大小
		batch_size = score_f.size(0)
		cls_token = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, 128]

		score_feature, attn_weights = self.fusion_block(cls_token, score_f)
		attn_weights = attn_weights.squeeze(dim=1)
		score_feature = score_feature.squeeze(dim=1)
		score = self.fc(score_feature)
		
		weight_expanded = attn_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # 2. 逐通道加权（广播机制自动将权重应用到每个空间位置）
		features_weighted = features_x * weight_expanded  # [2, 128, 6, 7, 6]

		features_weighted = self.locat1(features_weighted)  # 不需要位置向量的tr可以去掉这行
		map1 = self.ca1(features_weighted)
		features_weighted = self.de1(features_weighted, map1) * features_weighted
		features_weighted = self.pool1(features_weighted)
		features_weighted = features_weighted.view(features_weighted.shape[0], -1)
		# 输入分类器
		logits1 = self.classifier1(features_weighted)

		return logits1, score, attn_weights

class feature_Netin_tp4_multr_score_attention_fourScores(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(128, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)

		self.score_predict = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(6*8*6, 128),
            nn.LayerNorm(128),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(128, 64),
			nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 4),
			nn.Sigmoid()
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))


	def forward(self, x, scores):
		scores = scores.unsqueeze(dim=1)
		features_x = self.feature_extractor(x) 

		predicted_score = self.score_predict(features_x.view(features_x.shape[0], features_x.shape[1], -1)) 
		print(predicted_score.shape)
		score_attention_weight = torch.sigmoid(1 - torch.abs(predicted_score - scores).mean(dim=-1)) 
		
		weight_expanded = score_attention_weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # 2. 逐通道加权（广播机制自动将权重应用到每个空间位置）
		features_weighted = features_x * weight_expanded  

		features_weighted = self.locat1(features_weighted)  
		map1 = self.ca1(features_weighted)
		features_weighted = self.de1(features_weighted, map1) * features_weighted
		features_weighted = self.pool1(features_weighted)
		features_weighted = features_weighted.view(features_weighted.shape[0], -1)
		# 输入分类器
		logits1 = self.classifier1(features_weighted)

		return logits1, features_x, features_weighted, score_attention_weight


class feature_Netin_tp4_multr_class_and_ScoreFuseFourFeaturesToConcatWithClassFeatures(nn.Module):
	def __init__(self, dropout=0.0):
		nn.Module.__init__(self)
		self.feature_extractor = nn.Sequential(
			nn.Conv3d(1, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.Conv3d(16, 16, 3),
            nn.InstanceNorm3d(16, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(16, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.Conv3d(32, 32, 3),
            nn.InstanceNorm3d(32, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(32, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.Conv3d(64, 64, 3),
            nn.InstanceNorm3d(64, affine=True),
			nn.ReLU(),
			nn.MaxPool3d(2, stride=2),
			nn.Conv3d(64, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
			nn.Conv3d(128, 128, 3),
            nn.InstanceNorm3d(128, affine=True),
			nn.ReLU(),
            nn.Conv3d(128, 128, 1),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(),
        )

		
		# Transformer Block - 用于特征交互
		self.transformer_block = StandardViTBlock(embed_dim=128, num_heads=4, mlp_ratio=2, dropout=dropout, attn_dropout=dropout)
		# self.fusion_block = StandardViTBlock(dim=128, num_heads=4, mlp_ratio=2, dropout=dropout)
		self.classifier1 = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(256, 64),
            nn.LayerNorm(64),
			nn.ReLU(),
			nn.Linear(64, 2),
		)

		self.down = nn.Sequential(
			nn.Dropout(dropout),
			nn.Linear(256, 128),
            nn.LayerNorm(128),
			nn.ReLU(),
		)

		self.locat1 = PositionalEncoding3D(128)
		self.ca1 = SelfAttention(128)
		self.de1 = DecoderSelfAttention(128)
		self.pool1 = nn.AdaptiveAvgPool3d((1, 1, 1))

	def forward(self, x, scor_features:list):
		features_x = self.feature_extractor(x)
        
		features_x1 = self.locat1(features_x)  # 不需要位置向量的tr可以去掉这行
		map1 = self.ca1(features_x1)
		features_x1 = self.de1(features_x1, map1) * features_x1
		out1 = features_x1
		features_x1 = self.pool1(features_x1)
		features_x = features_x1.view(features_x1.shape[0], -1)
		
		expanded = [t.unsqueeze(1) for t in scor_features]
		score_feature = torch.cat(expanded, dim=1)
		score_fuse_feature = self.transformer_block(score_feature) # [B, 4, 128]
        # 对score_fuse_feature取最大池化和平均池化并拼接为[B, 256]
		max_pool = torch.max(score_fuse_feature, dim=1)[0]
		avg_pool = torch.mean(score_fuse_feature, dim=1)
		score_fuse_feature = torch.cat([max_pool, avg_pool], dim=1)
		score_fuse_feature = self.down(score_fuse_feature)
        # 拼接特征
		cls_feature = torch.cat([features_x, score_fuse_feature], dim=1)
        
		# 输入分类器
		logits1 = self.classifier1(cls_feature)

		return logits1, features_x1

# 如果直接运行此文件，则执行测试
if __name__ == "__main__":
    model = CNN3D(dropout=0.1)
    model.to('cuda:0')
    model.eval()  # 设置为评估模式
    
    # 生成模拟的3D医学图像输入 (batch_size=2, channels=1, depth=64, height=64, width=64)
    # 注意：输入尺寸需要足够大，以适应多次池化操作
    input_tensor = torch.randn(2, 1, 112, 128, 112)
    input_tensor = input_tensor.to('cuda:0')
    l = 0
    # 进行前向传播
    with torch.no_grad():  # 不计算梯度以提高效率
        l = model(input_tensor)
    print(l)
        




