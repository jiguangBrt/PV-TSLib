import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class VariateEmbedding(nn.Module):
    """类似 iTransformer 的变量级嵌入 (Variate-wise Embedding)"""
    def __init__(self, num_variates, d_model):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)  # 每个变量独立嵌入
        self.num_variates = num_variates
        
    def forward(self, x):
        # x: [B, L, C] -> [B, L, C, d_model]
        B, L, C = x.shape
        x = x.unsqueeze(-1)  # [B, L, C, 1]
        x = self.embedding(x)  # [B, L, C, d_model]
        return x.reshape(B, L, C * self.num_variates)  # 暂时不展开,后续调整

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Channel Attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        
    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        # Squeeze: Global Average Pooling
        squeeze = x.mean(dim=1)  # [B, C]
        # Excitation: FC-ReLU-FC-Sigmoid
        excitation = self.fc2(F.relu(self.fc1(squeeze)))  # [B, C]
        excitation = torch.sigmoid(excitation).unsqueeze(1)  # [B, 1, C]
        # Scale
        return x * excitation

class DualDomainMamba(nn.Module):
    """双域 Mamba: 时域 + 频域并行"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.time_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.freq_mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        # 融合层: 时域 + 频域 -> d_model
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x):
        # x: [B, L, d_model]
        B, L, C = x.shape
        
        # 时域分支
        time_out = self.time_mamba(x)  # [B, L, d_model]
        
        # 频域分支: FFT -> Mamba -> iFFT
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')  # [B, L//2+1, d_model] (复数)
        x_freq_real = torch.cat([x_freq.real, x_freq.imag], dim=-1)  # [B, L//2+1, 2*d_model]
        
        # 调整维度以适配 Mamba (需要补齐到 L 长度)
        # 方法: 用零填充或插值
        freq_len = x_freq.shape[1]
        if freq_len < L:
            pad_len = L - freq_len
            x_freq_padded = F.pad(x_freq_real, (0, 0, 0, pad_len))  # [B, L, 2*d_model]
        else:
            x_freq_padded = x_freq_real[:, :L, :]
            
        # 投影到 d_model 维度
        x_freq_input = x_freq_padded[..., :C]  # 简化: 只取实部 [B, L, d_model]
        freq_out = self.freq_mamba(x_freq_input)  # [B, L, d_model]
        
        # 拼接 + 融合
        dual_out = torch.cat([time_out, freq_out], dim=-1)  # [B, L, 2*d_model]
        fused = self.fusion(dual_out)  # [B, L, d_model]
        
        return fused

class Model(nn.Module):
    """
    PhysicsMamba: 物理去趋势 + MVMD + 双域 Mamba + SE Attention
    
    关键假设:
    - 输入数据已经过 MVMD 分解 (所有 IMF 已拼接在 enc_in 中)
    - 模型预测 PowerRes (去趋势功率), 外部加回 P_PHY 得到 OT
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.enc_in = configs.enc_in  # 输入 IMF 总维度 (48 或 56)
        self.c_out = configs.c_out    # 输出维度 (预测 PowerRes IMFs)
        
        # ⚠️ 修改这里: 使用 getattr 设置默认值,不依赖命令行参数
        d_state = getattr(configs, 'd_state', 16)      # 默认 16
        d_conv = getattr(configs, 'd_conv', 4)         # 默认 4
        expand = getattr(configs, 'expand', 2)         # 默认 2
        
        # 1. 输入投影: 所有 IMF -> d_model
        self.input_projection = nn.Linear(self.enc_in, self.d_model)
        
        # 2. 双域 Mamba 块 (可堆叠多层)
        self.num_layers = getattr(configs, 'e_layers', 2)
        self.dual_mamba_layers = nn.ModuleList([
            DualDomainMamba(
                d_model=self.d_model,
                d_state=d_state,    # 使用局部变量
                d_conv=d_conv,      # 使用局部变量
                expand=expand,      # 使用局部变量
            ) for _ in range(self.num_layers)
        ])
        
        # 3. SE Channel Attention
        self.se_block = SEBlock(self.d_model, reduction=16)
        
        # 4. 输出投影: d_model -> c_out (预测 PowerRes IMFs)
        self.output_projection = nn.Linear(self.d_model, self.c_out)
        
        # 5. 预测头: seq_len -> pred_len
        self.predict_head = nn.Linear(self.seq_len, self.pred_len)
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, seq_len, enc_in] - 所有 IMF 拼接后的输入
        B, L, _ = x_enc.shape
        
        # Step 1: 输入投影
        x = self.input_projection(x_enc)  # [B, seq_len, d_model]
        
        # Step 2: 多层双域 Mamba
        for mamba_layer in self.dual_mamba_layers:
            residual = x
            x = mamba_layer(x)  # [B, seq_len, d_model]
            x = x + residual  # 残差连接
            x = F.layer_norm(x, (self.d_model,))  # LayerNorm
        
        # Step 3: SE Channel Attention
        x = self.se_block(x)  # [B, seq_len, d_model]
        
        # Step 4: 时间维度投影 seq_len -> pred_len
        x = x.transpose(1, 2)  # [B, d_model, seq_len]
        x = self.predict_head(x)  # [B, d_model, pred_len]
        x = x.transpose(1, 2)  # [B, pred_len, d_model]
        
        # Step 5: 输出投影 d_model -> c_out (PowerRes IMFs)
        output = self.output_projection(x)  # [B, pred_len, c_out]
        
        return output  # [B, pred_len, c_out]