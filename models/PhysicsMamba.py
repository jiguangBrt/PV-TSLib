import torch
import torch.nn as nn
import torch.fft
from layers.Embed import DataEmbedding_inverted
from mamba_ssm import Mamba

class RLMambaBlock(nn.Module):
    """
    Residual Learning Mamba Block (Based on provided diagram)
    Structure:
    1. x -> Mamba -> y1
    2. Subtraction: res1 = x - y1
    3. Norm -> FF -> y2
    4. Subtraction: res2 = Norm(res1) - y2
    5. Gate/Activation -> Norm -> Output
    """
    def __init__(self, d_model, d_state, d_conv, expand):
        super(RLMambaBlock, self).__init__()
        self.d_model = d_model
        
        # Branch 1: Mamba
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        
        # Branch 2: Feed Forward
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Output Stage
        self.act = nn.SiLU() # Sigmoid/SiLU based on diagram sigma
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, D] or [B, V, D] depending on domain
        
        # --- Path 1: Mamba Learning ---
        mamba_out = self.mamba(x)
        
        # --- Subtraction 1: What Mamba missed ---
        res1 = x - mamba_out
        
        # --- Path 2: FF Learning ---
        res1_norm = self.norm1(res1)
        ff_out = self.ff(res1_norm)
        
        # --- Subtraction 2: What FF missed ---
        # Diagram logic: Input to FF minus Output of FF
        res2 = res1_norm - ff_out
        
        # --- Output Gate/Activation ---
        out = self.norm2(self.act(res2))
        
        # Note: Diagram shows a bottom connection (Input -> C -> Sigma).
        # Typically in RLMamba, this might be a highway connection. 
        # Here we return the refined residual representation.
        return out + x # Add original x for deep network stability (ResNet style)

class SpectralMambaBlock(nn.Module):
    """
    Freq-Domain Mamba with RLMamba Structure
    """
    def __init__(self, d_model, d_state, d_conv, expand):
        super(SpectralMambaBlock, self).__init__()
        # Use RLMamba structure but for Frequency Features
        # Since complex inputs are concatenated (real+imag), dim is doubled
        self.rl_mamba = RLMambaBlock(d_model * 2, d_state, d_conv, expand)
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # 1. FFT
        x_freq = torch.fft.rfft(x, dim=-1, norm='ortho')
        x_freq_concat = torch.cat([x_freq.real, x_freq.imag], dim=-1)
        
        # 2. RLMamba in Freq Domain
        x_freq_out = self.rl_mamba(x_freq_concat)
        
        # 3. Inverse
        n_freq = x_freq.shape[-1]
        x_out_real = x_freq_out[:, :, :n_freq]
        x_out_imag = x_freq_out[:, :, n_freq:]
        x_freq_complex = torch.complex(x_out_real, x_out_imag)
        x_out = torch.fft.irfft(x_freq_complex, n=x.shape[-1], dim=-1, norm='ortho')
        return x_out

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        
        # Inverted Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Dual-Domain Encoder Layers
        self.layers = nn.ModuleList()
        for i in range(configs.e_layers):
            self.layers.append(nn.ModuleDict({
                'time_rlmamba': RLMambaBlock(
                    configs.d_model, configs.d_ff, configs.d_conv, configs.expand
                ),
                'freq_rlmamba': SpectralMambaBlock(
                    configs.d_model, configs.d_ff, configs.d_conv, configs.expand
                ),
                'norm': nn.LayerNorm(configs.d_model)
            }))

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        
    def forecast(self, x_enc, x_mark_enc):
        # Normalize
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Embedding: [B, L, V] -> [B, V, D]
        x = self.enc_embedding(x_enc, x_mark_enc) 

        # Dual-Domain Processing
        for layer in self.layers:
            # Parallel Dual-Domain Branches
            out_time = layer['time_rlmamba'](x)
            out_freq = layer['freq_rlmamba'](x)
            
            # Fusion
            x = layer['norm'](out_time + out_freq)

        # Projection: [B, V, D] -> [B, V, Pred_Len]
        dec_out = self.projector(x)
        dec_out = dec_out.permute(0, 2, 1) # -> [B, Pred_Len, V]

        # De-Normalize
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc)