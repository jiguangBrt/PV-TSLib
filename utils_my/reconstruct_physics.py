import pandas as pd
import numpy as np

def reconstruct_ot(pred_powerres, p_phy):
    """
    物理重构: OT = PowerRes_pred + P_PHY
    
    Args:
        pred_powerres: [B, pred_len, n_imfs] - 模型预测的 PowerRes IMFs
        p_phy: [B, pred_len] - 物理基线 (从测试集读取)
    
    Returns:
        ot_pred: [B, pred_len] - 重构后的光伏功率预测
    """
    # 步骤1: 将 IMFs 求和得到 PowerRes
    powerres_total = pred_powerres.sum(axis=-1)  # [B, pred_len]
    
    # 步骤2: 加回物理基线
    ot_pred = powerres_total + p_phy
    
    return ot_pred

# 示例用法:
# pred = model(test_data)  # [B, 24, 48]
# p_phy = test_df['P_PHY'].values[-24:]  # 取最后24小时的物理基线
# ot_final = reconstruct_ot(pred, p_phy)