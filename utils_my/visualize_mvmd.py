import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.preprocessing import MinMaxScaler

# --- 📍 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 你的输入数据文件夹 (请修改为你实际跑出来的文件夹名，比如 solar_processed_mvmd_xian)
INPUT_DIR = os.path.join(BASE_DIR, '../dataset/solar_processed_mvmd_xian')
OUTPUT_ROOT = os.path.join(BASE_DIR, '../dataset/viz_results')

# --- 🎨 绘图参数 ---
K_MODES = 8  # 必须与你清洗时的 K_MODES 一致
SAMPLE_LEN = 400 # 只画前 N 个点，画太长了看不清波形细节，400点大概是4天的数据
DPI = 300   # 论文出版级清晰度

# 颜色盘 (便于区分不同变量)
COLORS = sns.color_palette("husl", 10) 

def plot_single_variable_decomposition(df, variable_name, imf_cols, save_dir):
    """
    Type A: 画出一个变量的 Origin 和它的 IMF1-K
    参考你提供的图片风格：左列 Origin+IMF1-4，右列 IMF5-8
    """
    # 准备数据 (只取前 SAMPLE_LEN 个点)
    data_slice = df.iloc[:SAMPLE_LEN]
    
    # 构建 Origin 数据 (如果是 PowerRes，它本身就是 Origin，如果是 IMF，需要把所有 IMF 加起来才是 Origin)
    # 你的脚本里已经没有原始的纯净变量列了，所以我们用 sum(IMFs) 近似重构 Origin 用于展示，或者去原来的 raw data 找
    # 这里为了方便，我们直接把所有 IMF 加起来作为 "Reconstructed Origin"
    origin_series = data_slice[imf_cols].sum(axis=1)
    
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(f"Decomposition of {variable_name}", fontsize=16, fontweight='bold')
    
    # 布局：左边 5 行 (Origin + IMF1-4)，右边 4 行 (IMF5-8) -> 其实用 5x2 的网格比较好
    # Row 0: Origin (span 2 cols or just left) -> Let's do 5 rows, 2 cols
    
    # Plot Origin
    ax_origin = plt.subplot2grid((5, 2), (0, 0), colspan=1)
    ax_origin.plot(data_slice.index, origin_series, color='darkred', linewidth=1.5)
    ax_origin.set_ylabel("Origin", fontweight='bold')
    ax_origin.grid(True, linestyle='--', alpha=0.5)
    ax_origin.set_title(f"Reconstructed Origin (Sum of IMFs)", fontsize=10)

    # Plot IMFs
    for k in range(K_MODES):
        imf_idx = k + 1
        col_name = f"{variable_name}_IMF{imf_idx}"
        
        # 决定放在左边还是右边
        if imf_idx <= 4:
            row = imf_idx # 1, 2, 3, 4
            col = 0
        else:
            row = imf_idx - 4 # 1, 2, 3, 4 (IMF5 is row 1)
            col = 1
            
        ax = plt.subplot2grid((5, 2), (row, col))
        
        # 挑选一种颜色
        color = COLORS[k % len(COLORS)]
        ax.plot(data_slice.index, data_slice[col_name], color=color, linewidth=1.2)
        ax.set_ylabel(f"IMF {imf_idx}", fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # 去掉 x 轴标签，除了最后一行
        if row != 4:
            ax.set_xticklabels([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    save_path = os.path.join(save_dir, f"{variable_name}_Decomposition.png")
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"    Saved: {save_path}")

def plot_multi_variable_overlay(df, variable_names, save_dir):
    """
    Type B: 同频率叠加图。
    将所有变量的 IMF_k 画在同一张子图里。
    为了能画在一起，必须先做 MinMax 归一化。
    """
    data_slice = df.iloc[:SAMPLE_LEN].copy()
    scaler = MinMaxScaler()
    
    # 获取所有相关的列名
    all_cols = []
    for v in variable_names:
        for k in range(1, K_MODES + 1):
            all_cols.append(f"{v}_IMF{k}")
            
    # 归一化数据 (仅为了绘图对比趋势，不改变原始数据)
    data_norm = pd.DataFrame(scaler.fit_transform(data_slice[all_cols]), columns=all_cols, index=data_slice.index)
    
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle(f"Multi-Variable Correlation by Frequency (Normalized)", fontsize=16, fontweight='bold')
    
    # 布局 4行2列 (对应 8 个 IMF)
    for k in range(K_MODES):
        imf_idx = k + 1
        ax = plt.subplot(4, 2, imf_idx)
        
        # 遍历所有变量，画出它们的第 k 个 IMF
        for i, var_name in enumerate(variable_names):
            col_name = f"{var_name}_IMF{imf_idx}"
            if col_name in data_norm.columns:
                ax.plot(data_norm.index, data_norm[col_name], label=var_name, color=COLORS[i], linewidth=1.0, alpha=0.8)
        
        ax.set_title(f"Component: IMF {imf_idx} (Aligned Frequency)", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 只在第一张图显示图例，避免遮挡
        if k == 0:
            ax.legend(loc='upper right', fontsize='small', framealpha=0.9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    save_path = os.path.join(save_dir, "All_Variables_Overlay.png")
    plt.savefig(save_path, dpi=DPI)
    plt.close()
    print(f"    Saved: {save_path}")

def process_visualization():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ Input directory not found: {INPUT_DIR}")
        return

    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    print(f"🔍 Found {len(csv_files)} datasets.")

    for file_path in csv_files:
        filename = os.path.basename(file_path)
        site_name = filename.replace('.csv', '')
        
        # 1. 创建单独的文件夹
        site_dir = os.path.join(OUTPUT_ROOT, site_name)
        if not os.path.exists(site_dir):
            os.makedirs(site_dir)
            
        print(f"\n🎨 Visualizing {site_name} ...")
        df = pd.read_csv(file_path)
        
        # 2. 识别数据集中有哪些变量被分解了
        # 逻辑：查找所有 _IMF1 结尾的列名，提取前缀
        decomposed_vars = []
        for col in df.columns:
            if col.endswith('_IMF1'):
                var_name = col.replace('_IMF1', '')
                decomposed_vars.append(var_name)
        
        print(f"    Variables found: {decomposed_vars}")
        
        # 3. 生成 Type A 图 (每个变量一张)
        for var in decomposed_vars:
            imf_cols = [f"{var}_IMF{k+1}" for k in range(K_MODES)]
            plot_single_variable_decomposition(df, var, imf_cols, site_dir)
            
        # 4. 生成 Type B 图 (所有变量叠加)
        plot_multi_variable_overlay(df, decomposed_vars, site_dir)

    print(f"\n✅ All visualizations saved to: {OUTPUT_ROOT}")

if __name__ == '__main__':
    process_visualization()