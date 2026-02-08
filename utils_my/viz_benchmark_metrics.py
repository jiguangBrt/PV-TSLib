import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ================= 🔧 配置区域 =================
PROJECT_ROOT = "/root/autodl-tmp/Time-Series-Library/"
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset/viz_results/")

# 定义模型顺序
MODEL_ORDER = [
    'iTransformer', 
    'PatchTST', 
    'Mamba', 
    'Transformer', 
    'Informer', 
    'Autoformer'
]

# ⚠️ 关键设置：异常值阈值
# 任何 MSE > 2.0 的结果都被视为训练失败 (Diverged)，不画在图里
MSE_THRESHOLD = 2.0 

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ... (extract_metrics 函数保持不变，为了节省篇幅省略，直接复用上一段代码) ...
def extract_metrics():
    # --- 请直接复制上一段代码中的 extract_metrics 函数体 ---
    # 为方便你运行，我把提取逻辑简化在这里
    records = []
    exp_dirs = [d for d in os.listdir(RESULTS_ROOT) if os.path.isdir(os.path.join(RESULTS_ROOT, d))]
    print(f"🔍 Found {len(exp_dirs)} experiment records...")
    
    for folder_name in exp_dirs:
        try:
            # 简化解析逻辑
            if 'Short24' in folder_name:
                horizon = 'Short Term (24)'
                tag = 'Short24'
            elif 'Long96' in folder_name:
                horizon = 'Long Term (96)'
                tag = 'Long96'
            else:
                continue
            
            model_name = "Unknown"
            for m in MODEL_ORDER:
                if m in folder_name:
                    model_name = m
                    break
            if model_name == "Unknown": continue

            # 提取 Site
            start_marker = 'forecast_'
            end_marker = f'_{tag}'
            start_idx = folder_name.find(start_marker) + len(start_marker)
            end_idx = folder_name.find(end_marker)
            if start_idx == -1 or end_idx == -1: continue
            site_name = folder_name[start_idx:end_idx]
            
            # 读取
            metric_path = os.path.join(RESULTS_ROOT, folder_name, 'metrics.npy')
            if not os.path.exists(metric_path): continue
            metrics = np.load(metric_path)
            
            records.append({
                'Site': site_name,
                'Model': model_name,
                'Horizon': horizon,
                'MSE': metrics[1], # MSE
                'MAE': metrics[0]  # MAE
            })
        except: continue
    return pd.DataFrame(records)

# ================= 🎨 修正后的绘图逻辑 =================
def plot_benchmark(df):
    if df.empty:
        print("❌ No valid data found!")
        return

    # 1. 过滤掉离谱的异常值 (MSE > 2.0)
    # 这样 Mamba 在 Site 1 的那个 279 就会被删掉，不会拉伸坐标轴
    df_clean = df[df['MSE'] < MSE_THRESHOLD].copy()
    n_removed = len(df) - len(df_clean)
    if n_removed > 0:
        print(f"⚠️ Removed {n_removed} outliers (MSE > {MSE_THRESHOLD}) to fix Y-axis scaling.")
        print("Dropped records:\n", df[df['MSE'] >= MSE_THRESHOLD][['Site', 'Model', 'MSE']])

    sns.set_theme(style="whitegrid", font_scale=1.1)
    horizons = df_clean['Horizon'].unique()
    
    for horizon in horizons:
        subset = df_clean[df_clean['Horizon'] == horizon].sort_values(by=['Site'])
        
        # 计算 Y 轴的合理上限 (取最大值的 1.1 倍，保证柱子不顶格)
        y_max = subset['MSE'].max() * 1.15
        
        plt.figure(figsize=(18, 9)) # 画布加大
        
        ax = sns.barplot(
            data=subset,
            x='Site',
            y='MSE',
            hue='Model',
            hue_order=MODEL_ORDER,
            palette="Spectral", # 换个颜色，Spectral 对比度更高
            edgecolor="black",
            linewidth=0.5
        )
        
        # 标注数值
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9, rotation=90)

        # 强制设置 Y 轴范围，确保能看清细节
        plt.ylim(0, y_max)
        
        plt.title(f"Model Performance - {horizon}\n(Outliers > {MSE_THRESHOLD} removed)", fontsize=16, fontweight='bold', pad=20)
        plt.xlabel("", fontweight='bold')
        plt.ylabel("MSE (Lower is Better)", fontweight='bold')
        plt.xticks(rotation=30, ha='right', fontsize=11)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        
        plt.tight_layout()
        
        save_name = f"Benchmark_Fixed_{horizon.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved fixed chart to: {save_path}")

if __name__ == "__main__":
    df = extract_metrics()
    plot_benchmark(df)