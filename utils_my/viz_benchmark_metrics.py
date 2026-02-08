import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ================= 🔧 配置区域 =================
PROJECT_ROOT = "/root/autodl-tmp/Time-Series-Library/"
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results/")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset/viz_results/")

# 模型顺序 (PhysicsMamba 放最前)
MODEL_ORDER = [
    'PhysicsMamba',
    'iTransformer', 
    'PatchTST', 
    'Mamba', 
    'Transformer', 
    'Informer', 
    'Autoformer'
]

# 异常值阈值 (MSE > 2.0 视为训练失败)
MSE_THRESHOLD = 2.0 

# 颜色配置
COLORS = {
    'PhysicsMamba': '#E74C3C',    # 红色 (主角)
    'iTransformer': '#3498DB',    # 蓝色
    'PatchTST': '#2ECC71',        # 绿色
    'Mamba': '#F39C12',           # 橙色
    'Transformer': '#9B59B6',     # 紫色
    'Informer': '#1ABC9C',        # 青色
    'Autoformer': '#95A5A6'       # 灰色
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= 📊 数据提取函数 =================
def extract_all_metrics():
    """
    提取所有性能指标:
    - MSE/MAE (从 metrics.npy)
    - ��练时间、推理时间、FLOPs (从 performance.npy)
    """
    records = []
    exp_dirs = [d for d in os.listdir(RESULTS_ROOT) if os.path.isdir(os.path.join(RESULTS_ROOT, d))]
    print(f"🔍 Found {len(exp_dirs)} experiment records in {RESULTS_ROOT}\n")
    
    for folder_name in exp_dirs:
        try:
            # Step 1: 提取 Horizon
            if 'Short24' in folder_name:
                horizon = 'Short Term (24)'
                tag = 'Short24'
            elif 'Long96' in folder_name:
                horizon = 'Long Term (96)'
                tag = 'Long96'
            else:
                continue
            
            # Step 2: 提取模型名
            model_name = "Unknown"
            for m in MODEL_ORDER:
                if m in folder_name:
                    model_name = m
                    break
            if model_name == "Unknown":
                continue
            
            # Step 3: 提取 Site
            start_marker = 'forecast_'
            end_marker = f'_{tag}'
            start_idx = folder_name.find(start_marker)
            if start_idx == -1:
                continue
            start_idx += len(start_marker)
            end_idx = folder_name.find(end_marker, start_idx)
            if end_idx == -1:
                continue
            site_name = folder_name[start_idx:end_idx]
            
            # Step 4: 读取基础指标 (metrics.npy)
            metric_path = os.path.join(RESULTS_ROOT, folder_name, 'metrics.npy')
            if not os.path.exists(metric_path):
                continue
            metrics = np.load(metric_path)
            mse = float(metrics[1])
            mae = float(metrics[0])
            
            # Step 5: 读取性能指标 (performance.npy) - 方案 A 修改后的输出
            perf_path = os.path.join(RESULTS_ROOT, folder_name, 'performance.npy')
            if os.path.exists(perf_path):
                try:
                    perf = np.load(perf_path, allow_pickle=True).item()
                    train_time = perf.get('total_train_time', None)
                    inference_time = perf.get('avg_inference_time', None)
                    flops = perf.get('flops', 'N/A')
                    params = perf.get('params', 'N/A')
                except:
                    train_time = None
                    inference_time = None
                    flops = 'N/A'
                    params = 'N/A'
            else:
                train_time = None
                inference_time = None
                flops = 'N/A'
                params = 'N/A'
            
            records.append({
                'Site': site_name,
                'Model': model_name,
                'Horizon': horizon,
                'MSE': mse,
                'MAE': mae,
                'Train Time (s)': train_time,
                'Inference Time (ms)': inference_time * 1000 if inference_time else None,
                'FLOPs': str(flops),
                'Params': str(params)
            })
            
            print(f"✅ {site_name:20s} | {model_name:15s} | {horizon:18s} | MSE={mse:.4f}")
            
        except Exception as e:
            print(f"⚠️ Failed: {folder_name[:50]}... - {e}")
            continue
    
    df = pd.DataFrame(records)
    print(f"\n📊 Total valid records: {len(df)}")
    return df

# ================= 🎨 可视化函数 =================
def plot_mse_comparison(df):
    """绘制 MSE 对比柱状图"""
    df_clean = df[df['MSE'] < MSE_THRESHOLD].copy()
    
    sns.set_theme(style="whitegrid", font_scale=1.1)
    horizons = sorted(df_clean['Horizon'].unique())
    
    for horizon in horizons:
        subset = df_clean[df_clean['Horizon'] == horizon].copy()
        subset = subset.sort_values(by=['Site', 'Model'])
        
        y_max = subset['MSE'].max() * 1.2
        
        fig, ax = plt.subplots(figsize=(20, 10))
        
        sns.barplot(
            data=subset,
            x='Site',
            y='MSE',
            hue='Model',
            hue_order=MODEL_ORDER,
            palette=COLORS,
            edgecolor="black",
            linewidth=0.6,
            ax=ax
        )
        
        # 标注数值
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8, rotation=90)
        
        ax.set_ylim(0, y_max)
        plt.title(
            f"MSE Comparison - {horizon}\n(Outliers > {MSE_THRESHOLD} removed)", 
            fontsize=18, fontweight='bold', pad=20
        )
        plt.xlabel("Site", fontsize=14, fontweight='bold')
        plt.ylabel("MSE (Lower is Better)", fontsize=14, fontweight='bold')
        plt.xticks(rotation=35, ha='right', fontsize=11)
        plt.legend(title='Model', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=11)
        
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        plt.tight_layout()
        
        save_name = f"MSE_Comparison_{horizon.replace(' ', '_').replace('(', '').replace(')', '')}.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()

def plot_inference_time(df):
    """绘制推理时间对比"""
    df_time = df[df['Inference Time (ms)'].notna()].copy()
    if df_time.empty:
        print("⚠️ No inference time data found. Skipping...")
        return
    
    # 按模型聚合平均推理时间
    df_avg = df_time.groupby('Model')['Inference Time (ms)'].mean().reset_index()
    df_avg = df_avg.sort_values('Inference Time (ms)')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df_avg['Model'], df_avg['Inference Time (ms)'], 
                   color=[COLORS.get(m, '#95A5A6') for m in df_avg['Model']],
                   edgecolor='black', linewidth=0.8)
    
    # 标注数值
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.2f} ms', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Average Inference Time (ms)', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.title('Inference Time Comparison (Avg per Batch)', fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'Inference_Time_Comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

def plot_train_time(df):
    """绘制训练时间对比"""
    df_time = df[df['Train Time (s)'].notna()].copy()
    if df_time.empty:
        print("⚠️ No training time data found. Skipping...")
        return
    
    # 按模型聚合平均训练时间
    df_avg = df_time.groupby('Model')['Train Time (s)'].mean().reset_index()
    df_avg['Train Time (min)'] = df_avg['Train Time (s)'] / 60
    df_avg = df_avg.sort_values('Train Time (min)')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(df_avg['Model'], df_avg['Train Time (min)'], 
                   color=[COLORS.get(m, '#95A5A6') for m in df_avg['Model']],
                   edgecolor='black', linewidth=0.8)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f} min', ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Average Training Time (minutes)', fontsize=12, fontweight='bold')
    plt.ylabel('Model', fontsize=12, fontweight='bold')
    plt.title('Training Time Comparison', fontsize=14, fontweight='bold', pad=15)
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'Training_Time_Comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

def plot_comprehensive_dashboard(df):
    """综合仪表盘: MSE + 时间 + FLOPs"""
    df_clean = df[df['MSE'] < MSE_THRESHOLD].copy()
    
    # 按模型聚合平均值
    agg_dict = {
        'MSE': 'mean',
        'MAE': 'mean',
        'Inference Time (ms)': 'mean',
        'Train Time (s)': 'mean'
    }
    df_summary = df_clean.groupby('Model').agg(agg_dict).reset_index()
    df_summary = df_summary.sort_values('MSE')
    
    # 创建 2x2 子图
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 子图 1: MSE
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(df_summary['Model'], df_summary['MSE'], 
                    color=[COLORS.get(m, '#95A5A6') for m in df_summary['Model']],
                    edgecolor='black', linewidth=0.8)
    ax1.set_title('Average MSE', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 子图 2: MAE
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(df_summary['Model'], df_summary['MAE'], 
                    color=[COLORS.get(m, '#95A5A6') for m in df_summary['Model']],
                    edgecolor='black', linewidth=0.8)
    ax2.set_title('Average MAE', fontsize=14, fontweight='bold')
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.005, 
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 子图 3: 推理时间
    ax3 = fig.add_subplot(gs[1, 0])
    df_time = df_summary[df_summary['Inference Time (ms)'].notna()]
    if not df_time.empty:
        bars3 = ax3.bar(df_time['Model'], df_time['Inference Time (ms)'], 
                        color=[COLORS.get(m, '#95A5A6') for m in df_time['Model']],
                        edgecolor='black', linewidth=0.8)
        ax3.set_title('Avg Inference Time', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Time (ms)', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', linestyle='--', alpha=0.3)
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 子图 4: 训练时间
    ax4 = fig.add_subplot(gs[1, 1])
    df_train = df_summary[df_summary['Train Time (s)'].notna()].copy()
    if not df_train.empty:
        df_train['Train Time (min)'] = df_train['Train Time (s)'] / 60
        bars4 = ax4.bar(df_train['Model'], df_train['Train Time (min)'], 
                        color=[COLORS.get(m, '#95A5A6') for m in df_train['Model']],
                        edgecolor='black', linewidth=0.8)
        ax4.set_title('Avg Training Time', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Time (minutes)', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(axis='y', linestyle='--', alpha=0.3)
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + 1, 
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Comprehensive Performance Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    save_path = os.path.join(OUTPUT_DIR, 'Comprehensive_Dashboard.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

def export_summary_table(df):
    """导出 FLOPs 和参数量汇总表"""
    # 提取每个模型的 FLOPs 和 Params (去重)
    flops_data = df[['Model', 'FLOPs', 'Params']].drop_duplicates(subset=['Model'])
    
    # 计算平均性能指标
    df_clean = df[df['MSE'] < MSE_THRESHOLD].copy()
    perf_summary = df_clean.groupby('Model').agg({
        'MSE': 'mean',
        'MAE': 'mean',
        'Inference Time (ms)': 'mean',
        'Train Time (s)': lambda x: x.mean() / 60  # 转为分钟
    }).reset_index()
    
    # 合并
    summary = perf_summary.merge(flops_data, on='Model', how='left')
    summary = summary.rename(columns={'Train Time (s)': 'Train Time (min)'})
    
    # 排序
    summary = summary.sort_values('MSE')
    
    # 保存 CSV
    csv_path = os.path.join(OUTPUT_DIR, 'Performance_Summary.csv')
    summary.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"\n📊 Performance summary saved to: {csv_path}")
    
    # 打印表格
    print("\n" + "="*100)
    print("Model Performance Summary".center(100))
    print("="*100)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(summary.to_string(index=False))
    print("="*100 + "\n")

def compute_improvement(df):
    """计算 PhysicsMamba 相对最佳 Baseline 的改进"""
    if 'PhysicsMamba' not in df['Model'].values:
        print("⚠️ No PhysicsMamba results found.")
        return
    
    df_clean = df[df['MSE'] < MSE_THRESHOLD].copy()
    
    results = []
    for horizon in df_clean['Horizon'].unique():
        for site in df_clean['Site'].unique():
            pm_data = df_clean[(df_clean['Model'] == 'PhysicsMamba') & 
                               (df_clean['Horizon'] == horizon) & 
                               (df_clean['Site'] == site)]
            if pm_data.empty:
                continue
            pm_mse = pm_data['MSE'].values[0]
            
            baselines = df_clean[(df_clean['Model'] != 'PhysicsMamba') & 
                                 (df_clean['Horizon'] == horizon) & 
                                 (df_clean['Site'] == site)]
            if baselines.empty:
                continue
            
            best_baseline = baselines.loc[baselines['MSE'].idxmin()]
            best_mse = best_baseline['MSE']
            best_model = best_baseline['Model']
            
            improvement = ((best_mse - pm_mse) / best_mse) * 100
            
            results.append({
                'Site': site,
                'Horizon': horizon,
                'PhysicsMamba MSE': pm_mse,
                'Best Baseline': best_model,
                'Baseline MSE': best_mse,
                'Improvement (%)': improvement
            })
    
    result_df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "PhysicsMamba_Improvement.csv")
    result_df.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print("PhysicsMamba vs Best Baseline".center(80))
    print("="*80)
    print(result_df.to_string(index=False))
    print("="*80)
    print(f"Average Improvement: {result_df['Improvement (%)'].mean():.2f}%")
    print(f"Wins: {len(result_df[result_df['Improvement (%)'] > 0])} / {len(result_df)}")
    print("="*80 + "\n")

# ================= 🚀 主函数 =================
if __name__ == "__main__":
    print("🔍 Extracting all metrics from results folder...\n")
    df = extract_all_metrics()
    
    if df.empty:
        print("❌ No data extracted. Check your results folder!")
    else:
        print(f"\n✅ Extracted {len(df)} records.")
        
        # 1. MSE 对比图
        print("\n🎨 Generating MSE comparison charts...")
        plot_mse_comparison(df)
        
        # 2. 推理时间对比
        print("\n⏱️ Generating inference time comparison...")
        plot_inference_time(df)
        
        # 3. 训练时间对比
        print("\n⏱️ Generating training time comparison...")
        plot_train_time(df)
        
        # 4. 综合仪表盘
        print("\n📊 Generating comprehensive dashboard...")
        plot_comprehensive_dashboard(df)
        
        # 5. 导出汇总表
        print("\n📋 Exporting summary table...")
        export_summary_table(df)
        
        # 6. 计算改进率
        print("\n📈 Computing improvement over baselines...")
        compute_improvement(df)
        
        print("\n🎉 All visualizations complete!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")