import os
import pandas as pd
import subprocess

# ================= 核心路径配置 =================
project_root = "/root/autodl-tmp/Time-Series-Library/"
data_root_abs = os.path.join(project_root, "dataset/solar_processed/")

# ================= 训练配置 =================
models_to_run = ['Mamba', 'iTransformer']
seq_len = 96
pred_len = 96
train_epochs = 1
batch_size = 8

# ================= 辅助函数 =================
def read_csv_safe(path):
    encodings = ['utf-8', 'gbk', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            return pd.read_csv(path, nrows=1, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise Exception(f"无法读取文件 {path}")

# ================= 主逻辑 =================
os.chdir(project_root)
print(f"已切换工作目录至: {os.getcwd()}")

if not os.path.exists(data_root_abs):
    print(f"❌ 错误：找不到数据目录 {data_root_abs}")
    exit(1)

all_files = [f for f in os.listdir(data_root_abs) if f.endswith('.csv')]
all_files.sort()

if not all_files:
    print("❌ 目录中没有 CSV 文件！")
    exit(1)

files = all_files[:1]
print(f"🧪 [测试模式] 仅选中第 1 个文件进行验证: {files[0]}\n")

for file_name in files:
    file_path = os.path.join(data_root_abs, file_name)
    try:
        df = read_csv_safe(file_path)
        feat_dim = len(df.columns) - 1
        print(f"    当前文件: {file_name} | 特征维度: {feat_dim}")
        
        for model_name in models_to_run:
            short_name = file_name.split('(')[0].strip().replace(' ', '_').lower()[:15]
            model_id = f"TEST_{model_name}_{short_name}"
            
            # ---> 关键修复：针对不同模型设置 d_ff <---
            if model_name == 'Mamba':
                d_ff_val = 32  # Mamba 的 d_state 必须 <= 256，通常取 16 或 32
                print(f"   🔧 针对 Mamba 修正 d_ff (d_state) = {d_ff_val}")
            else:
                d_ff_val = 2048 # Transformer 类模型通常很大
            
            print(f"   🚀 正在启动测试: {model_name} (ID: {model_id})...")
            
            cmd = (
                f"python run.py "
                f"--task_name long_term_forecast "
                f"--is_training 1 "
                f"--root_path \"{data_root_abs}\" "
                f"--data_path \"{file_name}\" "
                f"--model_id {model_id} "
                f"--model {model_name} "
                f"--data custom "
                f"--features M "
                f"--seq_len {seq_len} "
                f"--label_len 48 "
                f"--pred_len {pred_len} "
                f"--e_layers 2 "
                f"--d_layers 1 "
                f"--factor 3 "
                f"--enc_in {feat_dim} "
                f"--dec_in {feat_dim} "
                f"--c_out {feat_dim} "
                f"--des 'Exp' "
                f"--itr 1 "
                f"--batch_size {batch_size} "
                f"--train_epochs {train_epochs} "
                f"--patience 3 "
                f"--d_ff {d_ff_val} "  # 使用动态设置的 d_ff
            )
            
            try:
                subprocess.check_call(cmd, shell=True)
                print(f"   ✅ {model_name} 测试通过！\n")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ {model_name} 测试失败 (Code: {e.returncode})\n")
    except Exception as e:
        print(f"❌ 处理文件 {file_name} 出错: {e}\n")

print("🎉 测试结束！")