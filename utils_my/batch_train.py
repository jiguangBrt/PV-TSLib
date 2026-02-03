import os
import pandas as pd
import subprocess

# ================= 核心路径配置 (使用绝对路径) =================
# 1. 项目根目录 (run.py 所在的目录)
project_root = "/root/autodl-tmp/Time-Series-Library/"

# 2. 数据集目录 (你的CSV所在的目录)
data_root_abs = os.path.join(project_root, "dataset/solar_processed/")

# ================= 训练配置 =================
models_to_run = ['Mamba', 'iTransformer']
seq_len = 96
pred_len = 96

# ================= 辅助函数：健壮读取 =================
def read_csv_safe(path):
    encodings = ['utf-8', 'gbk', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            return pd.read_csv(path, nrows=1, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise Exception(f"无法读取文件 {path}")

# ================= 主逻辑 =================
# 1. 强制切换工作目录到项目根目录 (确保能找到 run.py)
os.chdir(project_root)
print(f"已切换工作目录至: {os.getcwd()}")

if not os.path.exists(data_root_abs):
    print(f"❌ 错误：找不到数据目录 {data_root_abs}")
    exit(1)

files = [f for f in os.listdir(data_root_abs) if f.endswith('.csv')]
files.sort()

print(f"    准备开始训练，共检测到 {len(files)} 个数据文件...\n")

for file_name in files:
    file_path = os.path.join(data_root_abs, file_name)
    
    try:
        # 读取特征维度
        df = read_csv_safe(file_path)
        feat_dim = len(df.columns) - 1
        
        print(f"    当前文件: {file_name}")
        print(f"    特征维度: {feat_dim} (enc_in={feat_dim})")
        
        for model_name in models_to_run:
            # 生成短ID
            short_name = file_name.split('(')[0].strip().replace(' ', '_').lower()[:15]
            model_id = f"{model_name}_{short_name}"
            
            print(f"   正在启动: {model_name} (ID: {model_id})...")
            
            # 构建命令 (注意 data_path 只传文件名，root_path 传目录)
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
                f"--batch_size 16 "
                f"--train_epochs 5 "
                f"--patience 3 "
            )
            
            try:
                # 这里的 shell=True 会在当前工作目录(即项目根目录)下执行
                subprocess.check_call(cmd, shell=True)
                print(f"   ✅ {model_name} 训练完成！\n")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ {model_name} 训练失败 (Code: {e.returncode})\n")
                
    except Exception as e:
        print(f"❌ 处理文件 {file_name} 出错: {e}\n")

print("🎉 全部任务结束！")