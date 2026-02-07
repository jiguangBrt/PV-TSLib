import os
import pandas as pd
import subprocess

# ================= 🔧 核心配置区域 =================
PROJECT_ROOT = "/root/autodl-tmp/Time-Series-Library/"
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset/solar_raw_clean/")

# 1. 定义基线模型
MODELS = [
    'iTransformer', 
    'PatchTST', 
    'Mamba',          
    'Transformer', 
    'Informer', 
    'Autoformer'
]

PRED_LENS = [24, 96] 
SEQ_LEN = 96

# ================= 🛠️ 辅助函数 =================
def get_csv_dim(path):
    try:
        df = pd.read_csv(path, nrows=1)
        return len(df.columns) - 1
    except Exception as e:
        print(f"❌ 无法读取文件 {path}: {e}")
        return None

def get_model_params(model_name):
    """
    针对 RTX 4090 的激进配置
    """
    # 基础配置 (Informer, Autoformer, Transformer)
    base_params = {
        "d_model": 512,
        "d_ff": 2048,
        "batch_size": 128,  # 激进提升: 32 -> 128
        "learning_rate": 0.0001
    }
    
    if model_name == 'Mamba':
        # Mamba 极度省显存，但需要控制 d_state
        return {
            "d_model": 512,    # 提升维度
            "d_ff": 32,        # d_state 保持 32
            "batch_size": 256, # 直接拉满
            "learning_rate": 0.001
        }
    elif model_name == 'iTransformer':
        # iTransformer 显存占用极低，计算极快
        return {
            "d_model": 512,
            "d_ff": 2048,
            "batch_size": 256, # 直接拉满
            "learning_rate": 0.0001
        }
    elif model_name == 'PatchTST':
        # PatchTST 显存占用稍高 (O(L^2) Attention)，保守一点
        return {
            "d_model": 512,
            "d_ff": 2048,
            "batch_size": 64,  # PatchTST 64 应该能吃满 4090
            "learning_rate": 0.0001
        }
    else:
        return base_params

# ================= 🚀 主逻辑 =================
def main():
    if os.getcwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)

    if not os.path.exists(DATA_ROOT):
        print(f"❌ 错误: 数据目录不存在 {DATA_ROOT}")
        return

    csv_files = [f for f in os.listdir(DATA_ROOT) if f.endswith('.csv')]
    csv_files.sort()
    
    total_tasks = len(csv_files) * len(MODELS) * len(PRED_LENS)
    print(f"🔍 发现 {len(csv_files)} 个站点，{len(MODELS)} 个模型。")
    print(f"🔥 [RTX 4090 Mode] 预计执行 {total_tasks} 次训练...\n")

    task_count = 0

    for csv_file in csv_files:
        file_path = os.path.join(DATA_ROOT, csv_file)
        feat_dim = get_csv_dim(file_path)
        if feat_dim is None: continue

        site_id_clean = csv_file.replace('.csv', '').replace('(', '').replace(')', '').replace(' ', '_')
        
        for model_name in MODELS:
            params = get_model_params(model_name)
            
            for pred_len in PRED_LENS:
                task_count += 1
                task_tag = "Short" if pred_len <= 48 else "Long"
                model_id_arg = f"{site_id_clean}_{task_tag}{pred_len}"
                
                print(f"[{task_count}/{total_tasks}] 🚀 {model_name} | {site_id_clean} | Len={pred_len} | BS={params['batch_size']}")

                cmd = (
                    f"python run.py "
                    f"--task_name long_term_forecast "
                    f"--is_training 1 "
                    f"--root_path \"{DATA_ROOT}\" "
                    f"--data_path \"{csv_file}\" "
                    f"--model_id {model_id_arg} "
                    f"--model {model_name} "
                    f"--data custom "
                    f"--features M "
                    f"--seq_len {SEQ_LEN} "
                    f"--label_len 48 "
                    f"--pred_len {pred_len} "
                    f"--e_layers 2 "
                    f"--d_layers 1 "
                    f"--factor 3 "
                    f"--enc_in {feat_dim} "
                    f"--dec_in {feat_dim} "
                    f"--c_out {feat_dim} "
                    f"--des 'Exp' "
                    f"--d_model {params['d_model']} "
                    f"--d_ff {params['d_ff']} "
                    f"--batch_size {params['batch_size']} "
                    f"--learning_rate {params['learning_rate']} "
                    f"--train_epochs 20 "     
                    f"--patience 5 "          
                    f"--num_workers 6 "       
                    f"--itr 1 "
                )

                try:
                    subprocess.run(cmd, shell=True, check=True)
                    print(f"   ✅ Done.\n")
                except subprocess.CalledProcessError:
                    print(f"   ❌ Failed. Skipping...\n")
    
    print("\n🎉 所有实验结束！")

if __name__ == "__main__":
    main()