import os
import pandas as pd
import subprocess

# ================= 🔧 核心配置区域 =================
PROJECT_ROOT = "/root/autodl-tmp/Time-Series-Library/"
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset/solar_processed_mvmd/")

MODELS = ['PhysicsMamba']
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

def get_physicsmamba_params():
    """
    PhysicsMamba 专用超参数
    d_state, d_conv, expand 在模型内部使用默认值
    """
    return {
        "d_model": 128,
        "e_layers": 2,
        "batch_size": 128,
        "learning_rate": 0.0001,
        "train_epochs": 20,
        "patience": 5,
    }

# ================= 🚀 主逻辑 =================
def main():
    if os.getcwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)

    if not os.path.exists(DATA_ROOT):
        print(f"❌ 错误: 数据目录不存在 {DATA_ROOT}")
        return

    csv_files = [f for f in os.listdir(DATA_ROOT) if f.endswith('.csv')]
    csv_files.sort()
    
    total_tasks = len(csv_files) * len(PRED_LENS)
    print(f"🔍 发现 {len(csv_files)} 个站点（MVMD 预处理后）")
    print(f"🔥 [PhysicsMamba - RTX 4090 Mode] 预计执行 {total_tasks} 次训练...\n")

    task_count = 0
    params = get_physicsmamba_params()

    for csv_file in csv_files:
        file_path = os.path.join(DATA_ROOT, csv_file)
        feat_dim = get_csv_dim(file_path)
        if feat_dim is None: 
            continue

        site_id_clean = csv_file.replace('.csv', '').replace('(', '').replace(')', '').replace(' ', '_')
        print(f"📊 {site_id_clean}: enc_in = {feat_dim}")
        
        for pred_len in PRED_LENS:
            task_count += 1
            task_tag = "Short" if pred_len <= 48 else "Long"
            model_id_arg = f"{site_id_clean}_{task_tag}{pred_len}"
            
            print(f"\n{'='*80}")
            print(f"[{task_count}/{total_tasks}] 🚀 {site_id_clean} | Pred={pred_len} | Batch={params['batch_size']}")
            print(f"{'='*80}\n")

            cmd = (
                f"python run.py "
                f"--task_name long_term_forecast "
                f"--is_training 1 "
                f"--root_path \"{DATA_ROOT}\" "
                f"--data_path \"{csv_file}\" "
                f"--model_id {model_id_arg} "
                f"--model PhysicsMamba "
                f"--data custom "
                f"--features M "
                f"--seq_len {SEQ_LEN} "
                f"--label_len 48 "
                f"--pred_len {pred_len} "
                f"--e_layers {params['e_layers']} "
                f"--d_layers 1 "
                f"--factor 3 "
                f"--enc_in {feat_dim} "
                f"--dec_in {feat_dim} "
                f"--c_out {feat_dim} "
                f"--des 'PhysicsMamba_MVMD' "
                f"--d_model {params['d_model']} "
                f"--batch_size {params['batch_size']} "
                f"--learning_rate {params['learning_rate']} "
                f"--train_epochs {params['train_epochs']} "
                f"--patience {params['patience']} "
                f"--num_workers 6 "
                f"--itr 1 "
                f"--use_gpu "
                f"--gpu 0"
            )

            try:
                subprocess.run(cmd, shell=True, check=True)
                print(f"\n✅ [{task_count}/{total_tasks}] 完成: {model_id_arg}\n")
            except subprocess.CalledProcessError as e:
                print(f"\n❌ [{task_count}/{total_tasks}] 失败: {model_id_arg}")
                print(f"   ⚠️ 跳过...\n")
    
    print("\n🎉 所有训练完成！")

if __name__ == "__main__":
    main()