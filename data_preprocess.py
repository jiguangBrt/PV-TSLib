import pandas as pd
import os

# ================= 配置区域 =================
input_folder = '/root/autodl-tmp/Time-Series-Library/dataset/solar_data' 
output_folder = '/root/autodl-tmp/Time-Series-Library/dataset/solar_processed' 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ================= 处理逻辑 =================
def process_file(file_path, save_path):
    # 尝试多种编码，专门针对特殊符号
    encodings_to_try = ['utf-8', 'gbk', 'cp1252', 'latin1']
    df = None
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break 
        except UnicodeDecodeError:
            continue
            
    if df is None:
        raise Exception("无法解码，请检查文件源！")

    # 1. 自动重命名
    # 第一列 -> date
    # 最后一列 -> OT (无论它叫 Power 还是其他)
    columns_map = {
        df.columns[0]: 'date',
        df.columns[-1]: 'OT'
    }
    df.rename(columns=columns_map, inplace=True)
    
    # 2. 时间标准化
    df['date'] = pd.to_datetime(df['date'])
    
    # 3. 填补缺失值
    if df.isnull().values.any():
        df = df.fillna(method='ffill').fillna(0)

    # 4. 计算特征数量 (总列数 - 时间列)
    feature_count = len(df.columns) - 1

    # 5. 保存 (UTF-8)
    df.to_csv(save_path, index=False, encoding='utf-8')
    
    return feature_count

# ================= 执行 =================
files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
files.sort() # 排序，方便看
print(f"开始处理 {len(files)} 个文件...\n")

for file in files:
    in_path = os.path.join(input_folder, file)
    out_path = os.path.join(output_folder, file)
    try:
        feat_dim = process_file(in_path, out_path)
        print(f"✅ {file}")
        print(f"   ➜ 参数建议: --enc_in {feat_dim} --dec_in {feat_dim} --c_out {feat_dim}")
    except Exception as e:
        print(f"❌ {file} 失败: {str(e)}")

print("\n🎉 处理完成！请根据上方打印的“参数建议”修改运行命令。")