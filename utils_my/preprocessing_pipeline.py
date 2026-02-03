import pandas as pd
import numpy as np
from vmdpy import VMD
import pvlib
from tqdm import tqdm
import multiprocessing
import os
import glob

# --- 📍 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, '../dataset/solar_raw_clean') # 读取清洗后的数据
OUTPUT_DIR = os.path.join(BASE_DIR, '../dataset/solar_processed_mvmd')

# --- ⚙️ VMD 参数配置 ---
K_MODES = 8
ALPHA = 2000
TAU = 0

# --- 🌍 物理参数 ---
DEFAULT_LAT = 37.0
DEFAULT_LON = 112.0

def calc_physics_baseline(df, lat=DEFAULT_LAT, lon=DEFAULT_LON):
    try:
        times = pd.to_datetime(df['date'])
        if times.dt.tz is None:
            times = pd.DatetimeIndex(times).tz_localize('UTC')
        else:
            times = pd.DatetimeIndex(times)

        location = pvlib.location.Location(lat, lon)
        cs = location.get_clearsky(times)
        
        ghi = cs['ghi'].values
        real_power = df['OT'].values
        
        valid_mask = ghi > 10 
        if np.sum(valid_mask) > 0:
            ratio = np.percentile(real_power[valid_mask], 95) / np.percentile(ghi[valid_mask], 95)
        else:
            ratio = 1.0
            
        p_phy = ghi * ratio
        p_phy = np.nan_to_num(p_phy, nan=0.0) 
        if len(p_phy) != len(df): p_phy = p_phy[:len(df)]
        return p_phy

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"⚠️ Physics calc failed: {e}. Using zeros.")
        return np.zeros(len(df))
    
def run_vmd(signal):
    if np.all(signal == signal[0]):
        return np.zeros((len(signal), K_MODES))
    try:
        u, _, _ = VMD(signal, ALPHA, TAU, K_MODES, 0, 1, 1e-7)
        return u.T
    except Exception as e:
        print(f"⚠️ VMD failed: {e}. Return zeros.")
        return np.zeros((len(signal), K_MODES))

def process_single_file(file_path):
    filename = os.path.basename(file_path)
    save_path = os.path.join(OUTPUT_DIR, filename)
    
    print(f"\n📄 Processing: {filename} ...")
    
    # 1. 读取数据 (已经是标准化的 clean 数据)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Read failed: {e}")
        return 0
    
    # 2. 物理计算
    if 'OT' not in df.columns:
        print(f"❌ Skipping {filename}: No 'OT' column found.")
        return 0
        
    p_raw = df['OT'].values
    p_phy = calc_physics_baseline(df)
    p_res = p_raw - p_phy
    
    # 3. 准备 MVMD 输入 (动态选择存在的列)
    targets = {}
    
    # (a) 核心变量
    targets['PowerRes'] = p_res
    
    # (b) 辅助变量 (如果存在才添加)
    if 'GHI' in df.columns:
        targets['GHI'] = df['GHI'].values
    if 'Temp' in df.columns:
        targets['Temp'] = df['Temp'].values
    if 'Humid' in df.columns:
        targets['Humid'] = df['Humid'].values
    if 'Pressure' in df.columns:
        targets['Pressure'] = df['Pressure'].values
    # 如果有 DNI/TSI 也可以加，根据你的需求
    
    # --- 检查 NaN/Inf ---
    for k, v in targets.items():
        if not np.isfinite(v).all():
            print(f"⚠️ Warning: {k} contains NaN/Inf. Filling with 0.")
            targets[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    # 4. 执行多进程 VMD
    pool = multiprocessing.Pool(processes=len(targets))
    results = []
    keys = []
    
    for key, signal in targets.items():
        keys.append(key)
        results.append(pool.apply_async(run_vmd, (signal,)))
    
    pool.close()
    pool.join()
    
    # 5. 组装结果
    df_out = pd.DataFrame()
    df_out['date'] = df['date']
    df_out['OT'] = p_raw      # 真值
    df_out['P_PHY'] = p_phy   # 物理基线
    
    # 放入分解后的分量
    for key, res in zip(keys, results):
        modes = res.get() # [L, K]
        for k in range(K_MODES):
            col_name = f'{key}_IMF{k+1}'
            df_out[col_name] = modes[:, k]
            
    # 6. 保存
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    df_out.to_csv(save_path, index=False)
    print(f"✅ Saved to: {save_path}")
    print(f"   Decomposed {len(targets)} variables -> {len(df_out.columns)-3} features.")
    
    # 返回特征数量 (减去 date, OT, P_PHY 这3个非输入列)
    feature_count = len(df_out.columns) - 3 
    return feature_count

if __name__ == '__main__':
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {INPUT_DIR}")
        exit()
        
    print(f"🔍 Found {len(csv_files)} files. Starting pipeline...")
    
    # 我们假设所有文件的特征维度是一样的，或者取第一个文件的维度作为参考
    # 如果每个文件因为缺失列导致特征维度不同，训练时会有麻烦（Enc_in 不匹配）
    # 但按照目前的清洗脚本，大部分主要列应该都在。
    
    enc_in_list = []
    for f in csv_files:
        feat_dim = process_single_file(f)
        enc_in_list.append(feat_dim)
        
    # 检查维度一致性
    if len(set(enc_in_list)) > 1:
        print("\n⚠️ WARNING: Feature dimensions inconsistent across files!")
        print(f"Dimensions found: {enc_in_list}")
        print("Model training might fail if batch_train mixes these files.")
    
    final_dim = enc_in_list[0] if enc_in_list else 0
    
    print("\n" + "="*50)
    print("🚀 All Done!")
    print(f"⚠️  Please use this for your run.py settings:")
    print(f"   --enc_in {final_dim} --c_out {final_dim}")
    print("="*50)