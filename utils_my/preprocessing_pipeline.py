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
INPUT_DIR = os.path.join(BASE_DIR, '../dataset/solar_raw_clean') 
OUTPUT_DIR = os.path.join(BASE_DIR, '../dataset/solar_processed_mvmd_xian') # 输出目录改名以示区分

# --- ⚙️ VMD 参数 ---
K_MODES = 8
ALPHA = 2000
TAU = 0

# --- 🌍 物理参数 (针对西安/西北地区修改) ---
# 西安大致坐标 (34.34 N, 108.94 E)
# 如果你知道具体电厂的经纬度，请在此处精确修改，这会显著提升 P_phy 的拟合度
DEFAULT_LAT = 34.0  
DEFAULT_LON = 109.0

# ⚠️ 关键设置：数据采集器的时间标准
# 即使电厂在西北，只要数据记录使用的是"北京时间"，这里必须是 'Asia/Shanghai'
DATA_TIMEZONE = 'Asia/Shanghai' 

def calc_physics_baseline(df, lat=DEFAULT_LAT, lon=DEFAULT_LON):
    try:
        # 1. 时间处理
        times = pd.to_datetime(df['date'])
        if times.dt.tz is None:
            # 告诉 pvlib：CSV里的时间是北京时间
            times = times.dt.tz_localize(DATA_TIMEZONE)
        else:
            times = times.dt.tz_convert(DATA_TIMEZONE)

        # 2. 物理建模 (使用西安经纬度 + 北京时间)
        location = pvlib.location.Location(lat, lon, tz=DATA_TIMEZONE)
        cs = location.get_clearsky(times)
        
        ghi_calc = cs['ghi'].values
        real_power = df['OT'].values
        
        # 3. 拟合系数计算
        valid_mask = ghi_calc > 10 
        if np.sum(valid_mask) > 0:
            # 计算这一天的光电转换效率近似值
            ratio = np.percentile(real_power[valid_mask], 95) / np.percentile(ghi_calc[valid_mask], 95)
            ratio = min(ratio, 2.0) 
        else:
            ratio = 0.0
            
        p_phy = ghi_calc * ratio
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
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"❌ Read failed: {e}")
        return 0
    
    if 'OT' not in df.columns:
        print(f"❌ Skipping {filename}: No 'OT' column found.")
        return 0
        
    # --- Step 1: 物理计算 (Xi'an Coords + BJ Time) ---
    p_raw = df['OT'].values
    p_phy = calc_physics_baseline(df)
    p_res = p_raw - p_phy
    
    # --- Step 2: 准备 MVMD 输入 ---
    targets = {}
    targets['PowerRes'] = p_res
    
    # 自动包含所有可能的变量
    potential_cols = ['GHI', 'DNI', 'TSI', 'Temp', 'Humid', 'Pressure']
    for col in potential_cols:
        if col in df.columns:
            targets[col] = df[col].values

    for k, v in targets.items():
        if not np.isfinite(v).all():
            targets[k] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Step 3: VMD ---
    pool_size = min(multiprocessing.cpu_count(), len(targets))
    pool = multiprocessing.Pool(processes=pool_size)
    results = []
    keys = []
    
    for key, signal in targets.items():
        keys.append(key)
        results.append(pool.apply_async(run_vmd, (signal,)))
    
    pool.close()
    pool.join()
    
    # --- Step 4: 输出 ---
    df_out = pd.DataFrame()
    df_out['date'] = df['date']
    df_out['OT'] = p_raw
    df_out['P_PHY'] = p_phy 
    
    for key, res in zip(keys, results):
        modes = res.get()
        for k in range(K_MODES):
            col_name = f'{key}_IMF{k+1}'
            df_out[col_name] = modes[:, k]
            
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    df_out.to_csv(save_path, index=False)
    
    feature_count = len(df_out.columns) - 3 
    print(f"✅ Saved to: {save_path} (Features: {feature_count})")
    return feature_count

if __name__ == '__main__':
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found.")
        exit()
        
    print(f"🔍 Processing {len(csv_files)} files with Location: Xi'an (Lat {DEFAULT_LAT}, Lon {DEFAULT_LON})...")
    
    for f in csv_files:
        process_single_file(f)
        
    print("\n🚀 All Done.")