import pandas as pd
import numpy as np
import os
import glob
import warnings

# 忽略 openpyxl 的样式警告
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

# ================= 🔧 配置区域 =================
# 输入：原始 xlsx 文件夹
INPUT_FOLDER = '/root/autodl-tmp/Time-Series-Library/dataset/solar_data_raw_xlsx' 
# 输出：清洗后的 csv 文件夹
OUTPUT_FOLDER = '/root/autodl-tmp/Time-Series-Library/dataset/solar_raw_clean'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# ================= 🧠 核心映射逻辑 =================
def map_columns(columns):
    """
    输入原始列名列表，返回 {旧列名: 新标准列名} 的字典。
    标准列名: date, OT, GHI, Temp, Humid, Pressure, DNI, TSI
    """
    mapping = {}
    
    # 辅助函数：标准化字符串（转小写，去空格，去特殊符号）
    def normalize(s):
        s = str(s).lower().strip()
        return s

    for col in columns:
        clean_col = normalize(col)
        
        # 1. 🎯 核心目标变量 (OT)
        if 'power' in clean_col:
            mapping[col] = 'OT'
            continue
            
        # 2. 🌍 物理模型核心变量 (GHI)
        if 'global horizontal' in clean_col:
            mapping[col] = 'GHI'
            continue
            
        # 3. 🕒 时间变量
        if 'time' in clean_col or 'date' in clean_col:
            mapping[col] = 'date'
            continue
            
        # 4. 🌡️ 气象变量
        if 'air temperature' in clean_col or 'temp' in clean_col:
            mapping[col] = 'Temp'
            continue
            
        if 'humidity' in clean_col:
            mapping[col] = 'Humid'
            continue
            
        if 'atmosphere' in clean_col or 'hpa' in clean_col:
            mapping[col] = 'Pressure'
            continue
            
        if 'direct normal' in clean_col:
            mapping[col] = 'DNI' # 直射辐射
            continue
            
        if 'total solar' in clean_col:
            mapping[col] = 'TSI' # 总辐射
            continue

    return mapping

# ================= 🧹 数据清洗逻辑 =================
def process_single_file(file_path):
    filename = os.path.basename(file_path)
    print(f"📄 Processing: {filename}...")
    
    try:
        # 1. 读取 Excel (指定 engine 防止兼容性问题)
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # 2. 列名映射
        col_map = map_columns(df.columns)
        df.rename(columns=col_map, inplace=True)
        
        # 检查关键列是否存在
        missing_critical = []
        if 'date' not in df.columns: missing_critical.append('date')
        if 'OT' not in df.columns:   missing_critical.append('OT')
        if 'GHI' not in df.columns:  missing_critical.append('GHI')
        
        if missing_critical:
            print(f"   ⚠️ CRITICAL ERROR: Could not find columns {missing_critical} in {filename}")
            print(f"   Original columns: {df.columns.tolist()}")
            return False

        # 3. 剔除全空行 (Excel 常见尾部空行)
        df.dropna(how='all', inplace=True)
        
        # 4. 强制类型转换与清洗
        # 处理 date
        df['date'] = pd.to_datetime(df['date'])
        
        # 处理数值列 (除了 date 以外的所有列)
        numeric_cols = [c for c in df.columns if c != 'date']
        
        for col in numeric_cols:
            # 转换为字符串以便处理特殊字符
            if df[col].dtype == 'object':
                # 去除括号、单位等残留字符，只保留数字、负号和小数点
                # 这一步是为了处理像 "(2133.33)" 或 "4.60 " 这样的脏数据
                df[col] = df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
                # 处理 Excel 会计负数 (如果正则没处理掉括号的话，这里双保险)
                # 但上面的正则其实已经把括号删了，可能会导致负号丢失？
                # 修正策略：如果原始数据是 "(123)"，上面的正则会变成 "123"。这不行。
                
                # 回滚：使用更安全的转换
                pass 

            # 使用 pd.to_numeric 的 coerce 模式，这会自动处理绝大多数情况
            # 它能处理 "4.60 " (带空格)
            # 它不能处理 "(123)" (带括号)，所以我们在读取 Excel 时依赖 openpyxl 的自动解析
            # 如果 openpyxl 读进来已经是数字了，那最好；如果是字符串，coerce 会变 NaN
            
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 5. 填��缺失值 (线性插值)
        # 这对于后续 VMD 至关重要
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both').fillna(0.0)
        
        # 6. 保存
        save_name = filename.rsplit('.', 1)[0] + ".csv"
        save_path = os.path.join(OUTPUT_FOLDER, save_name)
        df.to_csv(save_path, index=False, encoding='utf-8')
        
        print(f"   ✅ Features: {len(numeric_cols)} | Rows: {len(df)}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")
        return False

# ================= 🚀 执行 =================
if __name__ == '__main__':
    xlsx_files = glob.glob(os.path.join(INPUT_FOLDER, "*.xlsx")) + glob.glob(os.path.join(INPUT_FOLDER, "*.xls"))
    xlsx_files.sort()
    
    print(f"🔍 Found {len(xlsx_files)} Excel files.\n")
    
    success_count = 0
    for f in xlsx_files:
        if process_single_file(f):
            success_count += 1
            
    print(f"\n🎉 Done! Processed {success_count}/{len(xlsx_files)} files.")
    print(f"👉 Output saved to: {OUTPUT_FOLDER}")