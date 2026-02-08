import os
import csv

def scan_csv_headers(directory_path):
    """

    Args:
        directory_path (str): 要扫描的目录路径。
    """
    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory_path, filename)
            try:
                with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.reader(csvfile)
                    header = next(reader)  # 读取第一行
                    print(f"{filename}: {','.join(header)}")
            except FileNotFoundError:
                print(f"错误: 文件 '{filename}' 未找到。")
            except Exception as e:
                print(f"处理文件 '{filename}' 时发生错误: {e}")

# 示例用法
if __name__ == "__main__":
    # 请将 'your_directory_path_here' 替换为你的CSV文件所在的实际目录
    directory_to_scan = '/root/autodl-tmp/Time-Series-Library/dataset/solar_processed_mvmd'
    
    # 确保目录存在，否则会报错
    if not os.path.isdir(directory_to_scan):
        print(f"错误: 目录 '{directory_to_scan}' 不存在。请替换为你的实际目录。")

    scan_csv_headers(directory_to_scan)