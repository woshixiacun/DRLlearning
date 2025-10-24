import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)

exit()
import pandas as pd
import numpy as np
from pathlib import Path

def find_csv_files_pathlib():
    """
    使用pathlib查找CSV文件并构建完整路径
    """
    # (1) 定位本文件夹的路径
    current_dir = Path.cwd()
    print(f"当前文件夹路径: {current_dir}")
    
    # 查找所有CSV文件
    csv_files = list(current_dir.glob("**/*.csv"))
    csv_file = csv_files[0]

    return csv_file.absolute()

def load_data(file_path):
    """
    读取CSV文件，返回inputs和outputs
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        print("文件基本信息:")
        print(f"总列数: {df.shape[1]}")
        print(f"总行数: {df.shape[0]}")
        # print("\n列名:")
        # print(df.columns.tolist())
        
        # 提取第2-4列作为input（索引1,2,3）
        inputs = df.iloc[:, 1:4]
        # 提取第5列作为output（索引4）
        outputs = df.iloc[:, 4]
        
        print(f"\n数据提取完成:")
        print(f"inputs形状: {inputs.shape}")
        print(f"outputs形状: {outputs.shape}")
        
        # # 显示数据类型
        # print(f"\ninputs数据类型:")
        # print(inputs.dtypes)
        # print(f"outputs数据类型: {outputs.dtype}")
        
        return inputs, outputs
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None, None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None, None


if __name__ == "__main__":
    csv_files = find_csv_files_pathlib()
    inputs, outputs = load_data(csv_files)
    
    if inputs is not None:
        print("\n数据预览:")
        print("Inputs (前5行):")
        print(inputs.head())
        print("\nOutputs (前5行):")
        print(outputs.head())