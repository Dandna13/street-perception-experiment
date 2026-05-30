"""
data_preparation.py
用途: 加载 final_ml_dataset.csv，提取特征与目标变量，生成分组标签（空间交叉验证用）
输出: 生成 data_ready_for_modeling.csv 和 feature_names.txt
"""

import pandas as pd
import numpy as np


def main():
    # 文件路径
    input_file = "final_ml_dataset.csv"

    # 读取数据
    df = pd.read_csv(input_file)
    print(f"原始数据形状: {df.shape}")

    # 定义特征列 (Delta_* 和 CV_*)
    feature_cols = [c for c in df.columns if c.startswith(('Delta_', 'CV_'))]
    target_col = "O_Score_Rhythm"

    # 检查缺失值
    if df[feature_cols + [target_col]].isnull().any().any():
        print("存在缺失值，进行删除")
        df = df.dropna(subset=feature_cols + [target_col])

    # 生成空间分组标签 (模拟: 取 P1_id 的前三位作为街道组，实际应使用真实街道ID)
    # 注意: 此方法为示例，真实研究应使用已知的街道字段
    df['street_group'] = df['P1_id'].astype(str).str[:3]
    # 确保每个分组包含至少两个样本
    group_counts = df['street_group'].value_counts()
    valid_groups = group_counts[group_counts >= 2].index
    df = df[df['street_group'].isin(valid_groups)]
    print(f"有效分组数: {df['street_group'].nunique()}")

    # 保存处理后的数据
    output_file = "data_ready_for_modeling.csv"
    df.to_csv(output_file, index=False)
    print(f"准备数据已保存至 {output_file}")

    # 输出特征列表供其他脚本使用
    with open("feature_names.txt", "w") as f:
        f.write("\n".join(feature_cols))
    print("特征列表已保存至 feature_names.txt")


if __name__ == "__main__":
    main()