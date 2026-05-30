"""
prepare_new_target.py
用途：将新生成的序列相对得分 (seq_score_mean) 合并到原始建模数据集中，
并重新生成街道分组标签，输出可用于建模的新文件。
"""

import pandas as pd


def main():
    # 1. 读取原始特征数据（包含 Δ 和 CV 指标）
    df_features = pd.read_csv("final_ml_dataset.csv")
    print(f"原始特征数据形状: {df_features.shape}")

    # 2. 读取新生成的序列得分
    df_seq_scores = pd.read_csv("sequence_relative_scores.csv")
    print(f"序列得分数据形状: {df_seq_scores.shape}")

    # 3. 合并：根据 P1_id, P2_id 将 seq_score_mean 添加到特征数据中
    df_merged = df_features.merge(
        df_seq_scores[['P1_id', 'P2_id', 'seq_score_mean']],
        on=['P1_id', 'P2_id'],
        how='left'
    )

    # 检查是否有未匹配的行
    missing = df_merged['seq_score_mean'].isnull().sum()
    if missing > 0:
        print(f"警告: {missing} 个序列对未匹配到得分，将删除这些行")
        df_merged = df_merged.dropna(subset=['seq_score_mean'])

    # 4. 用新得分替换原来的 O_Score_Rhythm 列（或直接覆盖）
    df_merged['O_Score_Rhythm'] = df_merged['seq_score_mean']

    # 5. 生成街道分组标签（模拟：取 P1_id 的前三位作为街道组）
    #    实际研究应使用真实的街道 ID，此处仅用于空间交叉验证示例
    df_merged['street_group'] = df_merged['P1_id'].astype(str).str[:3]
    # 过滤掉只有一个样本的街道组
    group_counts = df_merged['street_group'].value_counts()
    valid_groups = group_counts[group_counts >= 2].index
    df_merged = df_merged[df_merged['street_group'].isin(valid_groups)]
    print(f"有效街道分组数: {df_merged['street_group'].nunique()}")

    # 6. 保存新的建模数据集
    output_file = "final_ml_dataset_new_target.csv"
    df_merged.to_csv(output_file, index=False)
    print(f"新建模数据集已保存至 {output_file}")

    # 7. 保存特征列表（供下一步建模使用）
    feature_cols = [c for c in df_merged.columns if c.startswith(('Delta_', 'CV_'))]
    with open("feature_names_new.txt", "w") as f:
        f.write("\n".join(feature_cols))
    print(f"特征列表已保存至 feature_names_new.txt，共 {len(feature_cols)} 个特征")


if __name__ == "__main__":
    main()