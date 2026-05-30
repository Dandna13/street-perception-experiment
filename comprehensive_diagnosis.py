import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# ================= 1. 加载数据 =================
df = pd.read_csv('merged_analysis_data.csv')
# 加载坐标文件 (用于地图可视化)
try:
    df_geo = pd.read_csv('sampled_points.csv')
    # 统一列名
    if 'id' in df_geo.columns:
        df_geo = df_geo.rename(columns={'id': 'point_id'})
    # 合并坐标
    df = pd.merge(df, df_geo[['point_id', 'lat', 'lon']], on='point_id', how='left')
except:
    print("⚠️ 未找到 sample_points.csv，输出结果将不包含经纬度。")

# ================= 2. 定义指标的正负向 =================
# 正向指标 (越高越好) -> 需要反向处理来计算“紧迫度”
positive_metrics = [
    'S_GVI', 'S_Walkability', 'S_SVF',
    'O_Score_Safety', 'O_Score_Comfort', 'O_Score_Beauty'
]

# 负向指标 (越低越好) -> 直接计入“紧迫度”
# 注意：根据您的回归分析，S_Enclosure 和 S_Motorization 通常会导致舒适度下降，视为负向
negative_metrics = [
    'S_Enclosure', 'S_Motorization', 'S_Clutter',
    'O_Score_Depressing'
]

# 确保列存在
pos_cols = [c for c in positive_metrics if c in df.columns]
neg_cols = [c for c in negative_metrics if c in df.columns]

# ================= 3. 数据归一化 (0-1) =================
scaler = MinMaxScaler()
df_norm = df.copy()

# 对所有参与计算的列进行归一化
analysis_cols = pos_cols + neg_cols
df_norm[analysis_cols] = scaler.fit_transform(df[analysis_cols])

# ================= 4. 计算“综合改造紧迫度” (Urgency Index) =================
# 逻辑：紧迫度 = (所有负向指标归一化值之和) + (所有正向指标的反向值之和)
# 含义：负面因素越多，或者正面因素越少，分值越高
df['Score_Neg'] = df_norm[neg_cols].mean(axis=1)
df['Score_Pos_Inverted'] = 1 - df_norm[pos_cols].mean(axis=1) # 正面越低，问题越大

df['Urgency_Index'] = (df['Score_Neg'] + df['Score_Pos_Inverted']) / 2 * 10
# 结果映射到 0-10 分 (10分代表最烂，必须马上改)

# ================= 5. K-Means 聚类 (识别问题模式) =================
# 我们把街道分为 4 类 (可根据需要调整 n_clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster_Label'] = kmeans.fit_predict(df_norm[analysis_cols])

# 分析每一类的特征
cluster_summary = df.groupby('Cluster_Label')[analysis_cols].mean()
print("\n=== 聚类特征分析 (归一化后均值) ===")
print(cluster_summary)

# 为每一类自动打标签 (简化版)
# 找出每一类里最突出的“短板” (即正向指标最低，或负向指标最高)
def get_cluster_name(row):
    # 找最严重的负向指标
    worst_neg = row[neg_cols].idxmax()
    # 找最缺乏的正向指标
    worst_pos = row[pos_cols].idxmin()
    return f"High {worst_neg} & Low {worst_pos}"

# ================= 6. 导出结果 =================
# 按紧迫度排序
df_final = df.sort_values(by='Urgency_Index', ascending=False)

output_file = 'final_comprehensive_diagnosis.csv'
df_final.to_csv(output_file, index=False)

print(f"\n✅ 综合诊断完成！")
print(f"结果已保存至 {output_file}")
print("列说明：")
print("  - Urgency_Index: 综合改造紧迫度 (0-10)，分数越高代表环境越差，越需要设计介入。")
print("  - Cluster_Label: 聚类类别 (0-3)，代表该街道属于哪种问题类型。")

print("\n--- Top 5 最急需改造的采样点 ---")
cols_to_show = ['point_id', 'Urgency_Index', 'Cluster_Label', 'lat', 'lon']
print(df_final[cols_to_show].head())
