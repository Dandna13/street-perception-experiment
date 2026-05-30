import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from scipy.spatial import cKDTree


# 1. 精确距离计算
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371000
    return c * r


# 2. 读取包含所有 1272 个点的插值底表
df = pd.read_csv("all_points_imputed_for_qgis.csv")

# 3. 构建空间 KDTree 并搜索 50 米阈值内的点对
coords = np.radians(df[['lat', 'lon']].values)
tree = cKDTree(coords)
radius_rad = 50 / 6371000
pairs = tree.query_pairs(r=radius_rad)

sequences = []
# 6 项需要求导的基线指标
indices_cols = ['S_GVI', 'S_SVF', 'S_Enclosure', 'S_Motorization', 'S_Walkability', 'S_Clutter']

# 4. 严格按照论文要求提取 [绝对差值 Delta] 与 [变异系数 CV]
for p1_idx, p2_idx in pairs:
    row1 = df.iloc[p1_idx]
    row2 = df.iloc[p2_idx]

    dist = haversine(row1['lon'], row1['lat'], row2['lon'], row2['lat'])

    if dist <= 50:
        seq_data = {
            'P1_id': int(row1['id']),
            'P2_id': int(row2['id']),
            'Distance_m': round(dist, 2)
        }

        for col in indices_cols:
            val1 = row1[col]
            val2 = row2[col]

            # 【量纲 1】：计算绝对差值 Delta
            seq_data[f'Delta_{col}'] = abs(val2 - val1)

            # 【量纲 2】：计算变异系数 CV (Coefficient of Variation = σ/μ)
            mean_val = (val1 + val2) / 2.0
            if mean_val == 0:
                cv_val = 0.0  # 防止分母为 0 导致报错
            else:
                # 计算两点的总体标准差
                std_val = np.std([val1, val2], ddof=0)
                cv_val = std_val / mean_val

            seq_data[f'CV_{col}'] = round(cv_val, 6)

        sequences.append(seq_data)

# 5. 导出符合要求的特征矩阵 F
seq_df = pd.DataFrame(sequences)
seq_df.to_csv("sequence_indices_full.csv", index=False)
print(f"数据处理完毕！生成了 {len(seq_df)} 个序列对。")
print(f"当前特征矩阵包含 {len(indices_cols) * 2} 项环境刺激变量（6 个 Delta + 6 个 CV）。")