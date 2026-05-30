import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from math import pi

# 1. 加载数据
df = pd.read_csv('core_features_diagnosis.csv')
features = ['S_GVI', 'S_SVF', 'S_Enclosure', 'O_Score_Safety', 'O_Score_Comfort', 'O_Score_Beauty']

# 2. 计算聚类效度 (轮廓系数)
score = silhouette_score(df[features], df['Core_Cluster'])
print(f"聚类轮廓系数 (Silhouette Score): {score:.3f}")

# 3. 准备雷达图数据
cluster_means = df.groupby('Core_Cluster')[features].mean()
# 归一化处理以便在同一雷达图中显示
cluster_norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min())


def draw_radar(data):
    categories = list(data.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(len(data)):
        values = data.iloc[i].values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2, label=f'Cluster {i}')
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("典型街道聚类特征对比雷达图")
    plt.show()


draw_radar(cluster_norm)