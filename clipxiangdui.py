import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ------------------------------
# 1. 读取数据
# ------------------------------
clip_df = pd.read_csv("clip_base_scores.csv")
trueskill_df = pd.read_csv("point_relative_scores.csv")

# 提取得分列
clip_scores = clip_df['base_score']
relative_scores = trueskill_df['relative_score']

# 计算统计量
mean_clip = clip_scores.mean()
std_clip = clip_scores.std()
cv_clip = std_clip / mean_clip

mean_rel = relative_scores.mean()
std_rel = relative_scores.std()
cv_rel = std_rel / mean_rel

# ------------------------------
# 2. 绘图（左右并列子图）
# ------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左侧：CLIP 绝对评分
n1, bins1, patches1 = ax1.hist(clip_scores, bins=30, density=True, alpha=0.6,
                               color='#2A5196', edgecolor='#372B8D', linewidth=0.5)
# 添加核密度估计
density1 = stats.gaussian_kde(clip_scores)
x1 = np.linspace(clip_scores.min(), clip_scores.max(), 200)
ax1.plot(x1, density1(x1), color='#98B5D3', linewidth=2, label='KDE')
ax1.set_xlabel('CLIP Absolute Score')
ax1.set_ylabel('Density')
ax1.set_title(f'CLIP Absolute Scores\nμ={mean_clip:.2f}, σ={std_clip:.2f}, CV={cv_clip:.3f}')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend()

# 右侧：TrueSkill 相对得分
n2, bins2, patches2 = ax2.hist(relative_scores, bins=30, density=True, alpha=0.6,
                               color='#98B5D3', edgecolor='#857BC0', linewidth=0.5)
density2 = stats.gaussian_kde(relative_scores)
x2 = np.linspace(relative_scores.min(), relative_scores.max(), 200)
ax2.plot(x2, density2(x2), color='#2A5196', linewidth=2, label='KDE')
ax2.set_xlabel('TrueSkill Relative Score')
ax2.set_ylabel('Density')
ax2.set_title(f'TrueSkill Relative Scores\nμ={mean_rel:.2f}, σ={std_rel:.2f}, CV={cv_rel:.3f}')
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend()

# 调整布局
plt.suptitle('Comparison of Score Distributions: Original vs. Revised Method', fontsize=14)
plt.tight_layout()

# ------------------------------
# 3. 保存图片
# ------------------------------
plt.savefig('score_distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('score_distribution_comparison.svg', bbox_inches='tight')
print("分布对比图已保存：score_distribution_comparison.png / .svg")
print(f"CLIP 绝对评分: 均值 {mean_clip:.3f}, 标准差 {std_clip:.3f}, 变异系数 {cv_clip:.3f}")
print(f"TrueSkill 相对得分: 均值 {mean_rel:.3f}, 标准差 {std_rel:.3f}, 变异系数 {cv_rel:.3f}")

# 可选：显示图形（交互环境）
# plt.show()