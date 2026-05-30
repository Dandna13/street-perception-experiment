import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ------------------------------
# 1. 读取干预对比数据
# ------------------------------
df = pd.read_csv("intervention_comparison.csv")  # 应包含 'before', 'after' 列
before = df['before']
after  = df['after']

# 配对样本数
n_pairs = len(before)

# ------------------------------
# 2. 计算统计量（中位数、四分位距、异常值）
# ------------------------------
def box_stats(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    median = np.median(data)
    # 异常值：< Q1-1.5*IQR 或 > Q3+1.5*IQR
    outliers = data[(data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)]
    return median, iqr, outliers

med_before, iqr_before, out_before = box_stats(before)
med_after,  iqr_after,  out_after  = box_stats(after)

# ------------------------------
# 3. 绘图
# ------------------------------
fig, ax = plt.subplots(figsize=(8, 6))

# 箱线图位置
positions = [1, 2]
bp = ax.boxplot([before, after],
                positions=positions,
                widths=0.5,
                patch_artist=True,
                showmeans=False,
                showfliers=True,           # 显示异常值
                whiskerprops=dict(color='#333', linewidth=1),
                capprops=dict(color='#333', linewidth=1),
                medianprops=dict(color='black', linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))

# 设置箱体颜色（按指定配色）
colors = ['#2A5196', '#98B5D3']
edgecolors = ['#372B8D', '#857BC0']
for patch, col, edge in zip(bp['boxes'], colors, edgecolors):
    patch.set_facecolor(col)
    patch.set_edgecolor(edge)
    patch.set_linewidth(1.5)

# 绘制配对连线（每个样本 before -> after）
# 为了视觉清爽，设置低透明度，且仅随机抽样或全部绘制（602条线仍可接受）
for i in range(n_pairs):
    ax.plot([1, 2], [before.iloc[i], after.iloc[i]],
            color='lightgray', linewidth=0.5, alpha=0.3, solid_capstyle='round')

# 绘制散点（jitter 避免完全重叠）
# 在 x=1 处添加 jitter
jitter = 0.05
x_before = np.random.normal(1, jitter, n_pairs)
x_after  = np.random.normal(2, jitter, n_pairs)
ax.scatter(x_before, before, c='#2A5196', s=10, alpha=0.4, edgecolors='none')
ax.scatter(x_after,  after,  c='#98B5D3', s=10, alpha=0.4, edgecolors='none')

# 添加统计信息文本
info_text = (f"Before  N={n_pairs}\n"
             f"Med: {med_before:.2f}\n"
             f"IQR: {iqr_before:.2f}\n"
             f"Outliers: {len(out_before)}")
ax.text(0.65, ax.get_ylim()[1]*0.95, info_text,
        transform=ax.get_xaxis_transform(),
        verticalalignment='top', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

info_text2 = (f"After   N={n_pairs}\n"
              f"Med: {med_after:.2f}\n"
              f"IQR: {iqr_after:.2f}\n"
              f"Outliers: {len(out_after)}")
ax.text(1.85, ax.get_ylim()[1]*0.95, info_text2,
        transform=ax.get_xaxis_transform(),
        verticalalignment='top', fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))

# 装饰
ax.set_xticks([1, 2])
ax.set_xticklabels(['Before', 'After'])
ax.set_ylabel('Predicted O_Score_Rhythm')
ax.set_title('Intervention Effect on Low-Score Sequences\n(Paired Boxplot with Individual Trajectories)')
ax.grid(axis='y', linestyle='--', alpha=0.4)

# 添加图例（箱体颜色）
legend_patches = [mpatches.Patch(color=colors[0], label='Before'),
                  mpatches.Patch(color=colors[1], label='After')]
ax.legend(handles=legend_patches, loc='upper right')

plt.tight_layout()

# ------------------------------
# 4. 保存图片
# ------------------------------
plt.savefig('intervention_boxplot.png', dpi=300, bbox_inches='tight')
plt.savefig('intervention_boxplot.svg', bbox_inches='tight')
print("图片已保存为 intervention_boxplot.png 和 intervention_boxplot.svg")

# 若需在交互环境中显示，取消下一行注释
# plt.show()