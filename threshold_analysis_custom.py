import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 加载数据 =================
df = pd.read_csv('merged_analysis_data.csv')

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac 系统
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 定义您关注的 3 组核心关系 =================
# 格式: (X轴_客观指标, Y轴_主观感知, 图表标题, 拟合线颜色)
analysis_pairs = [
    ('S_Enclosure', 'O_Score_Depressing', '空间围合度 vs 压抑感\n(实体边界的压迫极限)', '#d62728'),  # 红色
    ('S_SVF', 'O_Score_Depressing', '天空开阔度 vs 压抑感\n(顶面视角的释放效应)', '#1f77b4'),  # 蓝色
    ('S_Motorization', 'O_Score_Comfort', '机动化侵占度 vs 舒适度\n(车行空间对人本体验的剥夺)', '#ff7f0e')  # 橙色
]

# ================= 3. 绘制 1x3 面板图 =================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes = axes.flatten()

for i, (x_col, y_col, title, line_color) in enumerate(analysis_pairs):
    if x_col in df.columns and y_col in df.columns:
        ax = axes[i]

        # 绘制散点图与非线性拟合线 (order=2 代表二次多项式回归)
        # 散点颜色设为浅灰，突出红/蓝/橙色的趋势线
        sns.regplot(
            data=df,
            x=x_col,
            y=y_col,
            ax=ax,
            scatter_kws={'alpha': 0.3, 's': 15, 'color': 'gray', 'edgecolors': 'none'},
            line_kws={'color': line_color, 'linewidth': 3},
            order=2
        )

        # 设置图表样式
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel(f'{x_col}', fontsize=11)
        ax.set_ylabel(f'{y_col}', fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)

        # 优化坐标轴显示，避免数据挤在一起
        ax.set_xlim(df[x_col].min(), df[x_col].max())

# 整体布局与保存
plt.suptitle('运城老城区核心空间形态指标的感知阈值分析', fontsize=16, y=1.05)
plt.tight_layout()

output_img = 'threshold_custom_analysis.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"✅ 定制阈值曲线已生成: {output_img}")
plt.show()
