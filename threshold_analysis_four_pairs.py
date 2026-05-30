import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 加载数据 =================
df = pd.read_csv('merged_analysis_data.csv')

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac 系统
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 定义核心分析组合 (4组) =================
# 格式: (X轴_客观指标, Y轴_主观感知, 图表主标题, Y轴标签, 拟合线颜色)
analysis_pairs = [
    ('S_Enclosure', 'O_Score_Depressing', '空间围合度对【压抑感】的非线性剥夺', '压抑感 (Depressing)', '#d62728'),  # 红色
    ('S_GVI', 'O_Score_Comfort', '绿视率对【舒适度】的边际递减效应', '舒适度 (Comfort)', '#2ca02c'),  # 绿色
    ('S_Motorization', 'O_Score_Safety', '机动化侵占对【安全感】的线性威胁', '安全感 (Safety)', '#ff7f0e'),  # 橙色
    ('S_Clutter', 'O_Score_Beauty', '视觉干扰对【美观度】的破坏阈值', '美观度 (Beauty)', '#9467bd')  # 紫色
]

# ================= 3. 绘制 2x2 面板图 =================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (x_col, y_col, title, y_label, line_color) in enumerate(analysis_pairs):
    if x_col in df.columns and y_col in df.columns:
        ax = axes[i]

        # 绘制散点图与非线性拟合线 (order=2 代表二次多项式回归)
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
        ax.set_xlabel(f'客观指标：{x_col}', fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)

        # 优化坐标轴显示范围，紧贴数据分布
        ax.set_xlim(df[x_col].min(), df[x_col].max())
        ax.set_ylim(max(0, df[y_col].min() - 0.5), min(10, df[y_col].max() + 0.5))

# 整体布局与保存
plt.suptitle('运城老城区 S-O 机制解析：四大核心物理指标的主观感知阈值特征', fontsize=18, y=1.02)
plt.tight_layout()

output_img = 'threshold_analysis_four_pairs.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"✅ 四组核心阈值曲线已生成: {output_img}")
plt.show()
