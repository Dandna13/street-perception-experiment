import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 加载合并后的数据 =================
# 确保当前目录下有 merged_analysis_data.csv
df = pd.read_csv('merged_analysis_data.csv')

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac
plt.rcParams['axes.unicode_minus'] = False

# ================= 2. 定义需要分析的“核心组合” =================
# 格式: (X轴_客观指标, Y轴_主观感知, 图表标题, 拟合线颜色)
# 这里精选了 4 组在城市空间中最具代表性的研究假设
analysis_pairs = [
    ('S_Enclosure', 'O_Score_Depressing', '空间围合度对【压抑感】的非线性影响', 'red'),
    ('S_GVI', 'O_Score_Comfort', '绿视率对【舒适度】的阈值效应', 'green'),
    ('S_Motorization', 'O_Score_Safety', '机动化侵占度对【安全感】的剥夺效应', 'orange'),
    ('S_Clutter', 'O_Score_Beauty', '视觉干扰度对【美观度】的破坏趋势', 'purple')
]

# ================= 3. 绘制阈值分析图面板 =================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (x_col, y_col, title, line_color) in enumerate(analysis_pairs):
    if x_col in df.columns and y_col in df.columns:
        ax = axes[i]

        # 绘制散点图与非线性拟合线 (order=2 代表二次多项式回归，捕捉曲线拐点)
        sns.regplot(
            data=df,
            x=x_col,
            y=y_col,
            ax=ax,
            scatter_kws={'alpha': 0.4, 's': 20, 'color': 'gray'},  # 底层散点
            line_kws={'color': line_color, 'linewidth': 3},  # 拟合曲线
            order=2  # 使用二次多项式探索非线性/边际效应
        )

        # 设置图表样式
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(f'客观指标：{x_col}', fontsize=12)
        ax.set_ylabel(f'主观得分：{y_col}', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

# 整体布局与保存
plt.suptitle('运城老城区 S-O 机制解析：关键物理指标的非线性感知阈值', fontsize=18, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])

output_img = 'threshold_analysis_curves.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"✅ 非线性阈值曲线已生成: {output_img}")
plt.show()
