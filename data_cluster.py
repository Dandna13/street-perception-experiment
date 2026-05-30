import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示（防止绘图乱码，若报错可注释掉）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 加载数据 =================
df_s = pd.read_csv('s_layer_indicators.csv')
df_o = pd.read_csv('perception_scores_final.csv')

# ================= 2. 数据聚合 (按采样点取均值) =================
print("正在按采样点聚合数据...")
# S层：计算每个点 4 个朝向的物理指标均值
s_cols = [c for c in df_s.columns if c.startswith('S_')]
df_s_agg = df_s.groupby('point_id')[s_cols].mean().reset_index()

# O层：计算每个点 4 个朝向的感知分均值
o_cols = [c for c in df_o.columns if c.startswith('O_Score_')]
df_o_agg = df_o.groupby('point_id')[o_cols].mean().reset_index()

# ================= 3. 合并表格 =================
df_final = pd.merge(df_s_agg, df_o_agg, on='point_id', how='inner')
# 处理空值（如果有采样点缺失感知分，直接剔除）
df_final = df_final.dropna()
df_final.to_csv('merged_analysis_data.csv', index=False)
print(f"数据合并完成！共计 {len(df_final)} 个有效采样点。")

# ================= 4. 训练模型与解析机制 =================
target_aspects = [c for c in df_final.columns if c.startswith('O_Score_')]
features = s_cols  # 使用所有 S 层指标作为输入

for target in target_aspects:
    print(f"\n--- 正在分析维度: {target} ---")
    X = df_final[features]
    y = df_final[target]

    # 训练随机森林
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 计算 R² 分数 (模型解释力)
    r2 = model.score(X, y)
    print(f"模型的 R² 解释力: {r2:.4f}")

    # 获取特征重要性
    importances = model.feature_importances_
    feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

    # 绘图：特征重要性 (Feature Importance)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
    plt.title(f'特征重要性分析: {target}\n(R² = {r2:.2f})', fontsize=14)
    plt.xlabel('贡献度 (Importance Score)')
    plt.ylabel('S层物理指标')
    plt.tight_layout()

    # 保存图片 (用于论文图表)
    plt.savefig(f'importance_{target}.png', dpi=300)
    print(f"✅ 已生成并保存特征重要性图: importance_{target}.png")

# ================= 5. 生成阈值曲线 (示例：GVI vs Comfort) =================
# 自动寻找对 Comfort 影响最大的指标
if 'O_Score_Comfort' in df_final.columns:
    top_feature = feat_imp.index[0]  # 比如是 S_GVI
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df_final, x=top_feature, y='O_Score_Comfort',
                order=2, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title(f'非线性阈值关系: {top_feature} vs 舒适度', fontsize=12)
    plt.savefig('threshold_curve.png', dpi=300)
    print(f"✅ 已生成非线性拟合图: threshold_curve.png")

print("\n🚀 所有分析图表已生成，请在文件夹中查看 .png 文件。")
