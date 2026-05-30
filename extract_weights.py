import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ================= 1. 加载数据 =================
df = pd.read_csv('merged_analysis_data.csv')

# 获取 S层 和 O层 的列名
s_cols = [c for c in df.columns if c.startswith('S_')]
o_cols = [c for c in df.columns if c.startswith('O_Score_')]

# ================= 2. 训练模型并提取数值 =================
results = []

for target in o_cols:
    X = df[s_cols]
    y = df[target]

    # 训练随机森林模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # 获取 R² 精度
    r2 = model.score(X, y)

    # 获取 6 个物理指标的特征重要性 (权重)
    importances = model.feature_importances_

    # 记录到字典中
    # 将 O_Score_Safety 等转为中文，方便做表
    target_name_map = {
        'O_Score_Safety': '安全感 (Safety)',
        'O_Score_Comfort': '舒适度 (Comfort)',
        'O_Score_Beauty': '美观度 (Beauty)',
        'O_Score_Depressing': '压抑感 (Depressing)'
    }

    row_data = {
        '感知维度': target_name_map.get(target, target),
        '模型精度 (R²)': round(r2, 4)
    }

    # 将小数转化为百分比格式，方便论文阅读 (例如 0.2145 -> 21.45%)
    for i, col in enumerate(s_cols):
        row_data[col] = f"{importances[i] * 100:.2f}%"

    results.append(row_data)

# ================= 3. 输出为表格 =================
df_results = pd.DataFrame(results)

# 打印到终端 (Markdown 格式)
print("\n" + "=" * 50)
print("表 X：四项主观感知维度的随机森林回归精度与特征贡献度")
print("=" * 50)
print(df_results.to_markdown(index=False))
print("=" * 50 + "\n")

# 导出为 CSV，可以用 Excel 打开并复制进 Word
df_results.to_csv('rf_model_weights_summary.csv', index=False, encoding='utf-8-sig')
print("✅ 数值表格已成功导出为: rf_model_weights_summary.csv")
