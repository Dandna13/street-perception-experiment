"""
intervention_simulation.py
模拟干预：对低分路段的设计优化，通过修改 Δ 和 CV 特征值并重新预测得分，
比较干预前后的统计显著性，输出雷达图和箱线图。
实际应用中应根据真实干预重建后的分割结果计算新的特征矩阵。
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['svg.fonttype'] = 'none'
sns.set_style("whitegrid")

def load_model_and_low_segments():
    model = joblib.load("rf_model_new.pkl")
    df = pd.read_csv("final_ml_dataset_new_target.csv")
    with open("feature_names_new.txt", "r") as f:
        feature_cols = [line.strip() for line in f.readlines()]
    # 直接使用评分<3.0的序列对作为干预对象
    low_seg = df[df['O_Score_Rhythm'] < 3.0].copy()
    print(f"共 {len(low_seg)} 个低分序列对参与干预模拟")
    return model, df, feature_cols, low_seg

def simulate_intervention(df_low, feature_cols, reduction_factor=0.7):
    """
    仅对特征列进行缩放，其他列（如 P1_id, P2_id, Distance_m 等）保持不变。
    返回特征矩阵（仅特征列）的干预后版本。
    """
    df_intervention = df_low.copy()
    for col in feature_cols:
        df_intervention[col] = df_low[col] * reduction_factor
    # 只返回特征列
    return df_intervention[feature_cols]

def statistical_test(y_before, y_after):
    diff = y_after - y_before
    if len(diff) < 3:
        print("样本量过少，无法进行正态性检验，执行 Wilcoxon 符号秩检验")
        stat, p = stats.wilcoxon(y_after, y_before)
        test_name = "Wilcoxon signed-rank test"
    else:
        if stats.shapiro(diff).pvalue > 0.05:
            stat, p = stats.ttest_rel(y_after, y_before)
            test_name = "Paired t-test"
        else:
            stat, p = stats.wilcoxon(y_after, y_before)
            test_name = "Wilcoxon signed-rank test"
    print(f"{test_name}: statistic = {stat:.3f}, p-value = {p:.4f}")
    return test_name, stat, p

def plot_intervention_boxplot(y_before, y_after):
    plt.figure(figsize=(7,6))
    plt.boxplot([y_before, y_after], labels=['Before', 'After'], patch_artist=True)
    plt.ylabel('Predicted O_Score_Rhythm')
    plt.title('Intervention Effect on Low-Score Sequences')
    plt.tight_layout()
    plt.savefig('intervention_boxplot.svg', format='svg')
    plt.close()
    print("箱线图已保存: intervention_boxplot.svg")

def plot_radar_comparison(before_vals, after_vals, feature_cols, title="Δ指标干预前后对比"):
    import numpy as np
    N = len(feature_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # 首尾闭合
    angles += angles[:1]
    # 同样将数据首尾闭合
    before_vals_closed = np.append(before_vals, before_vals[0])
    after_vals_closed = np.append(after_vals, after_vals[0])

    fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
    ax.plot(angles, before_vals_closed, 'o-', linewidth=2, label='Before')
    ax.plot(angles, after_vals_closed, 'o-', linewidth=2, label='After')
    ax.fill(angles, before_vals_closed, alpha=0.25)
    ax.fill(angles, after_vals_closed, alpha=0.25)
    ax.set_xticks(angles[:-1])
    short_labels = [c.replace('Delta_', 'Δ').replace('CV_', 'CV')[:12] for c in feature_cols]
    ax.set_xticklabels(short_labels, size=8, rotation=45)
    ax.legend(loc='upper right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('radar_delta_comparison.svg', format='svg')
    plt.close()
    print("雷达图已保存: radar_delta_comparison.svg")

def main():
    print("加载模型和低分数据...")
    model, df, feature_cols, low_seg = load_model_and_low_segments()

    if len(low_seg) == 0:
        print("没有需要干预的低分序列，程序退出。")
        return

    # 干预前的特征矩阵和预测
    X_before = low_seg[feature_cols]
    y_before = model.predict(X_before)

    # 模拟干预（可根据实际设计调整 reduction_factor）
    reduction_factor = 0.7   # 使 Δ 和 CV 减少30%，模拟“韵律化”干预的效果
    X_after = simulate_intervention(low_seg, feature_cols, reduction_factor)
    y_after = model.predict(X_after)

    # 统计检验
    print("\n=== 干预效果统计检验 ===")
    statistical_test(y_before, y_after)

    # 箱线图
    plot_intervention_boxplot(y_before, y_after)

    # 雷达图（选取第一个样本作为示例）
    first_idx = 0
    before_vals = X_before.iloc[first_idx].values
    after_vals = X_after.iloc[first_idx].values
    plot_radar_comparison(before_vals, after_vals, feature_cols)

    # 输出改善摘要
    improvement = y_after - y_before
    print(f"\n平均得分提升: {improvement.mean():.4f} (95% CI: [{np.percentile(improvement,2.5):.4f}, {np.percentile(improvement,97.5):.4f}])")

    # 保存干预前后对比表
    comp_df = pd.DataFrame({'before': y_before, 'after': y_after, 'improvement': improvement})
    comp_df.to_csv("intervention_comparison.csv", index=False)
    print("干预对比表已保存至 intervention_comparison.csv")

if __name__ == "__main__":
    main()