"""
intervention_evaluation.py
依赖: 训练好的模型 rf_model.pkl，以及原始点数据 all_points_imputed_for_qgis.csv
如果需要模拟干预，使用 data_ready_for_modeling.csv 中低分路段并施加假设改善
实际应用时需加载干预后的特征文件 (intervention_features.csv)
"""

import os

os.environ['MPLBACKEND'] = 'Agg'

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import joblib

plt.rcParams['svg.fonttype'] = 'none'


def load_model():
    return joblib.load("rf_model.pkl")


def load_data():
    df = pd.read_csv("data_ready_for_modeling.csv")
    with open("feature_names.txt", "r") as f:
        feature_cols = [line.strip() for line in f.readlines()]
    return df, feature_cols


def generate_intervention_example(df, feature_cols):
    """模拟干预：选择原始得分低于3.0且连续的路段，将Delta和CV指标降低25%"""
    low_mask = df['O_Score_Rhythm'] < 3.0
    df_low = df[low_mask].copy()
    if len(df_low) == 0:
        print("没有低分路段，随机选取部分样本模拟")
        df_low = df.sample(frac=0.3, random_state=42).copy()
    df_intervention = df_low.copy()
    for c in feature_cols:
        df_intervention[c] = df_low[c] * 0.75  # 假设干预使变化幅度减小25%
    return df_low, df_intervention


def statistical_test(y_before, y_after):
    diff = y_after - y_before
    # 正态性检验
    if stats.shapiro(diff).pvalue > 0.05:
        stat, p = stats.ttest_rel(y_after, y_before)
        test_name = "Paired t-test"
    else:
        stat, p = stats.wilcoxon(y_after, y_before)
        test_name = "Wilcoxon signed-rank test"
    print(f"{test_name}: statistic = {stat:.3f}, p-value = {p:.4f}")
    return test_name, stat, p


def plot_intervention_boxplot(y_before, y_after):
    plt.figure(figsize=(7, 6))
    plt.boxplot([y_before, y_after], labels=['Before', 'After'], patch_artist=True)
    plt.ylabel('Predicted O_Score_Rhythm')
    plt.title('Intervention Effect on Street Sequence Perception')
    plt.tight_layout()
    plt.savefig('intervention_boxplot.svg', format='svg')
    plt.close()


def plot_radar_comparison(before_vals, after_vals, feature_cols, title="Delta指标对比"):
    import numpy as np
    N = len(feature_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, before_vals, 'o-', linewidth=2, label='Before')
    ax.plot(angles, after_vals, 'o-', linewidth=2, label='After')
    ax.fill(angles, before_vals, alpha=0.25)
    ax.fill(angles, after_vals, alpha=0.25)
    ax.set_xticks(angles[:-1])
    # 简化标签显示
    short_labels = [c.replace('Delta_', 'Δ').replace('CV_', 'CV')[:10] for c in feature_cols]
    ax.set_xticklabels(short_labels, size=8)
    ax.legend(loc='upper right')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('radar_delta_comparison.svg', format='svg')
    plt.close()


def export_spatial_predictions(df, feature_cols, model, points_file="all_points_imputed_for_qgis.csv"):
    """
    为每个采样点分配序列得分（序列对的后点）并保存供QGIS可视化
    """
    # 预测每个序列对的得分
    X = df[feature_cols]
    y_pred = model.predict(X)
    df_pred = df[['P1_id', 'P2_id']].copy()
    df_pred['pred_score'] = y_pred

    # 读取点数据
    points = pd.read_csv(points_file)
    # 合并：将预测得分赋给后点 (P2_id)
    points_with_score = points.merge(df_pred, left_on='id', right_on='P2_id', how='left')
    # 对于没有匹配的点（如起始点），用相邻插值或保留缺失，这里简单用全局均值填充
    points_with_score['pred_score'] = points_with_score['pred_score'].fillna(points_with_score['pred_score'].mean())
    # 输出经纬度 + 预测得分
    out = points_with_score[['lon', 'lat', 'pred_score']]
    out.to_csv("predicted_scores_for_qgis.csv", index=False)
    print("空间预测数据已导出至 predicted_scores_for_qgis.csv")


def main():
    # 加载模型和数据
    print("加载模型...")
    model = load_model()
    df, feature_cols = load_data()

    # 生成干预示例（实际应用中请替换为真实的干预数据）
    print("生成模拟干预数据...")
    df_before, df_after = generate_intervention_example(df, feature_cols)

    # 预测
    print("预测干预前后得分...")
    y_before = model.predict(df_before[feature_cols])
    y_after = model.predict(df_after[feature_cols])

    # 统计检验
    print("\n=== 统计检验 ===")
    statistical_test(y_before, y_after)

    # 箱线图
    print("绘制箱线图...")
    plot_intervention_boxplot(y_before, y_after)

    # 雷达图（选取第一个样本作为示例）
    print("绘制雷达图...")
    first_idx = 0
    before_vals = df_before.iloc[first_idx][feature_cols].values
    after_vals = df_after.iloc[first_idx][feature_cols].values
    plot_radar_comparison(before_vals, after_vals, feature_cols)

    # 导出空间数据
    print("导出空间预测数据...")
    export_spatial_predictions(df, feature_cols, model)

    # 导出干预前后对比表
    comp_df = pd.DataFrame({'before': y_before, 'after': y_after, 'improvement': y_after - y_before})
    comp_df.to_csv("intervention_comparison.csv", index=False)
    print("干预对比表已保存至 intervention_comparison.csv")

    # 输出改善统计
    improvement = np.mean(y_after - y_before)
    print(
        f"\n平均得分提升: {improvement:.4f} (95% CI: [{np.percentile(y_after - y_before, 2.5):.4f}, {np.percentile(y_after - y_before, 97.5):.4f}])")


if __name__ == "__main__":
    main()