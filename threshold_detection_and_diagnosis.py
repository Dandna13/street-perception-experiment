"""
threshold_detection_and_diagnosis.py
简化版：不做自动阈值检测（避免版本兼容问题），只进行空间诊断和输出低分路段。
阈值可通过观察 PDP 图手动判断。
"""

import os

os.environ['MPLBACKEND'] = 'Agg'

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
import joblib
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['svg.fonttype'] = 'none'
sns.set_style("whitegrid")


def load_model_and_data():
    model = joblib.load("rf_model_new.pkl")
    df = pd.read_csv("final_ml_dataset_new_target.csv")
    with open("feature_names_new.txt", "r") as f:
        feature_cols = [line.strip() for line in f.readlines()]
    X = df[feature_cols]
    y = df['O_Score_Rhythm']
    return model, df, X, y, feature_cols


def plot_pdp_ice_for_top_features(model, X, feature_cols, top_n=3):
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance',
                                                                                            ascending=False)
    top_features = imp_df.head(top_n)['feature'].tolist()
    fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 4))
    if top_n == 1:
        axes = [axes]
    for i, feat in enumerate(top_features):
        PartialDependenceDisplay.from_estimator(
            model, X, [feat], kind='both', ax=axes[i],
            ice_lines_kw={'color': 'lightblue', 'alpha': 0.3},
            pd_line_kw={'color': 'red', 'linewidth': 2}
        )
        axes[i].set_title(f'PDP & ICE for {feat}')
    plt.tight_layout()
    plt.savefig('pdp_ice_top3_new.svg', format='svg')
    plt.close()
    print("PDP+ICE 图已保存: pdp_ice_top3_new.svg")
    return top_features


def identify_low_score_segments(df, score_col='O_Score_Rhythm', threshold=3.0, min_length_m=150):
    low_mask = df[score_col] < threshold
    segments = []
    current_start = None
    current_length = 0
    current_scores = []
    for idx, row in df.iterrows():
        if low_mask[idx]:
            if current_start is None:
                current_start = idx
            current_length += row.get('Distance_m', 30.0)  # 默认间隔30米
            current_scores.append(row[score_col])
        else:
            if current_start is not None and current_length >= min_length_m:
                segments.append({
                    'start_idx': current_start,
                    'end_idx': idx - 1,
                    'length_m': current_length,
                    'mean_score': np.mean(current_scores)
                })
            current_start = None
            current_length = 0
            current_scores = []
    if current_start is not None and current_length >= min_length_m:
        segments.append({
            'start_idx': current_start,
            'end_idx': df.index[-1],
            'length_m': current_length,
            'mean_score': np.mean(current_scores)
        })
    seg_df = pd.DataFrame(segments)
    if len(seg_df) > 0:
        print(f"识别到 {len(seg_df)} 个连续的劣质路段（评分<{threshold}，长度≥{min_length_m}m）")
    else:
        print(f"未发现符合条件的劣质路段")
    return seg_df


def export_spatial_diagnosis(df, points_file="all_points_imputed_for_qgis.csv", output_file="spatial_diagnosis.csv"):
    """
    将序列得分赋给采样点（后点），并输出用于GIS可视化的文件。
    需要 points_file 包含 'id', 'lon', 'lat' 列。
    """
    points = pd.read_csv(points_file)
    if 'id' not in points.columns:
        raise KeyError("点文件必须包含 'id' 列，请检查文件格式。")
    # 将序列得分映射到后点
    seq_scores = df[['P2_id', 'O_Score_Rhythm']].copy()
    seq_scores = seq_scores.rename(columns={'P2_id': 'id', 'O_Score_Rhythm': 'seq_score'})
    points_with_seq = points.merge(seq_scores, on='id', how='left')
    # 缺失的点（如起点）用全局均值填充
    mean_score = points_with_seq['seq_score'].mean()
    points_with_seq['seq_score'] = points_with_seq['seq_score'].fillna(mean_score)
    points_with_seq['is_low'] = (points_with_seq['seq_score'] < 3.0).astype(int)
    out_cols = ['lon', 'lat', 'seq_score', 'is_low']
    points_with_seq[out_cols].to_csv(output_file, index=False)
    print(f"空间诊断数据已保存至 {output_file}")
    return points_with_seq


def main():
    print("加载模型和数据...")
    model, df, X, y, feature_cols = load_model_and_data()

    print("\n=== 绘制前3个重要特征的 PDP+ICE 图 ===")
    top_features = plot_pdp_ice_for_top_features(model, X, feature_cols, top_n=3)

    print("\n=== 识别低分路段 ===")
    low_segments = identify_low_score_segments(df, score_col='O_Score_Rhythm', threshold=3.0, min_length_m=150)
    if len(low_segments) > 0:
        low_segments.to_csv("low_score_segments.csv", index=False)
        print("低分路段信息已保存至 low_score_segments.csv")

    print("\n=== 导出GIS可视化数据 ===")
    try:
        export_spatial_diagnosis(df, points_file="all_points_imputed_for_qgis.csv", output_file="spatial_diagnosis.csv")
    except FileNotFoundError:
        print("未找到 all_points_imputed_for_qgis.csv，请确认文件存在。")
    except KeyError as e:
        print(f"点文件列名错误: {e}")

    # 输出特征重要性排序
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance',
                                                                                            ascending=False)
    print("\n=== 特征重要性前5名 ===")
    print(imp_df.head(5))

    print("\n诊断完成！请观察 pdp_ice_top3_new.svg 中的曲线，手动判断阈值。")


if __name__ == "__main__":
    main()