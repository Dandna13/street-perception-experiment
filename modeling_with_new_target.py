"""
modeling_with_new_target.py
用途：使用新的目标变量（相对感知得分）训练随机森林与 XGBoost，
进行空间分组交叉验证，输出特征重要性、SHAP 图、PDP/ICE 图等。
所有图表保存为 SVG 格式。
"""

import os

os.environ['MPLBACKEND'] = 'Agg'

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.inspection import PartialDependenceDisplay
import shap
import warnings
import joblib
from scipy import stats

warnings.filterwarnings('ignore')

# 设置绘图风格
plt.rcParams['svg.fonttype'] = 'none'
sns.set_style("whitegrid")


def load_data():
    df = pd.read_csv("final_ml_dataset_new_target.csv")
    with open("feature_names_new.txt", "r") as f:
        feature_cols = [line.strip() for line in f.readlines()]
    target_col = "O_Score_Rhythm"
    X = df[feature_cols]
    y = df[target_col]
    groups = df['street_group']
    return X, y, groups, feature_cols


def evaluate_model(model, X, y, groups, model_name):
    gkf = GroupKFold(n_splits=10)
    r2_scores = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring='r2')
    mae_scores = -cross_val_score(model, X, y, cv=gkf, groups=groups, scoring='neg_mean_absolute_error')
    print(
        f"{model_name}: R2 = {r2_scores.mean():.3f} ± {r2_scores.std():.3f}, MAE = {mae_scores.mean():.3f} ± {mae_scores.std():.3f}")
    return r2_scores.mean(), mae_scores.mean()


def plot_feature_importance(model, feature_cols, model_name):
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values('importance',
                                                                                            ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df.head(10), x='importance', y='feature', palette='viridis')
    plt.title(f'{model_name} - Top 10 Feature Importance')
    plt.tight_layout()
    plt.savefig(f'{model_name}_feature_importance.svg', format='svg')
    plt.close()
    print(f"  特征重要性图已保存: {model_name}_feature_importance.svg")
    return imp_df


def shap_analysis(model, X, feature_cols, model_name):
    print("  正在计算 SHAP 值...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.tight_layout()
    plt.savefig(f'{model_name}_shap_summary.svg', format='svg')
    plt.close()
    print("  SHAP summary 已保存")
    # 依赖图 (前3重要特征)
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_}).sort_values('importance',
                                                                                                           ascending=False)
    top3 = imp_df.head(3)['feature'].tolist()
    for feat in top3:
        plt.figure()
        shap.dependence_plot(feat, shap_values, X, feature_names=feature_cols, show=False)
        plt.tight_layout()
        plt.savefig(f'{model_name}_shap_dependence_{feat}.svg', format='svg')
        plt.close()
        print(f"  SHAP 依赖图 {feat} 已保存")
    return top3


def plot_pdp_ice(model, X, feature_cols, top3_features):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, feat in enumerate(top3_features):
        PartialDependenceDisplay.from_estimator(
            model, X, [feat], kind='both', ax=axes[i],
            ice_lines_kw={'color': 'lightblue', 'alpha': 0.3},
            pd_line_kw={'color': 'red', 'linewidth': 2}
        )
        axes[i].set_title(f'PDP & ICE for {feat}')
    plt.tight_layout()
    plt.savefig('pdp_ice_top3.svg', format='svg')
    plt.close()
    print("  PDP+ICE 图已保存")


def plot_pred_vs_true(y_true, y_pred, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True O_Score_Rhythm')
    plt.ylabel('Predicted')
    plt.title(f'{model_name} - Prediction vs Truth')
    plt.tight_layout()
    plt.savefig(f'{model_name}_pred_vs_true.svg', format='svg')
    plt.close()
    print(f"  预测散点图已保存")


def diagnostic_analysis(df, feature_cols, target_col):
    """诊断目标变量分布及特征相关性"""
    print("\n=== 诊断分析 ===")
    print("目标变量 O_Score_Rhythm 统计:")
    print(df[target_col].describe())
    # 直方图
    plt.figure(figsize=(6, 4))
    sns.histplot(df[target_col], kde=True)
    plt.title('Distribution of O_Score_Rhythm')
    plt.tight_layout()
    plt.savefig('target_distribution_new.svg', format='svg')
    plt.close()
    print("目标变量分布图已保存: target_distribution_new.svg")

    # 特征与目标的相关性
    corr = df[feature_cols + [target_col]].corr()[target_col].sort_values(ascending=False)
    print("\n特征与目标变量的相关系数:")
    print(corr)
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    corr_without_target = corr.drop(target_col)
    sns.barplot(x=corr_without_target.values, y=corr_without_target.index)
    plt.title('Feature Correlation with O_Score_Rhythm')
    plt.xlabel('Pearson Correlation')
    plt.tight_layout()
    plt.savefig('feature_target_correlation_new.svg', format='svg')
    plt.close()
    print("相关系数图已保存: feature_target_correlation_new.svg")

    unique_ratio = df[target_col].nunique() / len(df)
    print(f"目标变量唯一值比例: {unique_ratio:.2%}")


def main():
    print("加载数据...")
    X, y, groups, feature_cols = load_data()
    df = pd.read_csv("final_ml_dataset_new_target.csv")
    print(f"数据形状: X {X.shape}, y {y.shape}, 分组数 {groups.nunique()}")

    # 诊断分析
    diagnostic_analysis(df, feature_cols, "O_Score_Rhythm")

    # 初始化模型
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)

    print("\n交叉验证评估（空间分组）:")
    rf_r2, rf_mae = evaluate_model(rf, X, y, groups, "RandomForest")
    xgb_r2, xgb_mae = evaluate_model(xgb, X, y, groups, "XGBoost")

    print("\n训练全量随机森林...")
    rf.fit(X, y)
    xgb.fit(X, y)

    print("\n绘制特征重要性图...")
    plot_feature_importance(rf, feature_cols, "RandomForest")

    print("\n开始 SHAP 分析...")
    top3_features = shap_analysis(rf, X, feature_cols, "RandomForest")

    print("\n绘制 PDP+ICE 图...")
    plot_pdp_ice(rf, X, feature_cols, top3_features)

    print("\n绘制预测 vs 真实图...")
    y_pred_rf = rf.predict(X)
    plot_pred_vs_true(y, y_pred_rf, "RandomForest")

    print("\n保存模型...")
    joblib.dump(rf, "rf_model_new.pkl")
    joblib.dump(xgb, "xgb_model_new.pkl")
    print("全部完成！模型已保存为 rf_model_new.pkl 和 xgb_model_new.pkl")

    # 输出最终性能摘要
    print("\n=== 最终建模性能 ===")
    print(f"随机森林: R2 = {rf_r2:.3f}, MAE = {rf_mae:.3f}")
    print(f"XGBoost:   R2 = {xgb_r2:.3f}, MAE = {xgb_mae:.3f}")


if __name__ == "__main__":
    main()