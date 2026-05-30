import pandas as pd
import trueskill
import csv
import math

INPUT_CSV = 'comparisons.csv'
OUTPUT_SCORES = 'perception_scores_final.csv'


def compute_trueskill_multi_aspect(data_path):
    # 读取数据
    df_comp = pd.read_csv(data_path)

    # 获取所有唯一的维度 (Safety, Comfort, Beauty...)
    aspects = df_comp['Aspect'].unique()
    print(f"检测到维度: {aspects}")

    # 存储最终结果的字典: {image_name: {aspect1: score, aspect2: score...}}
    final_scores = {}

    # === 对每个维度单独计算 TrueSkill ===
    for aspect in aspects:
        print(f"正在计算维度: {aspect} ...")

        env = trueskill.TrueSkill(mu=25.0, sigma=8.333)
        ratings = {}  # 当前维度的评分表

        # 筛选出当前维度的比赛记录
        subset = df_comp[df_comp['Aspect'] == aspect]

        for _, row in subset.iterrows():
            w_img, l_img = row['Winner_Image'], row['Loser_Image']

            if w_img not in ratings: ratings[w_img] = env.create_rating()
            if l_img not in ratings: ratings[l_img] = env.create_rating()

            ratings[w_img], ratings[l_img] = env.rate_1vs1(ratings[w_img], ratings[l_img])

        # 归一化该维度的分数到 0-10
        mus = [r.mu for r in ratings.values()]
        if not mus: continue
        min_mu, max_mu = min(mus), max(mus)

        for img, rating in ratings.items():
            norm_score = 10 * (rating.mu - min_mu) / (max_mu - min_mu + 1e-5)

            if img not in final_scores:
                final_scores[img] = {}

            # 保存为 "O_Score_Comfort" 这样的格式
            col_name = f"O_Score_{aspect}"
            final_scores[img][col_name] = round(norm_score, 2)

    # === 转换为 DataFrame 并整理格式 ===
    print("正在整理最终表格...")
    results = []
    for img, scores in final_scores.items():
        # 解析 ID
        try:
            basename = img.rsplit('.', 1)[0]
            parts = basename.split('_')
            point_id = parts[0]
            heading = parts[1]
        except:
            point_id = 0
            heading = 0

        row = {
            'point_id': point_id,
            'heading': heading,
            'filename': img
        }
        row.update(scores)  # 把各个维度的分数加进去
        results.append(row)

    df_final = pd.DataFrame(results)

    # 排序
    try:
        df_final['point_id'] = df_final['point_id'].astype(int)
        df_final = df_final.sort_values(by=['point_id', 'heading'])
    except:
        pass

    df_final.to_csv(OUTPUT_SCORES, index=False)
    print(f"✅ 计算完成！所有维度的得分已合并保存至 {OUTPUT_SCORES}")
    print(df_final.head())


if __name__ == "__main__":
    compute_trueskill_multi_aspect(INPUT_CSV)
