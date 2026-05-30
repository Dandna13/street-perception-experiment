"""
generate_relative_perception.py
使用 CLIP 成对比较 + TrueSkill 生成每个采样点的相对感知得分，
并计算序列对的感知得分。
适配图像命名格式：{id}_0.jpg 或 {id}_0.png
图像文件夹：street_views
"""

import os
import random
import pandas as pd
import numpy as np
import torch
import clip
from PIL import Image
import trueskill
from tqdm import tqdm

# ======================== 配置参数 ========================
IMAGE_DIR = "street_views"                 # 存放街景图像的文件夹，文件名格式 {id}_0.jpg 或 {id}_0.png
POINTS_CSV = "all_points_imputed_for_qgis.csv"   # 包含所有点 id 的文件
SEQUENCE_CSV = "final_ml_dataset.csv"       # 包含序列对 P1_id, P2_id 的文件
OUTPUT_POINT_SCORES = "point_relative_scores.csv"
OUTPUT_SEQUENCE_SCORES = "sequence_relative_scores.csv"
COMPARISONS_PER_POINT = 15                  # 每个点参与的比较次数
RANDOM_SEED = 42
TRUESKILL_INIT_MU = 25.0
TRUESKILL_INIT_SIGMA = 25.0 / 3
BETA = 0.3                                  # 可选，若想用带惩罚的公式
# ==========================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
trueskill.setup(mu=TRUESKILL_INIT_MU, sigma=TRUESKILL_INIT_SIGMA)

# 检测可用设备（如果没有 CUDA 则自动使用 CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 加载 CLIP 模型
model, preprocess = clip.load("ViT-B/32", device=device)
text = clip.tokenize(["a comfortable, safe and rhythmical street walking experience"]).to(device)

def get_image_similarity(image_path):
    """返回 CLIP 相似度（作为胜负判据）"""
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    except Exception as e:
        print(f"加载图像失败 {image_path}: {e}")
        return 0.0
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).squeeze().item()
    return similarity

def find_image_path(point_id, image_dir):
    """
    在目录中查找 {id}_0.jpg 或 {id}_0.png，返回完整路径，若不存在则返回 None
    """
    for ext in ['.jpg', '.png']:
        candidate = os.path.join(image_dir, f"{point_id}_0{ext}")
        if os.path.exists(candidate):
            return candidate
    return None

def main():
    # 1. 读取所有点 id
    points_df = pd.read_csv(POINTS_CSV)
    point_ids = points_df['id'].values
    n_points = len(point_ids)
    print(f"共 {n_points} 个采样点")

    # 2. 预计算所有图像的 CLIP 相似度（若已缓存则跳过）
    cache_file = "clip_similarity_cache.csv"
    if os.path.exists(cache_file):
        sim_df = pd.read_csv(cache_file)
        similarity = dict(zip(sim_df['id'], sim_df['similarity']))
        print("从缓存加载相似度")
    else:
        similarity = {}
        for pid in tqdm(point_ids, desc="计算 CLIP 相似度"):
            img_path = find_image_path(pid, IMAGE_DIR)
            if img_path:
                sim = get_image_similarity(img_path)
                similarity[pid] = sim
            else:
                print(f"警告: 图像 {pid}_0 不存在，使用 0.0")
                similarity[pid] = 0.0
        # 保存缓存
        pd.DataFrame([(k, v) for k, v in similarity.items()], columns=['id', 'similarity']).to_csv(cache_file, index=False)
        print("相似度缓存已保存")

    # 3. 生成成对比较记录 (winner, loser)
    comparisons = []
    for pid in tqdm(point_ids, desc="生成成对比较"):
        # 随机选择对手（排除自己）
        opponents = random.sample([p for p in point_ids if p != pid], COMPARISONS_PER_POINT)
        for op in opponents:
            sim_i = similarity[pid]
            sim_j = similarity[op]
            if sim_i > sim_j:
                comparisons.append({'winner': pid, 'loser': op})
            elif sim_i < sim_j:
                comparisons.append({'winner': op, 'loser': pid})
            # 相等则跳过
    print(f"共生成 {len(comparisons)} 条有效比较记录")

    # 4. TrueSkill 更新每个点的 Rating
    ratings = {pid: trueskill.Rating() for pid in point_ids}
    for comp in tqdm(comparisons, desc="TrueSkill 更新"):
        w_rating = ratings[comp['winner']]
        l_rating = ratings[comp['loser']]
        new_w, new_l = trueskill.rate_1vs1(w_rating, l_rating)
        ratings[comp['winner']] = new_w
        ratings[comp['loser']] = new_l

    # 5. 提取 mu 并归一化到 0-10
    mu_vals = np.array([ratings[pid].mu for pid in point_ids])
    min_mu, max_mu = mu_vals.min(), mu_vals.max()
    norm_scores = (mu_vals - min_mu) / (max_mu - min_mu) * 10.0

    point_result = pd.DataFrame({
        'id': point_ids,
        'trueskill_mu': mu_vals,
        'relative_score': norm_scores
    })
    point_result.to_csv(OUTPUT_POINT_SCORES, index=False)
    print(f"点的相对得分已保存至 {OUTPUT_POINT_SCORES}")
    print(f"Mu 范围: {min_mu:.3f} ~ {max_mu:.3f}")
    print(f"分数范围: {norm_scores.min():.3f} ~ {norm_scores.max():.3f}")

    # 6. 生成序列对的感知得分
    seq_df = pd.read_csv(SEQUENCE_CSV)
    # 建立 id -> relative_score 的映射
    score_map = dict(zip(point_result['id'], point_result['relative_score']))
    seq_df['score_P1'] = seq_df['P1_id'].map(score_map)
    seq_df['score_P2'] = seq_df['P2_id'].map(score_map)
    # 计算序列得分（均值法）
    seq_df['seq_score_mean'] = (seq_df['score_P1'] + seq_df['score_P2']) / 2.0
    # 可选：带惩罚的公式
    seq_df['seq_score_penalty'] = (seq_df['score_P1'] + seq_df['score_P2']) / 2.0 - BETA * np.abs(seq_df['score_P1'] - seq_df['score_P2'])

    out_seq = seq_df[['P1_id', 'P2_id', 'seq_score_mean', 'seq_score_penalty']]
    out_seq.to_csv(OUTPUT_SEQUENCE_SCORES, index=False)
    print(f"序列对得分已保存至 {OUTPUT_SEQUENCE_SCORES}")

    # 7. 将点的相对得分合并回 GIS 点文件（用于空间可视化）
    points_with_score = points_df.merge(point_result, on='id', how='left')
    points_with_score[['lon', 'lat', 'relative_score']].to_csv("points_with_relative_score.csv", index=False)
    print("带有相对得分的点文件已保存至 points_with_relative_score.csv")

if __name__ == "__main__":
    main()