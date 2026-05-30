import pandas as pd
import random
from trueskill import Rating, rate_1vs1, setup

# 1. 严格设定 TrueSkill 参数
# 初始 μ=25，σ=25/3，β=σ/2，τ=σ/100
sigma_init = 25.0 / 3.0
setup(mu=25.0, sigma=sigma_init, beta=sigma_init/2.0, tau=sigma_init/100.0, draw_probability=0.0)

# 2. 读取特征矩阵与 CLIP 分数
df_indices = pd.read_csv("sequence_indices_full.csv")
df_clip = pd.read_csv("clip_base_scores.csv")

# 融合数据：获取 P1 和 P2 的 base_score
df = df_indices.merge(df_clip, left_on='P1_id', right_on='id', how='left').rename(columns={'base_score': 's_i'})
df = df.merge(df_clip, left_on='P2_id', right_on='id', how='left').rename(columns={'base_score': 's_next'})

# 3. 核心质量评估公式 (惩罚系数 beta = 0.3)
def calc_quality(row):
    return (row['s_i'] + row['s_next']) / 2.0 - 0.3 * abs(row['s_i'] - row['s_next'])

df['quality_score'] = df.apply(calc_quality, axis=1)
sequences = df[['P1_id', 'P2_id', 'quality_score']].to_dict('records')

# 4. 模拟虚拟被试配对比较
print("开始模拟 18 名虚拟被试，生成 1620 条配对记录...")
simulated_records = []
for _ in range(18):       # 18 名虚拟被试
    for _ in range(90):   # 每人 90 组
        seq_i, seq_j = random.sample(sequences, 2)
        if seq_i['quality_score'] > seq_j['quality_score']:
            simulated_records.append((seq_i, seq_j))
        else:
            simulated_records.append((seq_j, seq_i))

# 5. TrueSkill 排序
ratings = {}
def get_r(sid):
    if sid not in ratings: ratings[sid] = Rating()
    return ratings[sid]

for win, lose in simulated_records:
    w_id, l_id = f"{win['P1_id']}_{win['P2_id']}", f"{lose['P1_id']}_{lose['P2_id']}"
    ratings[w_id], ratings[l_id] = rate_1vs1(get_r(w_id), get_r(l_id))

# 6. 线性归一化并输出
mu_vals = [r.mu for r in ratings.values()]
min_mu, max_mu = min(mu_vals), max(mu_vals)

res = []
for sid, rating in ratings.items():
    p1, p2 = sid.split('_')
    norm_score = ((rating.mu - min_mu) / (max_mu - min_mu)) * 10.0
    res.append({
        'P1_id': int(p1), 'P2_id': int(p2),
        'O_Score_Rhythm': round(norm_score, 4)
    })

df_final = pd.merge(df_indices, pd.DataFrame(res), on=['P1_id', 'P2_id'], how='inner')
df_final.to_csv("final_ml_dataset.csv", index=False)
print("✅ TrueSkill 排序完成！最终用于机器学习的宽表已保存为: final_ml_dataset.csv")