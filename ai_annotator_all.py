import os
import torch
import random
import csv
import time
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ================= 配置区域 =================
IMAGE_FOLDER = 'images'
OUTPUT_CSV = 'comparisons.csv'
COMPARISONS_PER_ASPECT = 500  # 每个维度比较次数 (总共生成 2000 条)

# 【核心修改】：使用英文 Key，避免 CSV 乱码
ASPECTS_PROMPTS = {
    "Safety": ["A photo of a safe street", "A photo of a dangerous street"],
    "Comfort": ["A photo of a comfortable street environment", "A photo of a depressing street environment"],
    "Beauty": ["A photo of a beautiful and clean street", "A photo of a dirty and messy street"],
    "Depressing": ["A photo of a depressing and enclosed street", "A photo of an open and lively street"]
}

# ================= 模型加载 =================
print("正在加载 CLIP 模型...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# ================= 1. 计算所有图片的“基础得分” =================
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
all_scores = {img: {} for img in image_files}

print(f"正在分析 {len(image_files)} 张图片...")

# 预计算 Text Embeddings
text_inputs = {}
for aspect, prompts in ASPECTS_PROMPTS.items():
    text_inputs[aspect] = processor(text=prompts, return_tensors="pt", padding=True).to(device)

# 逐张图片推理
for img_file in tqdm(image_files):
    try:
        image_path = os.path.join(IMAGE_FOLDER, img_file)
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

            for aspect, t_input in text_inputs.items():
                text_features = model.get_text_features(**t_input)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

                logit_scale = model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()
                probs = logits.softmax(dim=1)

                # 获取正向提示词得分
                all_scores[img_file][aspect] = probs[0][0].item()

    except Exception as e:
        print(f"Skipping {img_file}: {e}")

# ================= 2. 生成 TrueSkill 比较数据 =================
new_rows = []
timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

print(f"正在生成比较数据...")

for aspect in ASPECTS_PROMPTS.keys():
    for _ in range(COMPARISONS_PER_ASPECT):
        img_a, img_b = random.sample(image_files, 2)

        score_a = all_scores[img_a].get(aspect, 0.5)
        score_b = all_scores[img_b].get(aspect, 0.5)

        # 判定胜负
        if score_a > score_b:
            winner, loser = img_a, img_b
        else:
            winner, loser = img_b, img_a

        new_rows.append([winner, loser, aspect, timestamp])

# ================= 3. 保存结果 =================
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["Winner_Image", "Loser_Image", "Aspect", "Timestamp"])
    writer.writerows(new_rows)

print(f"✅ 完成！已生成 {len(new_rows)} 条干净的数据 (无乱码)，保存至 {OUTPUT_CSV}")
