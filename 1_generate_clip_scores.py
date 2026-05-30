import os
import torch
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import warnings

warnings.filterwarnings("ignore")

IMAGE_FOLDER = "street_views"
OUTPUT_FILE = "clip_base_scores.csv"
TOTAL_POINTS = 1272

print("加载模型: CLIP ViT-B/32 ...")
# 严格按照文档要求加载 ViT-B/32 模型
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)

# 严格遵循文档设定的正向评价 Prompt
positive_prompt = "a comfortable, safe and rhythmical street walking experience"
# 引入对立 prompt 以完成 Softmax 归一化计算
negative_prompt = "an uncomfortable, dangerous and monotonous street walking experience"
prompts = [positive_prompt, negative_prompt]

results = []
print(f"开始计算 {TOTAL_POINTS} 个采样点的基础感知分 (base_score)...")

for pid in range(TOTAL_POINTS):
    img_name = f"{pid}_0.jpg"  # 提取前进方向（0°）街景图像
    img_path = os.path.join(IMAGE_FOLDER, img_name)

    if os.path.exists(img_path):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                outputs = model(**inputs)

            # 计算余弦相似度并经 softmax 归一化至 [0,1] 区间
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).numpy()

            # 提取正向提示词的概率，并线性拉伸至 0–10 分
            base_score = round(float(probs[0][0]) * 10.0, 4)
        except Exception:
            base_score = 5.0
    else:
        base_score = 5.0  # 无图点位赋予 5.0 分中立值

    results.append({'id': pid, 'base_score': base_score})

    if pid > 0 and pid % 200 == 0:
        print(f"进度: {pid} / {TOTAL_POINTS}")

df_clip = pd.DataFrame(results)
df_clip.to_csv(OUTPUT_FILE, index=False)
print(f"单点打分完成，已保存至 {OUTPUT_FILE}")