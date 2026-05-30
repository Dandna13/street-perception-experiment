import os
import torch
import pandas as pd
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
IMAGE_FOLDER = 'images'          # 图片文件夹路径
OUTPUT_CSV = 's_layer_indicators.csv'  # 输出文件路径

# ================= ADE20K 类别映射逻辑 =================
# ADE20K 有 150 类 (0-149)，我们将相关类别 ID 映射到您的 6 大指标逻辑中
# 参考: ADE20K Index (0-based typically for HuggingFace models)

# 1. 绿视率 (GVI) 组件: 树(4), 草地(9), 植物(17), 棕榈树(72), 花(66)
IDX_GREEN = [4, 9, 17, 66, 72]

# 2. 天空开阔度 (SVF) 组件: 天空(2)
IDX_SKY = [2]

# 3. 空间围合度 (Enclosure) 组件:
# 墙(0), 建筑(1), 房子(25), 栅栏(32), 摩天大楼(48), 栏杆(84), 遮阳棚(68)
IDX_ENCLOSURE = [0, 1, 25, 32, 48, 68, 84]

# 4. 机动化侵占度 (Motorization) 组件:
# 道路(6), 汽车(20), 公交车(80), 卡车(83), 货车(102)
# *注意：论文公式包含 Road，故此处也包含 IDX_ROAD
IDX_ROAD = [6]
IDX_VEHICLE = [20, 80, 83, 102]
IDX_MOTORIZATION = IDX_ROAD + IDX_VEHICLE

# 5. 步行友好 (Walkability) 组件: 人行道(11), 小径(52)
IDX_SIDEWALK = [11, 52]

# 6. 视觉干扰度 (Clutter) 组件:
# 红绿灯(18), 路灯(19), 广告牌/路牌(21), 电线杆(81), 垃圾桶(87), 摩托车(116), 自行车(127)
# *ADE20K 能识别更多干扰物，这里比 Cityscapes 更全面
IDX_CLUTTER = [18, 19, 21, 81, 87, 116, 127]

# ================= 模型加载 =================
print("正在加载 SegFormer (ADE20K) 模型...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 使用 ADE20K 权重的模型
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model.to(device)

# ================= 处理逻辑 =================
results = []
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(('.jpg', '.png', '.jpeg'))]
print(f"找到 {len(image_files)} 张图片，开始处理...")

for img_file in tqdm(image_files):
    try:
        # 1. 解析文件名 {id}_{heading}.jpg
        basename = os.path.splitext(img_file)[0]
        parts = basename.split('_')
        if len(parts) >= 2:
            point_id = parts[0]
            heading = parts[1]
        else:
            point_id = basename
            heading = 0

        # 2. 读取与预处理
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        # 3. 推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 4. 上采样
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # 5. 计算指标
        total_pixels = pred_seg.size
        counts = np.bincount(pred_seg.flatten(), minlength=150)

        def get_sum_pct(indices):
            pixel_sum = 0
            for idx in indices:
                if idx < len(counts):
                    pixel_sum += counts[idx]
            return pixel_sum / total_pixels

        # 计算具体指标
        gvi = get_sum_pct(IDX_GREEN)
        svf = get_sum_pct(IDX_SKY)
        enclosure = get_sum_pct(IDX_ENCLOSURE)
        motorization = get_sum_pct(IDX_MOTORIZATION)
        clutter = get_sum_pct(IDX_CLUTTER)

        # Walkability 特殊公式: Sidewalk / (Road + Sidewalk + 0.001)
        # 注意：这里我们用 pixel counts 来算，避免除法误差
        sidewalk_pixels = sum(counts[i] for i in IDX_SIDEWALK if i < len(counts))
        road_pixels = sum(counts[i] for i in IDX_ROAD if i < len(counts))
        walkability = sidewalk_pixels / (road_pixels + sidewalk_pixels + 0.001)

        # 存入结果
        results.append({
            'point_id': point_id,
            'heading': heading,
            'filename': img_file,
            'S_GVI': gvi,
            'S_SVF': svf,
            'S_Enclosure': enclosure,
            'S_Motorization': motorization,
            'S_Walkability': walkability,
            'S_Clutter': clutter
        })

    except Exception as e:
        print(f"处理图片 {img_file} 时出错: {e}")

# ================= 保存结果 =================
df = pd.DataFrame(results)
try:
    df['point_id'] = df['point_id'].astype(int)
    df = df.sort_values(by=['point_id', 'heading'])
except:
    pass

df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ 处理完成！基于 ADE20K 的指标已保存至 {OUTPUT_CSV}")
