"""
街景语义分割 + 自定义配色 + 六项感知指标计算
模型：SegFormer (Cityscapes)
颜色：植被 #D9FCED / 天空 #405FFA / 建筑 #09BBCB / 道路 #3B3664 / 车辆 #07240F
"""

import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ================== 中文字体设置（Windows） ==================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ================== HuggingFace 镜像（国内加速） ==================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ================== 配置区 ==================
IMAGE_DIR = "./scored_images"           # 原图文件夹
OUTPUT_DIR = "./output_seg"             # 输出文件夹
MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 六项指标对应的 Cityscapes 类别 ID
METRIC_CLASSES = {
    '绿视率(GVI)':        [8, 9],          # Vegetation, Terrain
    '天空开敞度(Sky)':    [10],            # Sky
    '建筑界面率(Building)': [2],           # Building
    '道路率(Road)':       [0],             # Road
    '机动化程度(Motor)':   [13,14,15,16],  # Car, Truck, Bus, Train
    '步行空间率(Walk)':    [1, 11, 12]     # Sidewalk, Person, Rider
}

# ------------------ 自定义调色板（你指定的5色） ------------------
CUSTOM_COLORS = {
    '植被': np.array([217, 252, 237]),   # #D9FCED
    '天空': np.array([64, 95, 250]),     # #405FFA
    '建筑': np.array([9, 187, 200]),     # #09BBCB
    '道路': np.array([59, 54, 100]),     # #3B3664
    '车辆': np.array([7, 36, 15]),       # #07240F
}

# 为19类建立颜色映射，未指定类别用灰色
CLASS_TO_COLOR = {i: np.array([128, 128, 128], dtype=np.uint8) for i in range(19)}
CLASS_TO_COLOR[8]  = CUSTOM_COLORS['植被']   # Vegetation
CLASS_TO_COLOR[9]  = CUSTOM_COLORS['植被']   # Terrain
CLASS_TO_COLOR[10] = CUSTOM_COLORS['天空']   # Sky
CLASS_TO_COLOR[2]  = CUSTOM_COLORS['建筑']   # Building
CLASS_TO_COLOR[0]  = CUSTOM_COLORS['道路']   # Road
CLASS_TO_COLOR[13] = CUSTOM_COLORS['车辆']   # Car
CLASS_TO_COLOR[14] = CUSTOM_COLORS['车辆']   # Truck
CLASS_TO_COLOR[15] = CUSTOM_COLORS['车辆']   # Bus
CLASS_TO_COLOR[16] = CUSTOM_COLORS['车辆']   # Train

# ================== 模型加载 ==================
print("加载模型中...")
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================== 处理函数 ==================
def predict_and_colorize(image):
    """返回 (pred_class, color_seg)"""
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits  # [1, 19, H/4, W/4]
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode='bilinear', align_corners=False
    )
    pred_class = upsampled.argmax(dim=1).squeeze().cpu().numpy()

    # 使用自定义颜色上色
    color_seg = np.zeros((pred_class.shape[0], pred_class.shape[1], 3), dtype=np.uint8)
    for cls_id in range(19):
        color_seg[pred_class == cls_id] = CLASS_TO_COLOR[cls_id]
    return pred_class, color_seg

def compute_metrics(pred_class):
    """计算六项语义指标占比"""
    total = pred_class.size
    metrics = {}
    for name, ids in METRIC_CLASSES.items():
        mask = np.isin(pred_class, ids)
        ratio = mask.sum() / total
        metrics[name] = ratio
    return metrics

# ================== 批量处理 ==================
img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
all_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(img_extensions)]
print(f"找到 {len(all_files)} 张图片")

all_metrics = []

for idx, filename in enumerate(all_files):
    img_path = os.path.join(IMAGE_DIR, filename)
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"无法打开 {filename}: {e}")
        continue

    pred_class, color_seg = predict_and_colorize(image)
    metrics = compute_metrics(pred_class)
    metrics['filename'] = filename
    all_metrics.append(metrics)

    # 保存分割彩色图
    seg_img = Image.fromarray(color_seg)
    seg_save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_seg.png")
    seg_img.save(seg_save_path)

    print(f"[{idx+1}/{len(all_files)}] {filename} 完成. 绿视率={metrics['绿视率(GVI)']:.3f}")

# 保存指标CSV
df_metrics = pd.DataFrame(all_metrics)
csv_path = os.path.join(OUTPUT_DIR, "segmentation_metrics.csv")
df_metrics.to_csv(csv_path, index=False, encoding='utf-8')
print(f"指标CSV已保存至 {csv_path}")

# ================== 生成示例排版图（使用第一张） ==================
if all_files:
    demo_img_path = os.path.join(IMAGE_DIR, all_files[0])
    demo_image = Image.open(demo_img_path).convert("RGB")
    pred_class, color_seg = predict_and_colorize(demo_image)
    metrics = compute_metrics(pred_class)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(demo_image)
    axes[0].set_title("原图 (0° 街景)", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(color_seg)
    axes[1].set_title("语义分割 (自定义配色)", fontsize=14)
    axes[1].axis('off')

    # 标注指标
    textstr = "\n".join([f"{k}: {v*100:.1f}%" for k, v in metrics.items()])
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=12,
                 verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()
    demo_save_path = os.path.join(OUTPUT_DIR, "demo_example.png")
    plt.savefig(demo_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"示例图已保存至 {demo_save_path}")

print("全部完成！")