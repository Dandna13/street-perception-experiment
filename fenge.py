import os
import numpy as np
import torch
#import cv2
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# 配置区（按需修改）
# =========================================================
IMAGE_DIR = "./scored_images"          # 原图文件夹
OUTPUT_DIR = "./output_seg"            # 输出文件夹
MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 六项指标对应的 Cityscapes 类别 ID
METRIC_CLASSES = {
    '绿视率(GVI)':        [8, 9],      # Vegetation, Terrain
    '天空开敞度(Sky)':    [10],        # Sky
    '建筑界面率(Building)': [2],       # Building
    '道路率(Road)':       [0],         # Road
    '机动化程度(Motor)':   [13,14,15,16],  # Car, Truck, Bus, Train
    '步行空间率(Walk)':   [1, 11, 12]  # Sidewalk, Person, Rider
}

# Cityscapes 颜色映射（RGB）
CITYSCAPES_PALETTE = np.array([
    [128,64,128],[244,35,232],[70,70,70],[102,102,156],[190,153,153],
    [153,153,153],[250,170,30],[220,220,0],[107,142,35],[152,251,152],
    [70,130,180],[220,20,60],[255,0,0],[0,0,142],[0,0,70],
    [0,60,100],[0,80,100],[0,0,230],[119,11,32]
], dtype=np.uint8)

# =========================================================
# 初始化模型
# =========================================================
print("Loading model...")
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 处理函数
# =========================================================
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

    # 上色
    color_seg = np.zeros((pred_class.shape[0], pred_class.shape[1], 3), dtype=np.uint8)
    for cls_id, color in enumerate(CITYSCAPES_PALETTE):
        color_seg[pred_class == cls_id] = color
    return pred_class, color_seg

def compute_metrics(pred_class):
    total = pred_class.size
    metrics = {}
    for name, ids in METRIC_CLASSES.items():
        mask = np.isin(pred_class, ids)
        ratio = mask.sum() / total
        metrics[name] = ratio
    return metrics

# =========================================================
# 批量处理
# =========================================================
img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
all_files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(img_extensions)]
print(f"Found {len(all_files)} images.")

all_metrics = []

for idx, filename in enumerate(all_files):
    img_path = os.path.join(IMAGE_DIR, filename)
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error opening {filename}: {e}")
        continue

    pred_class, color_seg = predict_and_colorize(image)
    metrics = compute_metrics(pred_class)
    metrics['filename'] = filename
    all_metrics.append(metrics)

    # 保存分割彩色图
    seg_img = Image.fromarray(color_seg)
    seg_save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_seg.png")
    seg_img.save(seg_save_path)

    print(f"[{idx+1}/{len(all_files)}] {filename} done. GVI={metrics['绿视率(GVI)']:.3f}")

# 保存指标 CSV
df_metrics = pd.DataFrame(all_metrics)
csv_path = os.path.join(OUTPUT_DIR, "segmentation_metrics.csv")
df_metrics.to_csv(csv_path, index=False, encoding='utf-8')
print(f"All metrics saved to {csv_path}")

# =========================================================
# （可选）生成一张示例排版图（用第一张图片）
# =========================================================
if all_files:
    demo_img_path = os.path.join(IMAGE_DIR, all_files[0])
    demo_image = Image.open(demo_img_path).convert("RGB")
    pred_class, color_seg = predict_and_colorize(demo_image)
    metrics = compute_metrics(pred_class)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(demo_image)
    axes[0].set_title("Original Street View", fontsize=14)
    axes[0].axis('off')

    axes[1].imshow(color_seg)
    axes[1].set_title("Semantic Segmentation", fontsize=14)
    axes[1].axis('off')

    # 标注六项指标
    textstr = "\n".join([f"{k}: {v*100:.1f}%" for k, v in metrics.items()])
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=12,
                 verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()
    demo_save_path = os.path.join(OUTPUT_DIR, "demo_example.png")
    plt.savefig(demo_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Demo figure saved to {demo_save_path}")