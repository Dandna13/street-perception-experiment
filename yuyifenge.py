import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# 1. 初始化图像处理器和模型
print("正在加载模型，这可能需要一些时间...")
model_name = "nvidia/segformer-b3-finetuned-cityscapes"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)

# 2. 加载输入图像 (请将此路径替换为你实际要测试的图片路径)
image_path = "scored_images/1062_90.jpg"
try:
    image = Image.open(image_path).convert("RGB")
except FileNotFoundError:
    print(f"找不到图片: {image_path}，请检查路径是否正确。")
    exit()

# 3. 图像预处理与模型推理
print("正在进行语义分割...")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# 4. 获取预测的 logits 并调整尺寸回原图大小
logits = outputs.logits
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1], # 目标尺寸为 (高度, 宽度)
    mode="bilinear",
    align_corners=False,
)

# 5. 获取每个像素的类别预测结果
predicted_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

# 6. 可视化：为不同类别生成颜色映射
# 设定一个固定的随机数种子，确保每次同一类的颜色相同
np.random.seed(42)
palette = np.random.randint(0, 256, size=(256, 3), dtype=np.uint8)

color_seg = np.zeros((predicted_map.shape[0], predicted_map.shape[1], 3), dtype=np.uint8)
for label in np.unique(predicted_map):
    color_seg[predicted_map == label] = palette[label]

# 7. 显示原图和分割结果
# 解决 matplotlib 显示中文可能乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 用户
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # Mac 用户请取消注释此行
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(image)
axes[0].set_title("原始图像")
axes[0].axis("off")

axes[1].imshow(color_seg)
axes[1].set_title("语义分割结果")
axes[1].axis("off")

plt.tight_layout()
plt.show()