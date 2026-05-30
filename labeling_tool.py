import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
import os
import random
import csv
import time

# ================= 配置区域 =================
IMAGE_FOLDER = 'scored_images_before'  # 图片文件夹
OUTPUT_CSV = 'comparisons.csv'  # 结果保存路径
WINDOW_SIZE = "1000x700"  # 窗口大小
IMG_DISPLAY_SIZE = (450, 340)  # 图片显示尺寸 (宽, 高)

# 评价维度 (可在界面中切换)
ASPECTS = ["安全 (Safety)", "舒适 (Comfort)", "美观 (Beauty)", "压抑 (Depressing)"]


class StreetViewLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("街景感知数据采集工具 (S-O-R 实验)")
        self.root.geometry(WINDOW_SIZE)

        # 1. 初始化数据
        self.image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(self.image_files) < 2:
            messagebox.showerror("错误", f"文件夹 {IMAGE_FOLDER} 中图片数量不足！")
            root.destroy()
            return

        self.current_aspect = tk.StringVar(value=ASPECTS[0])
        self.count = 0
        self.ensure_csv_exists()

        # 2. 界面布局
        # 顶部控制区
        top_frame = tk.Frame(root, pady=10)
        top_frame.pack()

        tk.Label(top_frame, text="当前评价维度:", font=("微软雅黑", 12)).pack(side=tk.LEFT)
        style = ttk.Style()
        style.configure('TMenubutton', font=('微软雅黑', 11))
        self.aspect_menu = ttk.OptionMenu(top_frame, self.current_aspect, ASPECTS[0], *ASPECTS,
                                          command=self.update_question)
        self.aspect_menu.pack(side=tk.LEFT, padx=10)

        self.lbl_question = tk.Label(root, text="", font=("微软雅黑", 16, "bold"), fg="#333")
        self.lbl_question.pack(pady=10)

        # 图片显示区
        img_frame = tk.Frame(root)
        img_frame.pack(expand=True)

        # 左图
        self.btn_left = tk.Button(img_frame, command=lambda: self.record_choice("left"))
        self.btn_left.grid(row=0, column=0, padx=20)
        self.lbl_left_name = tk.Label(img_frame, text="", font=("Arial", 10), fg="gray")
        self.lbl_left_name.grid(row=1, column=0)

        # VS 标签
        tk.Label(img_frame, text="VS", font=("Arial", 20, "bold"), fg="#888").grid(row=0, column=1)

        # 右图
        self.btn_right = tk.Button(img_frame, command=lambda: self.record_choice("right"))
        self.btn_right.grid(row=0, column=2, padx=20)
        self.lbl_right_name = tk.Label(img_frame, text="", font=("Arial", 10), fg="gray")
        self.lbl_right_name.grid(row=1, column=2)

        # 底部状态栏
        self.lbl_status = tk.Label(root, text="准备就绪...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

        # 3. 开始第一轮
        self.update_question()
        self.load_next_pair()

    def ensure_csv_exists(self):
        if not os.path.exists(OUTPUT_CSV):
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Winner_Image", "Loser_Image", "Aspect", "Timestamp"])

    def update_question(self, *args):
        aspect = self.current_aspect.get()
        if "安全" in aspect:
            q = "哪张街道让你感觉更【安全】？"
        elif "舒适" in aspect:
            q = "哪张街道让你感觉更【舒适】？"
        elif "美观" in aspect:
            q = "哪张街道让你感觉更【美观】？"
        elif "压抑" in aspect:
            q = "哪张街道让你感觉更【压抑】？"
        else:
            q = f"请针对 {aspect} 进行选择"
        self.lbl_question.config(text=q)

    def load_next_pair(self):
        # 随机抽取两张不重复的图
        self.img_pair = random.sample(self.image_files, 2)

        # 加载图片
        self.photo_left = self.load_image(self.img_pair[0])
        self.photo_right = self.load_image(self.img_pair[1])

        # 更新界面
        self.btn_left.config(image=self.photo_left)
        self.btn_right.config(image=self.photo_right)
        self.lbl_left_name.config(text=self.img_pair[0])
        self.lbl_right_name.config(text=self.img_pair[1])

    def load_image(self, filename):
        path = os.path.join(IMAGE_FOLDER, filename)
        img = Image.open(path)
        img = img.resize(IMG_DISPLAY_SIZE, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def record_choice(self, winner_side):
        if winner_side == "left":
            winner = self.img_pair[0]
            loser = self.img_pair[1]
        else:
            winner = self.img_pair[1]
            loser = self.img_pair[0]

        aspect = self.current_aspect.get().split(' ')[0]  # 只取中文名，如"安全"

        # 写入 CSV
        with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([winner, loser, aspect, time.strftime("%Y-%m-%d %H:%M:%S")])

        self.count += 1
        self.lbl_status.config(text=f"已记录 {self.count} 组数据 | 刚刚选择: {winner} > {loser}")

        # 下一组
        self.load_next_pair()


if __name__ == "__main__":
    root = tk.Tk()
    app = StreetViewLabeler(root)
    root.mainloop()
