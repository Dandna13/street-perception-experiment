import tkinter as tk
from tkinter import messagebox, font
from PIL import Image, ImageTk
import pandas as pd
import os
import csv
import random

# === 风格配置 ===
COLOR_BG = "#F0F8FF"  # 爱丽丝蓝（底色）
COLOR_CARD = "#FFFFFF"  # 纯白（卡片色）
COLOR_ACCENT = "#007BFF"  # 科技蓝（按钮与箭头）
COLOR_TEXT = "#2C3E50"  # 深灰蓝（文字）
COLOR_BTN_HOVER = "#0056b3"  # 悬停色


class FreshSequenceScorer:
    def __init__(self, root):
        self.root = root
        self.root.title("城市步行空间视觉节奏感评价系统 v2.0")
        self.root.geometry("1300x750")
        self.root.configure(bg=COLOR_BG)

        # --- 路径配置 (请根据实际修改) ---
        self.csv_path = "sequence_indices_full.csv"
        self.img_folder = "street_views"  # 你的图片文件夹
        self.output_path = "seq_comparisons.csv"

        # 加载数据
        self.load_data()
        self.setup_ui()
        self.next_round()

    def load_data(self):
        if not os.path.exists(self.csv_path):
            messagebox.showerror("缺失文件", f"找不到 {self.csv_path}")
            self.root.destroy()
        self.df = pd.read_csv(self.csv_path)

        # 初始化结果文件
        if not os.path.exists(self.output_path):
            with open(self.output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["winner_p1", "winner_p2", "loser_p1", "loser_p2"])

    def setup_ui(self):
        # 字体设置
        title_font = font.Font(family="Microsoft YaHei", size=18, weight="bold")
        tip_font = font.Font(family="Microsoft YaHei", size=11)
        btn_font = font.Font(family="Microsoft YaHei", size=12, weight="bold")

        # 顶部标题栏
        header = tk.Frame(self.root, bg=COLOR_BG)
        header.pack(fill=tk.X, pady=30)

        tk.Label(header, text="哪一组街道的向前步行体验（空间过渡）让你感觉更有节奏感/更舒适？",
                 font=title_font, bg=COLOR_BG, fg=COLOR_TEXT).pack()
        tk.Label(header, text="提示：请观察图片间的变化幅度与连续性，点击下方按钮选择",
                 font=tip_font, bg=COLOR_BG, fg="#7F8C8D").pack(pady=5)

        # 主对比区域
        self.main_container = tk.Frame(self.root, bg=COLOR_BG)
        self.main_container.pack(expand=True, fill=tk.BOTH, padx=50)

        # 序列 A 卡片
        self.card_a = self.create_sequence_card("序列 A", tk.LEFT)
        # 中间分割线
        tk.Frame(self.main_container, width=2, bg="#D6EAF8").pack(side=tk.LEFT, fill=tk.Y, padx=20)
        # 序列 B 卡片
        self.card_b = self.create_sequence_card("序列 B", tk.RIGHT)

    def create_sequence_card(self, title, side):
        card_frame = tk.Frame(self.main_container, bg=COLOR_CARD, bd=0, highlightbackground="#D6EAF8",
                              highlightthickness=1)
        card_frame.pack(side=side, expand=True, fill=tk.BOTH, padx=10, pady=10)

        tk.Label(card_frame, text=title, font=("Helvetica", 14, "bold"), bg=COLOR_CARD, fg=COLOR_ACCENT).pack(pady=10)

        # 图片显示容器
        img_container = tk.Frame(card_frame, bg=COLOR_CARD)
        img_container.pack(pady=20)

        # 第一张图
        lbl1 = tk.Label(img_container, bg="#EBF5FB")
        lbl1.pack(side=tk.LEFT, padx=10)

        # 箭头
        arrow_label = tk.Label(img_container, text="→", font=("Arial", 30), bg=COLOR_CARD, fg=COLOR_ACCENT)
        arrow_label.pack(side=tk.LEFT)

        # 第二张图
        lbl2 = tk.Label(img_container, bg="#EBF5FB")
        lbl2.pack(side=tk.LEFT, padx=10)

        # 选择按钮
        btn = tk.Button(card_frame, text=f"点击选择 {title}", font=("微软雅黑", 12, "bold"),
                        bg=COLOR_ACCENT, fg="white", relief=tk.FLAT, cursor="hand2",
                        activebackground=COLOR_BTN_HOVER, activeforeground="white",
                        command=lambda t=title: self.vote(t), width=20, pady=8)
        btn.pack(pady=20)

        return {"lbl1": lbl1, "lbl2": lbl2, "data": None}

    def get_photo(self, pid):
        # 设定你要展示的前进视角，默认读取 "_0" (正前方)
        # 如果你的正前方是 90，请改成 angle = "90"
        angle = "0"

        # 匹配你的新命名规则：点位ID_视角.jpg (例如 1_0.jpg)
        img_name = f"{int(pid)}_{angle}.jpg"
        img_path = os.path.join(self.img_folder, img_name)

        try:
            img = Image.open(img_path)
        except:
            # 补全点位没有图时，生成清新色占位图
            img = Image.new('RGB', (300, 300), color='#D6DBDF')

        img = img.resize((320, 240), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def next_round(self):
        # 随机抽取两个序列对
        samples = self.df.sample(2).to_dict('records')
        self.card_a["data"] = samples[0]
        self.card_b["data"] = samples[1]

        # 更新图片
        self.img_a1 = self.get_photo(samples[0]['P1_id'])
        self.img_a2 = self.get_photo(samples[0]['P2_id'])
        self.img_b1 = self.get_photo(samples[1]['P1_id'])
        self.img_b2 = self.get_photo(samples[1]['P2_id'])

        self.card_a["lbl1"].config(image=self.img_a1)
        self.card_a["lbl2"].config(image=self.img_a2)
        self.card_b["lbl1"].config(image=self.img_b1)
        self.card_b["lbl2"].config(image=self.img_b2)

    def vote(self, choice):
        if choice == "序列 A":
            win, lose = self.card_a["data"], self.card_b["data"]
        else:
            win, lose = self.card_b["data"], self.card_a["data"]

        with open(self.output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([win['P1_id'], win['P2_id'], lose['P1_id'], lose['P2_id']])

        self.next_round()


if __name__ == "__main__":
    root = tk.Tk()
    # 设置全局默认字体
    default_font = font.nametofont("TkDefaultFont")
    default_font.configure(family="Microsoft YaHei", size=10)

    app = FreshSequenceScorer(root)
    root.mainloop()