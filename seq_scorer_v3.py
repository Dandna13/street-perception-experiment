import tkinter as tk
from tkinter import messagebox, font, simpledialog
from PIL import Image, ImageTk
import pandas as pd
import os
import csv
import random
import time

# === 风格与实验配置 ===
COLOR_BG = "#F0F8FF"
COLOR_CARD = "#FFFFFF"
COLOR_ACCENT = "#007BFF"
COLOR_TEXT = "#2C3E50"

TOTAL_STANDARD_TRIALS = 40  # 正常测试组数[cite: 2]
TOTAL_ATTENTION_CHECKS = 2  # 注意力检测组数[cite: 2]
TOTAL_TRIALS = TOTAL_STANDARD_TRIALS + TOTAL_ATTENTION_CHECKS

# 自定义注意力检测题 (你需要手动填入两组“极其明显、毫无争议”的对比序列ID)
# 格式: { 'winner': (P1_id, P2_id), 'loser': (P1_id, P2_id) }
ATTENTION_PAIRS = [
    {'winner': (10, 11), 'loser': (200, 201)},  # 替换为实际的ID
    {'winner': (50, 51), 'loser': (300, 301)}  # 替换为实际的ID
]


class ProfessionalSequenceScorer:
    def __init__(self, root):
        self.root = root
        self.root.title("动态空间序列感知实验系统 v3.0")
        self.root.geometry("1300x800")
        self.root.configure(bg=COLOR_BG)

        self.csv_path = "sequence_indices_full.csv"
        self.img_folder = "street_views"
        self.output_path = "seq_comparisons_pro.csv"

        # 实验状态变量
        self.participant_id = ""
        self.current_trial = 0
        self.start_time = 0
        self.trial_queue = []  # 存放打分序列

        self.load_data()
        self.get_participant_id()

    def get_participant_id(self):
        # 实验开始前登记被试编号
        self.root.withdraw()  # 隐藏主窗口
        pid = simpledialog.askstring("被试登记", "请输入被试编号 (例如: P001):")
        if not pid:
            self.root.destroy()
            return
        self.participant_id = pid
        self.root.deiconify()  # 显示主窗口
        self.setup_ui()
        self.prepare_trials()
        self.next_round()

    def load_data(self):
        if not os.path.exists(self.csv_path):
            messagebox.showerror("错误", f"找不到 {self.csv_path}")
            self.root.destroy()
        self.df = pd.read_csv(self.csv_path)

        # 初始化结果文件，增加响应时间和检测标记[cite: 2]
        if not os.path.exists(self.output_path):
            with open(self.output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Participant_ID", "Trial_Num", "Winner_P1", "Winner_P2",
                                 "Loser_P1", "Loser_P2", "Response_Time_s", "Is_Attention_Check", "Passed_Check"])

    def prepare_trials(self):
        # 抽取 40 组正常数据
        samples = self.df.sample(TOTAL_STANDARD_TRIALS).to_dict('records')
        for sample in samples:
            self.trial_queue.append({
                'type': 'standard',
                'p1_id': sample['P1_id'], 'p2_id': sample['P2_id']
            })

        # 插入 2 组注意力检测题
        for check in ATTENTION_PAIRS:
            self.trial_queue.append({
                'type': 'attention',
                'correct_win': check['winner'],
                'correct_lose': check['loser']
            })

        # 随机打乱实验顺序
        random.shuffle(self.trial_queue)

    def setup_ui(self):
        title_font = font.Font(family="Microsoft YaHei", size=16, weight="bold")

        # 进度条提示
        self.lbl_progress = tk.Label(self.root, text="", font=("Arial", 12), bg=COLOR_BG, fg="gray")
        self.lbl_progress.pack(pady=10)

        # 核心问题 (完全契合文档)[cite: 2]
        tk.Label(self.root, text="哪一组街道的向前步行体验（空间过渡）让你感觉更有节奏感/更舒适/更具安全感？",
                 font=title_font, bg=COLOR_BG, fg=COLOR_TEXT).pack(pady=10)

        self.main_container = tk.Frame(self.root, bg=COLOR_BG)
        self.main_container.pack(expand=True, fill=tk.BOTH, padx=50)

        self.card_a = self.create_sequence_card("序列 A", tk.LEFT)
        tk.Frame(self.main_container, width=2, bg="#D6EAF8").pack(side=tk.LEFT, fill=tk.Y, padx=20)
        self.card_b = self.create_sequence_card("序列 B", tk.RIGHT)

        self.animate_arrows()  # 启动动态箭头[cite: 2]

    def create_sequence_card(self, title, side):
        card_frame = tk.Frame(self.main_container, bg=COLOR_CARD, bd=0, highlightbackground="#D6EAF8",
                              highlightthickness=1)
        card_frame.pack(side=side, expand=True, fill=tk.BOTH, padx=10, pady=10)

        tk.Label(card_frame, text=title, font=("Helvetica", 14, "bold"), bg=COLOR_CARD, fg=COLOR_ACCENT).pack(pady=10)

        img_container = tk.Frame(card_frame, bg=COLOR_CARD)
        img_container.pack(pady=20)

        lbl1 = tk.Label(img_container, bg="#EBF5FB")
        lbl1.pack(side=tk.LEFT, padx=10)

        # 动态箭头标签
        arrow_lbl = tk.Label(img_container, text="→\n向前走", font=("Microsoft YaHei", 12, "bold"), bg=COLOR_CARD,
                             fg=COLOR_ACCENT)
        arrow_lbl.pack(side=tk.LEFT)

        lbl2 = tk.Label(img_container, bg="#EBF5FB")
        lbl2.pack(side=tk.LEFT, padx=10)

        btn = tk.Button(card_frame, text=f"选择 {title}", font=("微软雅黑", 14, "bold"),
                        bg=COLOR_ACCENT, fg="white", relief=tk.FLAT, cursor="hand2",
                        command=lambda t=title: self.vote(t), width=20, pady=10)
        btn.pack(pady=20)

        return {"lbl1": lbl1, "lbl2": lbl2, "arrow": arrow_lbl, "data": None}

    def animate_arrows(self):
        # 简单的动态箭头动画：颜色闪烁[cite: 2]
        current_color = self.card_a["arrow"].cget("foreground")
        next_color = "#85C1E9" if current_color == COLOR_ACCENT else COLOR_ACCENT
        self.card_a["arrow"].config(foreground=next_color)
        self.card_b["arrow"].config(foreground=next_color)
        self.root.after(600, self.animate_arrows)

    def get_photo(self, pid):
        img_name = f"{int(pid)}_0.jpg"  # 默认 0度视角
        img_path = os.path.join(self.img_folder, img_name)
        try:
            img = Image.open(img_path)
        except:
            img = Image.new('RGB', (300, 300), color='#D6DBDF')
        img = img.resize((320, 240), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img)

    def next_round(self):
        if self.current_trial >= TOTAL_TRIALS:
            messagebox.showinfo("完成", f"感谢参与！{self.participant_id} 的 42 组测试已全部完成。")
            self.root.destroy()
            return

        self.current_trial += 1
        self.lbl_progress.config(text=f"当前进度: {self.current_trial} / {TOTAL_TRIALS}")

        trial_data = self.trial_queue[self.current_trial - 1]

        # 准备本轮的两组序列数据
        if trial_data['type'] == 'standard':
            # 随机从题库再抽一个作为对比
            competitor = self.df.sample(1).iloc[0]
            seq1 = (trial_data['p1_id'], trial_data['p2_id'])
            seq2 = (competitor['P1_id'], competitor['P2_id'])
        else:
            seq1 = trial_data['correct_win']
            seq2 = trial_data['correct_lose']

        # 随机分配到 A 卡片 或 B 卡片
        choices = [seq1, seq2]
        random.shuffle(choices)
        self.card_a['data'] = choices[0]
        self.card_b['data'] = choices[1]

        # 记录当前 trial 的真实状态
        self.current_trial_info = trial_data

        # 渲染图片
        self.img_a1, self.img_a2 = self.get_photo(choices[0][0]), self.get_photo(choices[0][1])
        self.img_b1, self.img_b2 = self.get_photo(choices[1][0]), self.get_photo(choices[1][1])

        self.card_a["lbl1"].config(image=self.img_a1)
        self.card_a["lbl2"].config(image=self.img_a2)
        self.card_b["lbl1"].config(image=self.img_b1)
        self.card_b["lbl2"].config(image=self.img_b2)

        self.start_time = time.time()  # 开始计时[cite: 2]

    def vote(self, choice):
        # 记录响应时间[cite: 2]
        response_time = round(time.time() - self.start_time, 2)

        win_seq = self.card_a['data'] if choice == "序列 A" else self.card_b['data']
        lose_seq = self.card_b['data'] if choice == "序列 A" else self.card_a['data']

        # 处理注意力检测逻辑[cite: 2]
        is_attn = 1 if self.current_trial_info['type'] == 'attention' else 0
        passed = ""
        if is_attn:
            passed = 1 if win_seq == self.current_trial_info['correct_win'] else 0

        # 保存结果
        with open(self.output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.participant_id, self.current_trial,
                win_seq[0], win_seq[1], lose_seq[0], lose_seq[1],
                response_time, is_attn, passed
            ])

        self.next_round()


if __name__ == "__main__":
    root = tk.Tk()
    default_font = font.nametofont("TkDefaultFont")
    default_font.configure(family="Microsoft YaHei", size=10)
    app = ProfessionalSequenceScorer(root)
    root.mainloop()