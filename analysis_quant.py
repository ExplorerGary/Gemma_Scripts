# analysis_quant.py

'''
使用说明：
--date: 填写日期（必须）
--om: 填写种类 （非必须）
    可用的： Full
            Lora
            all
一个例子：
    python analysis_quant.py --date 20250731 --on Lora
    对20250731的Lora结果进行分析，对对应的：
        time_used
        train/loss
        train/mean_token_accuracy
    进行绘图
'''



import os
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

def scan_csv(base_dir: str):
    '''
    扫描 base_dir 下所有后缀为 .csv 的文件，返回绝对路径列表。
    - 忽略子目录
    - 忽略非 csv 文件
    - 忽略非文件（如目录、系统隐藏项）
    '''
    if not os.path.isdir(base_dir):
        raise ValueError(f"❌ 错误：路径不存在或不是一个目录: {base_dir}")

    avail_csv = []
    for file in os.listdir(base_dir):
        file_path = os.path.join(base_dir, file)
        if os.path.isfile(file_path) and file.endswith(".csv"):
            avail_csv.append(file_path)

    return sorted(avail_csv)

def clean_last_row_if_broken(df):
    # 如果最后一行包含空值，就移除
    if df.tail(1).isnull().values.any():
        df = df[:-1]
    return df

def parse_filename(filename):
    # e.g. Full_1e3_20250731.csv --> ("Full", "1e3", "20250731")
    base = os.path.basename(filename).replace(".csv", "")
    parts = base.split("_")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return None, None, None

def main(date=None, on=None):
    # 获取日志目录
    base_dir = os.path.dirname(__file__)
    log_storage = os.path.join(base_dir, "..", "Training_Logs")

    # 扫描所有 csv 文件
    avail_csv = scan_csv(base_dir=log_storage)

    # 筛选出匹配日期的 CSV
    filtered_csv = [csv for csv in avail_csv if date in os.path.basename(csv)]

    # 按需筛选类型（Full / Lora / all）
    if on in ["Full", "Lora"]:
        filtered_csv = [csv for csv in filtered_csv if os.path.basename(csv).startswith(on)]

    # 没有符合的文件，直接退出
    if not filtered_csv:
        print(f"❌ 没有找到符合 date={date} 和 on={on} 的日志文件")
        return

    # 保存绘图用的数据容器
    time_used_list = []
    loss_lines = []
    acc_lines = []

    for csv_path in filtered_csv:
        df = pd.read_csv(csv_path)
        df = clean_last_row_if_broken(df)

        model_type, scaling, _ = parse_filename(csv_path)
        label = f"{model_type}_{scaling}"

        # 计算耗时（单位秒）
        time_list = pd.to_datetime(df["time"])
        time_used = (time_list.iloc[-1] - time_list.iloc[0]).total_seconds()
        time_used_list.append((label, time_used))

        # 折线图数据
        loss_lines.append((label, df["step"], df["train/loss"]))
        acc_lines.append((label, df["step"], df["train/mean_token_accuracy"]))

    # ======== 绘图函数定义 ========

    def save_bar_time_used(data, save_path):
        import matplotlib.pyplot as plt
        import numpy as np

        labels, times = zip(*data)
        labels = list(labels)
        times = np.array(times)

        # 找 baseline: 以 *_None 为基准
        baseline_time = None
        baseline_label = None
        for label, time in data:
            if label.endswith("_None"):
                baseline_time = time
                baseline_label = label
                break

        # 如果没找到 *_None，则报错退出
        if baseline_time is None:
            raise ValueError("❌ 没有找到 *_None 模型作为基准，无法绘制相对柱状图")

        # 计算相对时间差（越大 = 越慢，越小 = 越快）
        time_deltas = times - baseline_time

        # 决定颜色
        colors = []
        for label, delta in zip(labels, time_deltas):
            if label == baseline_label:
                colors.append("gray")  # baseline 自己
            elif delta > 0:
                colors.append("red")   # 更慢
            elif delta < 0:
                colors.append("green") # 更快
            else:
                colors.append("gray")  # 完全一致也灰色

        # 绘图
        plt.figure(figsize=(12, 6))
        bars = plt.bar(labels, time_deltas, color=colors)

        # 标注每个柱子的数值（正负秒数）
        for bar, delta in zip(bars, time_deltas):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + (1 if delta >= 0 else -1),
                    f"{delta:+.1f}s", ha='center',
                    va='bottom' if delta >= 0 else 'top', fontsize=9)

        plt.axhline(0, color='black', linewidth=1)  # 基准线
        plt.title(f"Relative Training Time (based on {baseline_label})")
        plt.ylabel("Time (seconds)")
        plt.xticks(rotation=45)
        # ✅ 设置 Y 轴坐标为真实耗时值（baseline ± delta）
        ax = plt.gca()
        yticks = ax.get_yticks()
        real_ticks = yticks + baseline_time
        ax.set_yticks(yticks)  # 先设置 ticks（位置）
        ax.set_yticklabels([f"{tick:.1f}s" for tick in real_ticks])  # 再设置标签
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


    def save_line_plot(data_lines, ylabel, title, save_path):
        plt.figure(figsize=(10, 6))
        for label, x, y in data_lines:
            plt.plot(x, y, label=label)
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # ======== 输出图像到 Training_Logs ========
    time_plot_path = os.path.join(log_storage, f"{on}_{date}_time_used.png")
    loss_plot_path = os.path.join(log_storage, f"{on}_{date}_train_loss.png")
    acc_plot_path = os.path.join(log_storage, f"{on}_{date}_train_mean_token_accuracy.png")

    save_bar_time_used(time_used_list, time_plot_path)
    save_line_plot(loss_lines, "train/loss", f"Training Loss on {date}", loss_plot_path)
    save_line_plot(acc_lines, "mean_token_accuracy", f"Token Accuracy on {date}", acc_plot_path)

    print("✅ 分析完成，图表已保存至:", log_storage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="指定分析的日期，如 20250801")
    parser.add_argument("--on", type=str, default="all", required=False, help="分析类型：Full、Lora 或 all")
    
    args_ = parser.parse_args()
    main(date=args_.date, on=args_.on)
