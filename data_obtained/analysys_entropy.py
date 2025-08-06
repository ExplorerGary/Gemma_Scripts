import os
import pandas as pd
import matplotlib.pyplot as plt

# 获取当前脚本所在目录
base_dir = os.path.dirname(__file__)
rank_ids = [0, 1]
dfs = []

# 读取每个 rank 的 CSV 文件
for rank in rank_ids:
    csv_file = f"002_BUCKET_ENTROPY_rank{rank}.csv"
    csv_path = os.path.join(base_dir, csv_file)
    df = pd.read_csv(csv_path)
    df["rank"] = rank  # 添加 rank 列以便后续区分
    dfs.append(df)

# 合并所有 rank 的数据
merged_df = pd.concat(dfs, ignore_index=True)

# ===================================
# 1. 绘制各 rank 的 entropy 直方图
# ===================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for idx, rank in enumerate(rank_ids):
    rank_df = merged_df[merged_df["rank"] == rank]
    axes[idx].hist(rank_df["entropy"], bins=30, edgecolor='black')
    axes[idx].set_title(f"Rank {rank} 的 Entropy 分布")
    axes[idx].set_xlabel("Entropy")
    axes[idx].set_ylabel("Bucket 数量")

plt.tight_layout()
plt.savefig(os.path.join(base_dir, "rank_entropy_histogram.png"))
plt.close()

# ============================================
# 2. 按 step 统计平均 entropy 并绘制折线图
# ============================================

# 先提取 step 信息：bucket_name = R_1_E_0_S_1_B_0
def extract_step(bucket_name):
    parts = bucket_name.split("_")
    for i in range(len(parts)):
        if parts[i] == "S":
            return int(parts[i+1])
    return None

# 添加 step 列
merged_df["step"] = merged_df["bucket_name"].apply(lambda x: int(x.split("_")[5]))

# 按 rank 和 step 计算平均 entropy
grouped = merged_df.groupby(["rank", "step"])["entropy"].mean().reset_index()

# 绘图
plt.figure(figsize=(8, 5))
for rank in rank_ids:
    sub_df = grouped[grouped["rank"] == rank]
    plt.plot(sub_df["step"], sub_df["entropy"], marker='o', label=f"Rank {rank}")

plt.xlabel("Step")
plt.ylabel("平均 Entropy")
plt.title("不同 Step 下各 Rank 的平均 Entropy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "avg_entropy_by_step.png"))
plt.close()
