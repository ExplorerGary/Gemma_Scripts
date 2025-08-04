#!/bin/bash

##########################
### 配置部分 ###
##########################

BASE_DIR="/gpfsnyu/scratch/zg2598/Gemma/gemma-3-4b-pt/result_"  # HPC 上训练结果的基础路径
TYPE=("Full" "Lora")                                            # 支持的训练类型
SCALING_VALUES=("256" "1e3" "1e4" "1e5" "None")                  # 不同的 scaling 值
SUB_FOLDER="TRAINING_LOG"                                       # 每个实验下的日志子文件夹
FILE_NAME="training_log.csv"                                    # 想要获取的文件名
REMOTE_HOST="zg2598@hpclogin.shanghai.nyu.edu"                  # 远程主机地址（NetID要按需更换）

LOCAL_DIR="/drives/d/NYU_Files/2025 SPRING/Summer_Research/新/PYTHON/Gemma/Training_Logs"  # 本地目标路径

###################################################################################################
TIMESTAMP="20250731"                                            # 时间戳，可自由修改为当天日期或版本编号
###################################################################################################


##########################
### 代码部分 ###
##########################


##########################
### 本地目录检测 ###

# 如果本地目录不存在，则创建；否则提示已存在
if [ ! -d "${LOCAL_DIR}" ]; then
  echo "📂 本地目录不存在，正在创建：${LOCAL_DIR}"
  mkdir -p "${LOCAL_DIR}"
else
  echo "📂 本地目录已存在：${LOCAL_DIR}"
fi

##########################
### 主文件处理循环 ###

for t in "${TYPE[@]}"; do
  for s in "${SCALING_VALUES[@]}"; do

    # 构造远程完整路径
    REMOTE_PATH="${BASE_DIR}${t}_${s}/${SUB_FOLDER}/${FILE_NAME}"

    # 构造带时间戳的目标文件名，例如 Full_1e3_20250801.csv
    DEST_NAME="${t}_${s}_${TIMESTAMP}.csv"

    echo "📤 正在从远程复制文件: ${REMOTE_PATH}"
    echo "📥 保存为本地文件: ${LOCAL_DIR}/${DEST_NAME}"

    # 尝试复制文件
    scp "${REMOTE_HOST}:${REMOTE_PATH}" "${LOCAL_DIR}/${DEST_NAME}"

    # 检查复制是否成功
    if [ $? -eq 0 ]; then
      echo "✅ 成功保存为 ${DEST_NAME}"
    else
      echo "⚠️ 警告：未能复制 ${REMOTE_PATH}，可能文件不存在"
    fi

    echo "---------------------------------------------"

  done
done

echo "🎉 所有文件处理完成！"
