# src/config.py
import torch

# --- 数据集配置 ---
# 使用更小规模的数据集
DATASET_NAME = "glue"
DATASET_CONFIG = "sst2"  # 情感分析数据集，规模较小
# 使用完整数据集进行训练
# TRAIN_SUBSET_SIZE = 2000  # 注释掉子集大小限制
# VALID_SUBSET_SIZE = 500  # 注释掉子集大小限制
SOURCE_COL = 'sentence'
TARGET_COL = 'label'
# 使用本地部署的分词器 - 修改为绝对路径以确保正确加载
TOKENIZER_NAME = "../distilbert-base-uncased"

# --- 模型超参数 (更小型) ---# 确保 d_model % n_heads == 0
D_MODEL = 64
N_LAYERS = 8  # Encoder 层数减少
N_HEADS = 16  
D_K = D_MODEL // N_HEADS
D_V = D_MODEL // N_HEADS
D_FF = 512    # 前馈网络维度减小
DROPOUT = 0.3
MAX_SEQ_LEN = 128  # 句子长度较短

# --- 训练超参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10   # 减少训练轮数
BATCH_SIZE = 128  # 适当增大批次大小
LEARNING_RATE = 5e-5  # 适当提高学习率
ADAMW_BETAS = (0.9, 0.98)
ADAMW_EPS = 1e-9
WEIGHT_DECAY = 0.1
# 梯度裁剪
CLIP_GRAD_NORM = 1.0
# 学习率调度
SCHEDULER_WARMUP_STEPS = 500
SEED = 42

import os
# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 实验配置 ---
EXPERIMENT_ID = "layers_8_heads_16_encoder_decoder"  # 实验ID，用于区分不同实验结果

# --- 文件路径 ---
# 根据实验ID生成不同的保存路径
MODEL_SAVE_PATH = os.path.join(BASE_DIR, f'../models/text_classifier_{EXPERIMENT_ID}_checkpoint.pth')
LOSS_PLOT_PATH = os.path.join(BASE_DIR, f'../results/loss_curve_{EXPERIMENT_ID}.png')
ACC_PLOT_PATH = os.path.join(BASE_DIR, f'../results/accuracy_curve_{EXPERIMENT_ID}.png')
LOG_FILE_PATH = os.path.join(BASE_DIR, f'../results/training_log_{EXPERIMENT_ID}.txt')

# --- 实验控制开关 ---
USE_POSITIONAL_ENCODING = True
USE_RELATIVE_POSITION_ENCODING  = False  # 是否使用相对位置编码（设为True时忽略USE_POSITIONAL_ENCODING）
USE_RESIDUALS = True
USE_DECODER = True  # 是否使用解码器架构
N_DECODER_LAYERS = 4  # 解码器层数，None表示与编码器相同