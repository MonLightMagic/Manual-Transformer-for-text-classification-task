# 小型文本建模框架 (Small Text Modeling Framework)

## 项目简介

这是一个轻量级的Transformer文本分类框架，支持以下核心特性：

- 基于Transformer架构的文本分类模型
- 支持标准绝对位置编码
- 实现Shaw等人提出的相对位置编码
- 灵活的模型配置（编码器层数、注意力头数等）
- 自动化对比实验脚本
- 详细的训练日志和性能可视化

## 项目结构

```
small_text_modeling/
├── glue_data/          # 数据集目录（SST2情感分析数据集）
│   ├── sst2_test.json
│   ├── sst2_train.json
│   └── sst2_validation.json
├── models/             # 模型保存目录
├── results/            # 训练结果和可视化
├── scripts/            # 脚本目录
│   ├── run.sh          # 运行脚本
│   └── run_experiments.py  # 对比实验脚本
└── src/                # 源代码目录
    ├── config.py       # 配置文件
    ├── data.py         # 数据处理
    ├── layers.py       # Transformer层实现
    ├── model.py        # 模型定义
    ├── modules.py      # 核心模块（位置编码、注意力机制等）
    ├── train.py        # 训练逻辑
    └── utils.py        # 工具函数
```

## 功能特性

### 1. 位置编码实现

- **绝对位置编码**：标准的Transformer位置编码，使用正弦余弦函数
- **相对位置编码**：实现Shaw等人的相对位置编码方法，通过计算相对位置索引和嵌入向量

### 2. 灵活的模型配置

支持通过配置文件自定义以下参数：
- 模型维度（D_MODEL）
- 编码器层数（N_LAYERS）
- 注意力头数（N_HEADS）
- 残差连接（USE_RESIDUALS）
- 位置编码类型（USE_POSITIONAL_ENCODING / USE_RELATIVE_POSITION_ENCODING）

### 3. 对比实验支持

提供自动化实验脚本，支持：
- 基本实验（不使用相对位置编码）
- 使用相对位置编码的实验
- 不同编码器层数和注意力头数组合的实验
- 所有实验结果自动分类保存，便于对比分析

## 安装说明

### 1. 克隆项目（如果适用）

```bash
# 进入工作目录
cd /data/wzy/zy/
# 项目已在small_text_modeling目录下
```

### 2. 安装依赖

```bash
cd small_text_modeling
pip install -r requirements.txt
```

### 3. 准备数据集

确保`glue_data`目录下包含SST2数据集文件：
- sst2_train.json
- sst2_validation.json
- sst2_test.json

## 使用方法

### 1. 单个实验

运行基本训练流程：

```bash
cd src
python train.py
```

### 2. 运行对比实验

使用实验脚本进行多种配置的对比实验：

```bash
cd scripts
python run_experiments.py
```

运行后会显示以下选项：
```
请选择实验模式:
1. 运行预设的对比实验套件
   - 实验1: 基本实验（不使用相对位置编码）
   - 实验2: 使用相对位置编码的实验
   - 实验3-6: 不同encoder层数和注意力头数的组合（不使用相对位置编码）
2. 运行自定义实验
请输入选项 (1-2):
```

### 3. 自定义配置

可以直接修改`src/config.py`文件中的参数：

```python
# 模型参数
D_MODEL = 64
N_LAYERS = 6
N_HEADS = 8

# 实验配置
EXPERIMENT_ID = 'custom_exp'
USE_POSITIONAL_ENCODING = True
USE_RELATIVE_POSITION_ENCODING = False  # 设为True时忽略USE_POSITIONAL_ENCODING
USE_RESIDUALS = True
```

## 实验结果

实验结果保存在以下位置：

- **模型检查点**：`models/text_classifier_[实验ID]_checkpoint.pth`
- **训练日志**：`results/training_log_[实验ID].txt`
- **损失曲线**：`results/loss_curve_[实验ID].png`
- **准确率曲线**：`results/accuracy_curve_[实验ID].png`

## 核心实现说明

### 相对位置编码

相对位置编码实现在`src/modules.py`中的`RelativePositionEmbedding`类：

- 预计算相对位置嵌入向量
- 生成相对位置索引矩阵
- 在注意力计算中融合相对位置信息

### 注意力机制

增强的注意力计算实现在`scaled_dot_product_attention`函数中，支持传入`relative_pos_embedding`参数。

## 常见问题

### 1. 导入错误

项目已通过自动路径设置解决导入问题，确保可以从任何目录正确运行。

### 2. 实验配置冲突

当`USE_RELATIVE_POSITION_ENCODING = True`时，会自动忽略`USE_POSITIONAL_ENCODING`设置。

### 3. D_MODEL必须能被N_HEADS整除

确保配置的模型维度（D_MODEL）能被注意力头数（N_HEADS）整除，否则会导致维度不匹配错误。

## 技术栈

- Python 3
- PyTorch
- Transformers (Hugging Face)
- NumPy
- Matplotlib (用于可视化)

## 许可证

[MIT License]