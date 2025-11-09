#!/usr/bin/env python3
# 运行对比实验脚本 - 支持多种配置对比

import os
import sys
import subprocess
import re
from datetime import datetime

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# 配置文件路径
CONFIG_FILE = os.path.join(PROJECT_ROOT, 'src/config.py')

def update_config(config_file, experiment_id, params):
    """
    更新配置文件中的实验参数
    
    Args:
        config_file: 配置文件路径
        experiment_id: 实验ID
        params: 要更新的参数字典，包含 n_layers, n_heads, use_relative_pos 等
    """
    with open(config_file, 'r') as f:
        content = f.read()
    
    # 更新实验ID
    content = re.sub(r'EXPERIMENT_ID\s*=\s*[\'"][^\'"]*[\'"]', 
                   f'EXPERIMENT_ID = "{experiment_id}"', content)
    
    # 更新相对位置编码设置
    if 'use_relative_pos' in params:
        content = re.sub(r'USE_RELATIVE_POSITION_ENCODING\s*=\s*True|False', 
                       f'USE_RELATIVE_POSITION_ENCODING = {params["use_relative_pos"]}', content)
    
    # 更新encoder层数
    if 'n_layers' in params:
        content = re.sub(r'N_LAYERS\s*=\s*\d+', 
                       f'N_LAYERS = {params["n_layers"]}', content)
    
    # 更新注意力头数
    if 'n_heads' in params:
        content = re.sub(r'N_HEADS\s*=\s*\d+', 
                       f'N_HEADS = {params["n_heads"]}', content)
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    print(f"已更新实验配置:")
    print(f"- 实验ID: {experiment_id}")
    for key, value in params.items():
        print(f"- {key}: {value}")

def run_training():
    """
    运行训练脚本
    """
    train_script = os.path.join(PROJECT_ROOT, 'src/train.py')
    
    print("\n开始运行训练脚本...")
    
    # 构建环境变量，确保Python可以找到src模块
    env = os.environ.copy()
    # 将项目根目录添加到PYTHONPATH
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{PROJECT_ROOT}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = PROJECT_ROOT
    
    # 同时设置工作目录为项目根目录，双重保障
    result = subprocess.run([sys.executable, train_script], 
                           cwd=PROJECT_ROOT,  # 设置工作目录为项目根目录
                           env=env,           # 设置环境变量
                           capture_output=False, 
                           text=True)
    return result.returncode == 0

def run_single_experiment(experiment_id, params):
    """
    运行单个实验
    
    Args:
        experiment_id: 实验ID
        params: 实验参数
    """
    print(f"\n{'='*50}")
    print(f"正在运行实验: {experiment_id}")
    print(f"{'='*50}")
    
    # 更新配置
    update_config(CONFIG_FILE, experiment_id, params)
    
    # 运行训练
    success = run_training()
    
    if success:
        print(f"\n实验 {experiment_id} 已成功完成!")
        print(f"\n结果文件:")
        print(f"- 模型检查点: ../models/text_classifier_{experiment_id}_checkpoint.pth")
        print(f"- 训练日志: ../results/training_log_{experiment_id}.txt")
        print(f"- 损失曲线: ../results/loss_curve_{experiment_id}.png")
        print(f"- 准确率曲线: ../results/accuracy_curve_{experiment_id}.png")
        return True
    else:
        print(f"\n实验 {experiment_id} 运行失败!")
        return False

def get_default_params():
    """
    获取默认配置参数
    """
    with open(CONFIG_FILE, 'r') as f:
        content = f.read()
    
    n_layers_match = re.search(r'N_LAYERS\s*=\s*(\d+)', content)
    n_heads_match = re.search(r'N_HEADS\s*=\s*(\d+)', content)
    
    n_layers = int(n_layers_match.group(1)) if n_layers_match else 6
    n_heads = int(n_heads_match.group(1)) if n_heads_match else 8
    
    return {"n_layers": n_layers, "n_heads": n_heads}

def run_comparison_experiments():
    """
    运行对比实验
    - 实验1: 基本实验（不使用相对位置编码）
    - 实验2: 使用相对位置编码的实验
    - 实验3+: 不使用相对位置编码，使用不同的encoder层数和注意力头数
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_params = get_default_params()
    
    experiments = []
    
    # 实验1: 基本实验
    experiments.append({
        "experiment_id": f"base_{timestamp}",
        "params": {
            "use_relative_pos": False,
            "n_layers": default_params["n_layers"],
            "n_heads": default_params["n_heads"]
        }
    })
    
    # 实验2: 使用相对位置编码
    experiments.append({
        "experiment_id": f"relative_pos_{timestamp}",
        "params": {
            "use_relative_pos": True,
            "n_layers": default_params["n_layers"],
            "n_heads": default_params["n_heads"]
        }
    })
    
    # 实验3-6: 不同encoder层数和注意力头数的组合（不使用相对位置编码）
    layer_head_combinations = [
        # (n_layers, n_heads)
        (2, 4),    # 实验3: 较少的层数和头数
        (2, 8),    # 实验4: 较少的层数，标准头数
        (4, 4),    # 实验5: 中等层数，较少的头数
        (8, 16),   # 实验6: 较多的层数和头数
    ]
    
    for i, (n_layers, n_heads) in enumerate(layer_head_combinations, 3):
        experiments.append({
            "experiment_id": f"layers_{n_layers}_heads_{n_heads}_{timestamp}",
            "params": {
                "use_relative_pos": False,
                "n_layers": n_layers,
                "n_heads": n_heads
            }
        })
    
    # 运行所有实验
    print(f"\n准备运行 {len(experiments)} 个对比实验...")
    success_count = 0
    
    for exp in experiments:
        if run_single_experiment(exp["experiment_id"], exp["params"]):
            success_count += 1
    
    print(f"\n{'='*50}")
    print(f"对比实验完成!")
    print(f"成功运行: {success_count}/{len(experiments)}")
    print(f"{'='*50}")
    
    print("\n所有实验结果文件:")
    for exp in experiments:
        exp_id = exp["experiment_id"]
        print(f"\n实验 {exp_id}:")
        print(f"- 模型检查点: ../models/text_classifier_{exp_id}_checkpoint.pth")
        print(f"- 训练日志: ../results/training_log_{exp_id}.txt")
        print(f"- 损失曲线: ../results/loss_curve_{exp_id}.png")
        print(f"- 准确率曲线: ../results/accuracy_curve_{exp_id}.png")

def run_custom_experiment():
    """
    运行自定义实验
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    default_params = get_default_params()
    
    print("\n自定义实验配置:")
    custom_name = input("请输入实验名称: ")
    
    use_relative = input("是否使用相对位置编码? (y/n): ").lower() == 'y'
    
    try:
        n_layers = int(input(f"输入encoder层数 [默认: {default_params['n_layers']}]: ") or default_params['n_layers'])
        n_heads = int(input(f"输入注意力头数 [默认: {default_params['n_heads']}]: ") or default_params['n_heads'])
    except ValueError:
        print("输入无效，使用默认值")
        n_layers = default_params['n_layers']
        n_heads = default_params['n_heads']
    
    # 构建实验ID
    exp_id_parts = [custom_name]
    if use_relative:
        exp_id_parts.append("relative_pos")
    exp_id_parts.append(f"layers_{n_layers}")
    exp_id_parts.append(f"heads_{n_heads}")
    exp_id_parts.append(timestamp)
    
    experiment_id = "_".join(exp_id_parts)
    
    # 运行实验
    params = {
        "use_relative_pos": use_relative,
        "n_layers": n_layers,
        "n_heads": n_heads
    }
    
    run_single_experiment(experiment_id, params)

def main():
    print("=== 对比实验运行脚本 ===\n")
    
    # 询问用户想要运行的实验模式
    print("请选择实验模式:")
    print("1. 运行预设的对比实验套件")
    print("   - 实验1: 基本实验（不使用相对位置编码）")
    print("   - 实验2: 使用相对位置编码的实验")
    print("   - 实验3-6: 不同encoder层数和注意力头数的组合（不使用相对位置编码）")
    print("2. 运行自定义实验")
    
    choice = input("请输入选项 (1-2): ")
    
    if choice == '1':
        run_comparison_experiments()
    elif choice == '2':
        run_custom_experiment()
    else:
        print("无效选择，退出")
        return

if __name__ == "__main__":
    main()