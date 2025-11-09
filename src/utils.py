# src/utils.py
import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# 导入配置
from src import config

def set_seed(seed: int):
    """
    设置全局随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Global random seed set to {seed}")

def count_parameters(model: nn.Module) -> int:
    """
    统计模型参数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_loss_plot(train_losses, val_losses, filepath: str):
    """
    保存训练/验证损失曲线图
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 确保 results 目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    plt.close()
    print(f"Loss plot saved to {filepath}")

def save_accuracy_plot(train_accs, val_accs, filepath: str):
    """
    保存训练/验证准确率曲线图
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 确保 results 目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    plt.close()
    print(f"Accuracy plot saved to {filepath}")

def save_checkpoint(model, optimizer, epoch, loss, filepath: str):
    """
    保存模型检查点
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved to {filepath} (Epoch {epoch})")

def load_checkpoint(filepath: str, model, optimizer=None):
    """
    加载模型检查点
    
    Args:
        filepath: 检查点文件路径
        model: 要加载权重的模型
        optimizer: 要加载状态的优化器（可选）
        
    Returns:
        如果提供了optimizer，则返回(epoch, loss)
        否则返回None
    """
    if not os.path.exists(filepath):
        print("No checkpoint found, starting from scratch.")
        return 0, float('inf') if optimizer else None
        
    checkpoint = torch.load(filepath, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {filepath} (Epoch {epoch}, Loss {loss:.4f})")
        return epoch, loss
    else:
        print(f"Model weights loaded from {filepath}")
        return None