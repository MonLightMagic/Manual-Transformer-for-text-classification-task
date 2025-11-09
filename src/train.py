# src/train.py
import os
import sys

# 将项目根目录添加到Python路径中，确保可以导入src模块
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import time
import numpy as np

# 现在可以使用绝对导入
from src import config
from src.model import TextClassifier
from src.data import get_dataloaders
from src.utils import set_seed, count_parameters, save_loss_plot, save_accuracy_plot, save_checkpoint, load_checkpoint

def train_one_epoch(model, dataloader, optimizer, criterion, scheduler, device, clip_norm):
    """
    训练一个epoch
    """
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0
    
    pbar = tqdm(dataloader, desc=f"Training Epoch")
    for batch in pbar:
        # 获取数据
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        
        # 前向传播
        logits = model(input_ids, attention_mask)
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 计算准确率
        _, predictions = torch.max(logits, dim=1)
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        
        # 更新参数
        optimizer.step()
        
        # 更新学习率
        scheduler.step()
        
        # 累计损失和准确率
        epoch_loss += loss.item()
        epoch_correct += correct
        epoch_total += total
        
        # 更新进度条
        pbar.set_postfix({
            'loss': loss.item(),
            'accuracy': correct / total
        })

    return epoch_loss / len(dataloader), epoch_correct / epoch_total

def evaluate(model, dataloader, criterion, device):
    """
    评估模型
    """
    model.eval()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0
    valid_samples = 0  # 用于统计有效标签的样本数
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Evaluating")
        for batch in pbar:
            # 获取数据
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            
            # 前向传播
            logits = model(input_ids, attention_mask)
            
            # 过滤出标签不为-1的样本（有效标签）
            valid_mask = labels != -1
            valid_labels = labels[valid_mask]
            valid_logits = logits[valid_mask]
            
            # 只在有有效标签时计算损失和准确率
            if valid_labels.size(0) > 0:
                # 计算损失（只使用有效标签）
                loss = criterion(valid_logits, valid_labels)
                
                # 计算准确率
                _, predictions = torch.max(valid_logits, dim=1)
                correct = (predictions == valid_labels).sum().item()
                total = valid_labels.size(0)
                
                # 累计损失和准确率
                epoch_loss += loss.item() * total  # 乘以样本数以得到正确的平均损失
                epoch_correct += correct
                valid_samples += total
                
                # 更新进度条
                pbar.set_postfix({
                    'loss': loss.item(),
                    'accuracy': correct / total
                })
            else:
                # 没有有效标签时的进度条更新
                pbar.set_postfix({
                    'loss': 0.0,
                    'accuracy': 0.0
                })
                
            # 累计总样本数
            epoch_total += labels.size(0)
    
    # 计算平均损失和准确率（基于有效标签的样本）
    avg_loss = epoch_loss / valid_samples if valid_samples > 0 else 0
    avg_acc = epoch_correct / valid_samples if valid_samples > 0 else 0

    return avg_loss, avg_acc

def main():
    # --- 1. 设置与初始化 ---
    set_seed(config.SEED)
    device = config.DEVICE
    os.makedirs(os.path.dirname(config.LOG_FILE_PATH), exist_ok=True)
    log_file = open(config.LOG_FILE_PATH, 'w')
    
    def log_print(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log_print(f"Experiment ID: {config.EXPERIMENT_ID}")
    log_print(f"Using device: {device}")
    
    # 记录详细配置信息
    log_print("\nConfiguration details:")
    log_print(f"Model parameters: d_model={config.D_MODEL}, n_layers={config.N_LAYERS}, n_heads={config.N_HEADS}")
    log_print(f"Dataset: {config.DATASET_NAME} ({config.DATASET_CONFIG})")
    log_print(f"Training parameters: epochs={config.EPOCHS}, batch_size={config.BATCH_SIZE}")
    log_print(f"Optimization: lr={config.LEARNING_RATE}, weight_decay={config.WEIGHT_DECAY}")
    log_print(f"Dropout: {config.DROPOUT}, clip_norm={config.CLIP_GRAD_NORM}")
    
    # 记录位置编码相关配置
    log_print(f"Position encoding: use_positional={config.USE_POSITIONAL_ENCODING}, ")
    log_print(f"use_relative_pos={config.USE_RELATIVE_POSITION_ENCODING}, ")
    log_print(f"use_residuals={config.USE_RESIDUALS}")
    
    # --- 2. 数据加载 ---
    log_print("Loading data...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders()
    log_print(f"Train dataset size: {len(train_loader.dataset)}")
    log_print(f"Validation dataset size: {len(val_loader.dataset)}")
    log_print(f"Test dataset size: {len(test_loader.dataset)}")
    
    # --- 3. 模型初始化 ---
    log_print("Initializing model...")
    # 获取词汇表大小
    vocab_size = tokenizer.vocab_size
    # SST-2数据集有2个类别（积极/消极）
    num_classes = 2
    
    model = TextClassifier(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        max_seq_len=config.MAX_SEQ_LEN,
        num_classes=num_classes
    ).to(device)
    
    log_print(f"Model parameters: {count_parameters(model):,}")
    
    # --- 4. 优化器和损失函数 ---
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        betas=config.ADAMW_BETAS,
        eps=config.ADAMW_EPS,
        weight_decay=config.WEIGHT_DECAY
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.SCHEDULER_WARMUP_STEPS,
        num_training_steps=total_steps
    )
    
    # --- 5. 加载检查点 (如果存在) ---
    start_epoch = 0
    best_val_loss = float('inf')
    try:
        start_epoch, best_val_loss = load_checkpoint(config.MODEL_SAVE_PATH, model, optimizer)
    except Exception as e:
        log_print(f"No valid checkpoint found: {e}")
    
    # --- 6. 训练循环 ---
    log_print(f"Starting training from epoch {start_epoch + 1}")
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(start_epoch, config.EPOCHS):
        log_print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            clip_norm=config.CLIP_GRAD_NORM
        )
        
        # 评估
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        # 保存损失和准确率
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        log_print(f"Epoch time: {epoch_time:.2f}s")
        log_print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        log_print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 保存检查点
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch + 1, val_loss, config.MODEL_SAVE_PATH)
            log_print(f"New best model saved!")
    
    # --- 7. 保存训练结果 ---
    save_loss_plot(train_losses, val_losses, config.LOSS_PLOT_PATH)
    save_accuracy_plot(train_accs, val_accs, config.ACC_PLOT_PATH)
    
    # 打印最终结果
    log_print("\nTraining completed!")
    log_print(f"Best validation loss: {best_val_loss:.4f}")
    log_print(f"Final training accuracy: {train_accs[-1]:.4f}")
    log_print(f"Final validation accuracy: {val_accs[-1]:.4f}")
    
    log_file.close()

if __name__ == "__main__":
    main()