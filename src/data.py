# src/data.py
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json
import os
from transformers import AutoTokenizer
from src import config

class TextClassificationDataset(Dataset):
    """
    文本分类数据集 - 使用本地JSON文件
    """
    def __init__(self, tokenizer, split):
        # 构建本地数据文件路径
        data_dir = "./glue_data"
        filename_map = {
            'train': 'sst2_train.json',
            'validation': 'sst2_validation.json',
            'test': 'sst2_test.json'
        }
        
        if split not in filename_map:
            raise ValueError(f"不支持的split: {split}")
        
        file_path = os.path.join(data_dir, filename_map[split])
        
        # 从本地JSON文件加载数据
        print(f"正在加载本地数据集: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)
            
        self.tokenizer = tokenizer
        self.max_len = config.MAX_SEQ_LEN

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item[config.SOURCE_COL]
        label = item[config.TARGET_COL]

        # Tokenize 输入文本
        encoding = self.tokenizer(
            text, 
            max_length=self.max_len, 
            truncation=True, 
            padding=False,
            return_tensors=None
        )
        
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

def collate_fn(batch):
    """
    自定义Collate函数，用于处理batch数据
    """
    # 从第一个batch项获取pad_token_id，避免重复加载tokenizer
    pad_token_id = batch[0]['input_ids'][0] if batch else 0  # 使用第一个token作为默认值

    # 填充输入序列
    input_ids = pad_sequence(
        [item['input_ids'] for item in batch], 
        batch_first=True, 
        padding_value=pad_token_id
    )
    
    # 填充注意力掩码
    attention_masks = pad_sequence(
        [item['attention_mask'] for item in batch], 
        batch_first=True, 
        padding_value=0
    )
    
    # 收集标签
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    # 移动到设备
    input_ids = input_ids.to(config.DEVICE)
    attention_masks = attention_masks.to(config.DEVICE)
    labels = labels.to(config.DEVICE)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'label': labels
    }

def get_dataloaders():
    """
    获取训练、验证和测试数据加载器
    """
    # 加载预训练分词器
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    
    # 创建数据集
    train_dataset = TextClassificationDataset(
        tokenizer=tokenizer,
        split='train'
    )
    
    val_dataset = TextClassificationDataset(
        tokenizer=tokenizer,
        split='validation'
    )
    
    test_dataset = TextClassificationDataset(
        tokenizer=tokenizer,
        split='test'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader, tokenizer