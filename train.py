import torch
import torch.nn as nn
import torch.optim as optim
from model import DocumentSimilarityModel
from data_utils import download_sts, build_vocab, get_dataloader
from config import ModelConfig
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        # 将数据移到设备上
        doc1_ids = batch['doc1_ids'].to(device)
        doc2_ids = batch['doc2_ids'].to(device)
        scores = batch['score'].to(device)
        
        # 前向传播
        outputs = model(doc1_ids, doc2_ids)
        loss = criterion(outputs, scores)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, eval_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            doc1_ids = batch['doc1_ids'].to(device)
            doc2_ids = batch['doc2_ids'].to(device)
            scores = batch['score']
            
            outputs = model(doc1_ids, doc2_ids)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(scores.numpy())
    
    # 计算RMSE
    rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
    
    # 计算Spearman相关系数
    spearman_corr, p_value = spearmanr(all_preds, all_labels)
    
    # 打印一些预测结果
    # print("\n预测结果示例：")
    # for i in range(min(3, len(all_preds))):
    #     print(f"样本 {i+1}:")
    #     print(f"预测分数: {all_preds[i]:.4f}")
    #     print(f"真实分数: {all_labels[i]:.4f}")
    
    return rmse, spearman_corr

def main():
    # 加载配置
    config = ModelConfig()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 准备数据集
    data_paths = download_sts()
    
    # 构建词表
    vocab = build_vocab(data_paths['train'])
    print(f"词表大小: {vocab.vocab_size}")
    
    # 创建数据加载器
    train_loader = get_dataloader(
        data_paths['train'], 
        vocab, 
        batch_size=config.batch_size,
        doc1_max_len=config.doc1_max_len,
        doc2_max_len=config.doc2_max_len
    )
    dev_loader = get_dataloader(
        data_paths['dev'], 
        vocab, 
        batch_size=config.batch_size, 
        shuffle=False,
        doc1_max_len=config.doc1_max_len,
        doc2_max_len=config.doc2_max_len
    )
    
    # 初始化模型
    model = DocumentSimilarityModel(
        vocab_size=vocab.vocab_size,
        config=config
    ).to(device)
    
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 设置优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    # 训练参数
    best_spearman = -1  # 改为使用Spearman相关系数
    
    # 创建检查点目录
    os.makedirs('checkpoints', exist_ok=True)
    
    # 训练循环
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        
        # 训练
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        # 评估
        rmse, spearman = evaluate(model, dev_loader, device)
        print(f"Validation RMSE: {rmse:.4f}")
        print(f"Validation Spearman Correlation: {spearman:.4f}")
        
        # 保存最佳模型（基于Spearman相关系数）
        if spearman > best_spearman:
            best_spearman = spearman
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'vocab': vocab,
                'config': config,
                'best_spearman': best_spearman
            }, 'checkpoints/best_model.pth')
            print("Saved best model!")

if __name__ == "__main__":
    main() 