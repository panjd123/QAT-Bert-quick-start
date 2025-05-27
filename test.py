import torch
from model import DocumentSimilarityModel
import os
import pandas as pd
from data_utils import Vocab, DocumentPairDataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
import numpy as np
from config import ModelConfig
from scipy.stats import spearmanr

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint['vocab']
    config = checkpoint['config']
    
    model = DocumentSimilarityModel(
        vocab_size=vocab.vocab_size,
        config=config
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, vocab, config

def predict_similarity(model, vocab, doc1, doc2, device, config):
    # 对输入进行编码
    doc1_ids = torch.tensor([vocab.encode(doc1, max_len=config.doc1_max_len)], dtype=torch.long).to(device)
    doc2_ids = torch.tensor([vocab.encode(doc2, max_len=config.doc2_max_len)], dtype=torch.long).to(device)
    
    # 预测相似度
    with torch.no_grad():
        score = model(doc1_ids, doc2_ids)
    
    return score.item()

def evaluate_on_dataset(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
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
    
    return rmse, spearman_corr, all_preds, all_labels

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型和词表
    model_path = 'checkpoints/best_model.pth'
    if not os.path.exists(model_path):
        print("Error: Model not found. Please train the model first.")
        return
    
    model, vocab, config = load_model(model_path, device)
    
    # 加载验证集数据
    dev_path = 'data/sts/dev.tsv'
    dev_dataset = DocumentPairDataset(dev_path, vocab)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 在验证集上评估
    print("\n在验证集上评估：")
    rmse, spearman, dev_preds, dev_labels = evaluate_on_dataset(model, dev_loader, device)
    print(f"验证集RMSE: {rmse:.4f}")
    print(f"验证集Spearman相关系数: {spearman:.4f}")
    
    # 打印一些验证集样本的预测结果
    print("\n验证集样本预测结果：")
    for i in range(min(5, len(dev_dataset))):
        sample = dev_dataset[i]
        pred = dev_preds[i]
        true = dev_labels[i]
        print(f"\n样本 {i+1}:")
        print(f"文档1: {sample['doc1']}")
        print(f"文档2: {sample['doc2']}")
        print(f"预测相似度: {pred:.4f}")
        print(f"真实相似度: {true:.4f}")
    
    # 自定义测试用例
    print("\n自定义测试用例：")
    test_cases = [
        {
            "doc1": "The cat is sitting on the mat.",
            "doc2": "A cat is lying on the carpet."
        },
        {
            "doc1": "The weather is beautiful today.",
            "doc2": "It's raining heavily outside."
        },
        {
            "doc1": "I love reading science fiction novels.",
            "doc2": "Reading books is my favorite hobby."
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        similarity = predict_similarity(
            model, 
            vocab, 
            test_case["doc1"], 
            test_case["doc2"], 
            device,
            config
        )
        print(f"\n测试用例 {i}:")
        print(f"文档1: {test_case['doc1']}")
        print(f"文档2: {test_case['doc2']}")
        print(f"相似度分数: {similarity:.4f}")

if __name__ == "__main__":
    main() 