import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import requests

class Vocab:
    def __init__(self):
        self.word2idx = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
    def build_vocab(self, texts, min_freq=2):
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
                
        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
                
    def encode(self, text, max_len=128):
        tokens = ['[CLS]'] + text.split()[:max_len-2] + ['[SEP]']
        ids = [self.word2idx.get(token, self.word2idx['[UNK]']) for token in tokens]
        if len(ids) < max_len:
            ids += [self.word2idx['[PAD]']] * (max_len - len(ids))
        return ids[:max_len]

class DocumentPairDataset(Dataset):
    def __init__(self, data_path, vocab, doc1_max_len=64, doc2_max_len=128):
        self.vocab = vocab
        self.doc1_max_len = doc1_max_len
        self.doc2_max_len = doc2_max_len
        
        # 加载SICK数据集
        self.data = []
        df = pd.read_csv(data_path, sep='\t')
        
        for _, row in df.iterrows():
            self.data.append({
                'doc1': row['sentence_A'],
                'doc2': row['sentence_B'],
                'score': float(row['relatedness_score']) / 5.0,  # 归一化到0-1
                'entailment': row['entailment_judgment']
            })
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        doc1 = item['doc1']
        doc2 = item['doc2']
        score = item['score']
        
        # 对两个文档进行编码
        doc1_ids = self.vocab.encode(doc1, self.doc1_max_len)
        doc2_ids = self.vocab.encode(doc2, self.doc2_max_len)
        
        return {
            'doc1_ids': torch.tensor(doc1_ids, dtype=torch.long),
            'doc2_ids': torch.tensor(doc2_ids, dtype=torch.long),
            'score': torch.tensor(score, dtype=torch.float),
            'doc1': doc1,  # 添加原始文本
            'doc2': doc2   # 添加原始文本
        }

def download_sts():
    """下载并处理SICK数据集"""
    data_dir = 'data/sts'
    os.makedirs(data_dir, exist_ok=True)
    
    # 下载SICK训练集
    url = "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_train.txt"
    train_path = os.path.join(data_dir, 'train.tsv')
    
    if not os.path.exists(train_path):
        print("下载训练集...")
        response = requests.get(url)
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    # 下载SICK验证集
    url = "https://raw.githubusercontent.com/brmson/dataset-sts/master/data/sts/sick2014/SICK_trial.txt"
    dev_path = os.path.join(data_dir, 'dev.tsv')
    
    if not os.path.exists(dev_path):
        print("下载验证集...")
        response = requests.get(url)
        with open(dev_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
    
    return {
        'train': train_path,
        'dev': dev_path
    }

def build_vocab(data_path):
    """构建词表"""
    vocab = Vocab()
    df = pd.read_csv(data_path, sep='\t')
    
    # 收集所有文本
    all_texts = df['sentence_A'].tolist() + df['sentence_B'].tolist()
    
    # 构建词表
    vocab.build_vocab(all_texts)
    return vocab

def get_dataloader(data_path, vocab, batch_size=32, shuffle=True, doc1_max_len=64, doc2_max_len=128):
    """创建数据加载器"""
    dataset = DocumentPairDataset(data_path, vocab, doc1_max_len, doc2_max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8
    ) 