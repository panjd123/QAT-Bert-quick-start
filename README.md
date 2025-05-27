# 小型BERT文档相似度计算

这个项目实现了一个基于PyTorch的小型BERT模型，用于计算两篇文档之间的相似度。

## 项目结构

```
.
├── data/                   # 数据目录
├── checkpoints/           # 模型检查点目录
├── data_utils.py         # 数据处理工具
├── model.py              # 模型定义
├── train.py              # 训练脚本
├── test.py               # 测试脚本
└── requirements.txt      # 项目依赖
```

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- Transformers 4.5.0+
- 其他依赖见 requirements.txt

## 安装

1. 克隆项目
2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
python train.py
```

训练过程会自动：
1. 创建示例数据集
2. 初始化模型
3. 开始训练
4. 保存最佳模型到 checkpoints 目录

### 测试模型

```bash
python test.py
```

测试脚本会：
1. 加载训练好的模型
2. 运行示例测试用例
3. 输出预测的相似度分数

## 模型架构

- 使用小型BERT配置（2层，2个注意力头）
- 使用中文BERT词表
- 输出层使用Sigmoid函数，输出范围在0-1之间

## 注意事项

- 当前使用的是示例数据集，实际使用时请替换为真实数据
- 模型参数可以根据需要在 model.py 中调整
- 训练参数可以在 train.py 中调整 