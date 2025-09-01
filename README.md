
# Perceptron-BERT (Fusion Head) — IPO Movement Classifier  
# Perceptron-BERT（仅融合头）— IPO 涨跌预测模型
### everything in the hugging face https://huggingface.co/xxzws/Perceptron-BERT
## all data code in baidu drive  https://pan.baidu.com/s/11-xw26yCHguXqGpjVwbjUg?pwd=g2ru 提取码: g2ru 

## 1) Model Intro / 模型介绍

**EN**  
Perceptron-BERT is a multimodal classifier that fuses **tabular financial features (MLP mid-layer, 762-dim)** and **text features from the IPO prospectus (BERT CLS, 768-dim)**.  
This repository provides the **final fusion head** (`fusion_model.pth`, a single linear layer) used on top of **concatenated features** = `[MLP_mid(762) || BERT_CLS(768)]` → logits(2).

**ZH**  
Perceptron-BERT 是将 **结构化财务特征（MLP 中间层 762 维）** 与 **招股书文本特征（BERT 的 CLS 向量 768 维）** 进行拼接后输入到 **融合分类器（线性层）** 的二分类模型。  
本仓库仅提供最终的 **融合头权重** `fusion_model.pth`，用于对 **拼接后的特征** `[MLP_mid(762) || BERT_CLS(768)]` 进行预测输出（2 类 logits）。

> ℹ️ The fusion head **does not** compute features itself. You must generate the two feature blocks with the **same preprocessing pipeline used in training** (StandardScaler→UMAP→StandardScaler for tabular; BERT-Base-Chinese tokenizer/encoder for text).  
> ℹ️ 融合头 **不负责** 自行提取特征。请使用训练时**一致**的预处理与编码流程生成两块特征（结构化端：标准化→UMAP→再标准化；文本端：BERT-Base-Chinese 的分词与编码）。

---

## 2) How to Call (Using `fusion_model.pth` Only) / 调用代码（仅依赖 `fusion_model.pth`）

**EN**  
Below is a minimal inference example. It loads the fusion head and runs prediction on **precomputed** features:
- `tab_mid.npy`: shape `(N, 762)` — tabular mid features from your trained MLP branch  
- `bert_cls.npy`: shape `(N, 768)` — CLS vectors from your BERT branch

**ZH**  
下例展示最小化推理流程：仅加载融合头，对**已预先计算**的特征进行预测：  
- `tab_mid.npy`：形状 `(N, 762)`，来自已训练 MLP 分支的中间层输出  
- `bert_cls.npy`：形状 `(N, 768)`，来自 BERT 分支的 CLS 向量

```
import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification


class IPOClassifier(nn.Module):
    def __init__(self, num_original, num_categorical, mid_dim, num_classes):
        super().__init__()
        self.numeric_branch = nn.Sequential(
            nn.Linear(30, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 256)
        )
        self.missing_branch = nn.Linear(num_original, 64)
        self.final_branch = nn.Sequential(nn.Linear(320, 256), nn.BatchNorm1d(256), nn.ReLU())
        self.categorical_layer = nn.Sequential(nn.Linear(num_categorical, 128), nn.ReLU())
        self.mid_layer = nn.Sequential(nn.Linear(384, mid_dim), nn.LeakyReLU())
        self.classifier = nn.Linear(mid_dim, num_classes, bias=False)

    def forward(self, main_numeric, missing_indicator, categorical_input):
        num_out = self.numeric_branch(main_numeric)
        missing_out = self.missing_branch(missing_indicator)
        final_combined = self.final_branch(torch.cat([num_out, missing_out], dim=1))
        cat_out = self.categorical_layer(categorical_input)
        mid = self.mid_layer(torch.cat([final_combined, cat_out], dim=1))
        logits = self.classifier(mid)
        return logits, mid

class CombinedModel(nn.Module):
    def __init__(self, extra_net, bert_fs_model, mid_dim, num_classes):
        super().__init__()
        self.extra_net = extra_net
        self.bert_encoder = bert_fs_model.bert
        d1 = mid_dim
        d2 = bert_fs_model.config.hidden_size
        self.fusion_classifier = nn.Linear(d1 + d2, num_classes)

    def forward(self, main_numeric, missing_indicator, categorical, input_ids, attention_mask):
        _, extra_mid = self.extra_net(main_numeric, missing_indicator, categorical)
        bert_outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = bert_outputs.last_hidden_state[:, 0, :]
        logits = self.fusion_classifier(torch.cat([extra_mid, bert_cls], dim=1))
        return logits



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mid_dim = 762
num_classes = 2
num_original = 53
num_categorical = 72

dummy_extra = IPOClassifier(num_original, num_categorical, mid_dim, num_classes)
dummy_bert = BertForSequenceClassification.from_pretrained("bert", num_labels=num_classes)


fusion_model = CombinedModel(dummy_extra, dummy_bert, mid_dim, num_classes)
fusion_model.load_state_dict(torch.load("fusion_model.pth", map_location=device))
fusion_model.to(device).eval()

print("✅ 模型加载成功，可直接推理")

```

**Tips / 提示**

* If you need a single-sample demo, pass shape `(1, 762)` and `(1, 768)` arrays.  
* 若做单样本预测，请传入形状分别为 `(1, 762)` 与 `(1, 768)` 的数组。

---

## 3) Data Sources / 数据来源

**EN**

* **IPO prospectus text** (used by BERT branch): official prospectuses of Shanghai Stock Exchange A-share IPOs.  
* **Financial & market indicators** (used by MLP branch): public disclosures and databases used to construct structured features (including missingness flags).  
* Historical **open/close prices & volumes** for the first 5 trading days post-listing (for label construction, DTW clustering, and evaluation).

**ZH**

* **招股书文本**（用于 BERT 分支）：上交所 A 股 IPO 招股说明书等官方文本。  
* **财务与市场结构化指标**（用于 MLP 分支）：基于公开披露与数据库整理（含缺失指示列）。  
* 上市后 **前 5 个交易日的开/收盘价与成交量**（用于标签构造、DTW 聚类与评估）。

> 示例参考来源（论文中提及）：**上交所**、**国泰安（CSMAR/GTJA 等数据库）** 及公开公司文件。  
> Note: Ensure you have proper data licenses and comply with exchange/database terms.

---
```
