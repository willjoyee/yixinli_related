#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/2/13 13:05
# @Author : AwetJodie


import torch

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# from transformers import BertTokenizer
# from torch import nn
# from transformers import BertModel
# from torch.optim import Adam
# from tqdm import tqdm
# import sys

# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, df):
#         # self.labels = [labels[label] for label in df['category']]
#         self.labels = [float(label) for label in df['warmth']]
#         self.texts = [text for text in df['sentence']]

#     def __len__(self):
#         return len(self.labels)
#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         label = self.labels[idx]
#         # 使用tokenizer对文本进行编码
#         encoding = tokenizer.encode_plus(text,
#                                          padding='max_length',
#                                          max_length=128,
#                                          truncation=True,
#                                          return_tensors="pt")
#         # 将编码结果转换为张量
#         input_ids = encoding['input_ids'].squeeze(0)
#         attention_mask = encoding['attention_mask'].squeeze(0)
#         return input_ids, attention_mask, label
#         # 这三行是干嘛的？？ 参考poe


# # construct the bert model
# class BertClassifier(nn.Module):
#     def __init__(self, dropout=0.5):
#         super(BertClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-chinese')
#         self.dropout = nn.Dropout(dropout)
#         # self.linear_professional = nn.Linear(768, 1)  # 用于专业度的线性层
#         self.linear_warmth = nn.Linear(768, 1)  # 用于温暖程度的线性层
#         self.relu = nn.ReLU()

#     def forward(self, input_ids, attention_mask):
#         _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
#         dropout_output = self.dropout(pooled_output)
#         # professional_output = self.linear_professional(dropout_output)
#         warmth_output = self.linear_warmth(dropout_output)
#         # professional_score = self.relu(professional_output)
#         warmth_score = self.relu(warmth_output)
#         # professional_score = F.sigmoid(professional_output)
#         warmth_score = F.sigmoid(warmth_output)
#         return warmth_score

# # train the bert model
# def train(model, train_data, val_data, learning_rate, epochs):
#     # 通过Dataset类获取训练和验证集
#     train, val = Dataset(train_data), Dataset(val_data)
#     # DataLoader根据batch_size获取数据，训练时选择打乱样本
#     train_dataloader = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
#     val_dataloader = torch.utils.data.DataLoader(val, batch_size=2)
#     # 判断是否使用GPU
#     # use_cuda = torch.cuda.is_available()
#     # device = torch.device("cuda" if use_cuda else "cpu")
#     use_cuda = torch.cuda.is_available()
#     if not use_cuda:
#         print("没有可用的GPU，停止运算")
#         sys.exit()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     # 定义损失函数和优化器
#     # criterion = nn.CrossEntropyLoss()
#     # optimizer = Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss作为损失函数
#     optimizer = Adam(model.parameters(), lr=learning_rate)

#     if use_cuda:
#         model = model.cuda()
#         criterion = criterion.cuda()
#     # 开始进入训练循环
#     for epoch_num in range(epochs):
#         # 定义两个变量，用于存储训练集的准确率和损失
#         total_loss_train = 0
#         total_acc_train = 0

#         model.train()  # 设置模型为训练模式

#         for input_ids, attention_mask, labels in tqdm(train_dataloader):
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)
#             labels = labels.to(device)

#             optimizer.zero_grad()

#             warmth_score = model(input_ids, attention_mask)

#             # 计算损失
#             # professional_loss = criterion(professional_score.squeeze(1), labels.float())
#             warmth_loss = criterion(warmth_score.squeeze(1), labels.float())
#             # loss = professional_loss + warmth_loss
#             loss = warmth_loss

#             total_loss_train += loss.item()

#             # 计算准确率
#             # professional_pred = torch.sigmoid(professional_score) > 0.5
#             # warmth_pred = torch.sigmoid(warmth_score) > 0.5
#             # acc = ((professional_pred == labels) & (warmth_pred == labels)).sum().item()
#             # total_acc_train += acc

#             # loss.backward()
#             # optimizer.step()
#             # 模型更新
#             model.zero_grad()
#             loss.backward()
#             optimizer.step()
#         # ------ 验证模型 -----------
#         # 定义两个变量，用于存储验证集的准确率和损失
#         total_acc_val = 0
#         total_loss_val = 0
#         model.eval()

#         with torch.no_grad():
#             for input_ids, attention_mask, labels in val_dataloader:
#                 input_ids = input_ids.to(device)
#                 attention_mask = attention_mask.to(device)
#                 labels = labels.to(device)

#                 warmth_score = model(input_ids, attention_mask)

#                 # professional_loss = criterion(professional_score.squeeze(1), labels.float())
#                 warmth_loss = criterion(warmth_score.squeeze(1), labels.float())
#                 # loss = professional_loss + warmth_loss
#                 loss = warmth_loss

#                 total_loss_val += loss.item()

#                 # professional_pred = torch.sigmoid(professional_score) > 0.5
#                 # warmth_pred = torch.sigmoid(warmth_score) > 0.5
#                 # acc = ((professional_pred == labels) & (warmth_pred == labels)).sum().item()
#                 # total_acc_val += acc

#         # avg_loss_train = total_loss_train / len(train_dataloader)
#         # avg_loss_val = total_loss_val / len(val_dataloader)

#         print(
#             f'''Epochs: {epoch_num + 1} 
#               | Train Loss: {total_loss_train / len(train_data): .3f} 
#               | Val Loss: {total_loss_val / len(val_data): .3f} ''')
#               #| Val Accuracy: {total_acc_val / len(val_data): .3f}
#         #| Train Accuracy: {total_acc_train / len(train_data): .3f}


#     # evaluate the bert model
# def evaluate(model, test_data):
#     test = Dataset(test_data)
#     test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     if use_cuda:
#         model = model.cuda()
#     criterion = nn.BCEWithLogitsLoss()
#     total_loss_test = 0
#     # total_acc_test = 0
#     # total_samples = 0
#     with torch.no_grad():
#         for input_ids, attention_mask, labels in test_dataloader:
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)
#             labels = labels.to(device)

#             warmth_score = model(input_ids, attention_mask)
#             # professional_loss = criterion(professional_score.squeeze(1), labels.float())
#             warmth_loss = criterion(warmth_score.squeeze(1), labels.float())
#             # loss = professional_loss + warmth_loss
#             loss = warmth_loss
#             total_loss_test += loss.item()

#             # professional_pred = torch.sigmoid(professional_score) > 0.5
#             # warmth_pred = torch.sigmoid(warmth_score) > 0.5
#             # acc = ((professional_pred == labels) & (warmth_pred == labels)).sum().item()
#             # total_acc_test += acc
#             # total_samples += len(labels)
#     print(f' Test Loss: {total_loss_test / len(test_data): .3f} ')
#     # print(f'Test Accuracy: {total_acc_test / total_samples:.3f}')



# # upload the data
# # bbc_text_df.head()
# train_text_df = pd.read_csv('D:\WUJia\phd-project\YiXinLi\yixinli_data3\sentence\\train_text_warmth_new0224.csv',encoding='gbk')
# df = pd.DataFrame(train_text_df)
# np.random.seed(112)
# df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
# [int(.8*len(df)), int(.9*len(df))])
# print(len(df_train),len(df_val), len(df_test))


# # set the parameters of bert
# EPOCHS = 5
# model = BertClassifier()
# LR = 1e-6
# train(model, df_train, df_val, LR, EPOCHS)
# evaluate(model, df_test)

# class pred_Dataset(torch.utils.data.Dataset):
#     def __init__(self, df2):
#         self.texts = [text for text in df2['sentence']]

#     def __len__(self):
#         return len(self.texts)

#     def __getitem__(self, idx):
#         text = self.texts[idx]
#         encoding = tokenizer.encode_plus(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
#         input_ids = encoding['input_ids'].squeeze(0)
#         attention_mask = encoding['attention_mask'].squeeze(0)
#         return input_ids, attention_mask  # 返回None作为占位符标签
    

# def predict(model, data):
#     dataset = pred_Dataset(data)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)
#     use_cuda = torch.cuda.is_available()
#     device = torch.device("cuda" if use_cuda else "cpu")
#     if use_cuda:
#         model = model.cuda()
#     warmth_scores = []
#     with torch.no_grad():
#         for input_ids, attention_mask, in dataloader:
#             input_ids = input_ids.to(device)
#             attention_mask = attention_mask.to(device)

#             warmth_score = model(input_ids, attention_mask)

#             warmth_scores.extend(warmth_score.tolist())

#     return warmth_scores[0]

 
# pred_text_df = pd.read_excel('D:\WUJia\phd-project\YiXinLi\yixinli_data3\sentence\\predict_warmth_newtext.xlsx')
# df2 = pd.DataFrame(pred_text_df)
# print(df2.head())

# warmth_scores = predict(model, df2)

# # 将原始文本和对应的 warmth_scores 存储到新的 DataFrame
# output_df = pd.DataFrame({'text': df2['sentence'], 'warmth_score': warmth_scores})

# # 将 DataFrame 保存为 CSV 文件
# output_df.to_csv('predict_text_warmth4.csv', index=False)






