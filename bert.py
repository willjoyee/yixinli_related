import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from imblearn.over_sampling import SMOTE
from transformers import RobertaTokenizer
# 定义准备数据集的函数
def prepare_dataset(encodings, labels_warm, labels_competence):
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels_warm = torch.tensor(labels_warm).float()  # 假设评分是浮点数
    labels_competence = torch.tensor(labels_competence).float()  # 假设评分是浮点数
    dataset = TensorDataset(input_ids, attention_mask, labels_warm, labels_competence)
    return dataset

# 读取数据
# df_warm = pd.read_excel('train_competence_noid_0323.xlsx')
# df_competence = pd.read_excel('train_warmth_noid_0325.xlsx')

df_competence = pd.read_excel('train_competence_noid_0323.xlsx')
df_warm = pd.read_excel('train_warmth_noid_0325.xlsx')


df = pd.merge(df_warm, df_competence, on='sentence', suffixes=('_warm', '_competence'))

# 划分数据集
train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=(1/2), random_state=42)

# 初始化tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # 假设模型名称为'roberta-chinese-base'
# 定义编码句子的函数
def encode_sentences(sentences):
    return tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

# 对训练集、验证集和测试集进行编码
train_encodings = encode_sentences(train_data['sentence'].tolist())
validation_encodings = encode_sentences(validation_data['sentence'].tolist())
test_encodings = encode_sentences(test_data['sentence'].tolist())


# 将数据转换为Tensor
train_labels_warm = train_data['warmth'].values
train_labels_competence = train_data['competence'].values
validation_labels_warm = validation_data['warmth'].values
validation_labels_competence = validation_data['competence'].values
test_labels_warm = test_data['warmth'].values
test_labels_competence = test_data['competence'].values

# 创建TensorDataset
train_dataset = prepare_dataset(train_encodings, train_labels_warm, train_labels_competence)
validation_dataset = prepare_dataset(validation_encodings, validation_labels_warm, validation_labels_competence)
test_dataset = prepare_dataset(test_encodings, test_labels_warm, test_labels_competence)

# 创建DataLoader
batch_size = 8
train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
validation_loader = DataLoader(validation_dataset, sampler=SequentialSampler(validation_dataset), batch_size=batch_size)
test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)


import torch
import torch.nn as nn
from transformers import BertModel
from torch.optim import Adam,AdamW
import torch.nn.functional as F
from tqdm import tqdm
from transformers import RobertaModel
# from data_loader import train_loader,validation_loader,test_loader

# 定义模型
import torch.nn as nn
from transformers import BertModel

class WarmCompetenceClassifier(nn.Module):
    def __init__(self, bert_model_name='bert-large-uncased'):
        super(WarmCompetenceClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.warm_classifier = nn.Linear(self.bert.config.hidden_size, 6)  # 6个输出对应0-5的评分
        self.competence_classifier = nn.Linear(self.bert.config.hidden_size, 9)  # 9个输出对应0,1,2,3,4,5,2.5,3.5,4.5的评分

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        warm_logits = self.warm_classifier(pooled_output)
        competence_logits = self.competence_classifier(pooled_output)
        return warm_logits, competence_logits


# 初始化模型
device = torch.device('cpu')
model = WarmCompetenceClassifier()
model.to(device)

# 定义损失函数
warm_criterion = nn.CrossEntropyLoss()
competence_criterion = nn.CrossEntropyLoss()

# 定义优化器
optimizer = AdamW(model.parameters(), lr=4e-5)
# optimizer = optim.SGD(bert_classifier_model.parameters(), lr=0.01)

# 定义准确度计算函数
def calculate_accuracy(logits, labels):
    _, preds = torch.max(logits, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# 训练和验证模型
num_epochs = 15
accuracy_list = []
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc='Training'):
        input_ids, attention_mask, warm_labels, competence_labels = [b.to(device) for b in batch]

        optimizer.zero_grad()

        warm_logits, competence_logits = model(input_ids, attention_mask)
        warm_loss = warm_criterion(warm_logits, warm_labels.long())
        competence_loss = competence_criterion(competence_logits, competence_labels.long())
        loss = warm_loss + competence_loss  # 总损失是两个损失的和

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 计算平均损失
    avg_train_loss = train_loss / len(train_loader)

    # 验证模式
    model.eval()
    val_warm_accuracy = 0.0
    val_competence_accuracy = 0.0
    with torch.no_grad():
        for batch in tqdm(validation_loader, desc='Validation'):
            input_ids, attention_mask, warm_labels, competence_labels = [b.to(device) for b in batch]

            warm_logits, competence_logits = model(input_ids, attention_mask)

            val_warm_accuracy += calculate_accuracy(warm_logits, warm_labels.long())
            val_competence_accuracy += calculate_accuracy(competence_logits, competence_labels.long())

    # 计算平均准确率
    avg_val_warm_accuracy = val_warm_accuracy / len(validation_loader)
    avg_val_competence_accuracy = val_competence_accuracy / len(validation_loader)
    accuracy_list.append(avg_val_warm_accuracy)

    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Warm Accuracy: {avg_val_warm_accuracy:.4f}, Val Competence Accuracy: {avg_val_competence_accuracy:.4f}')
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

# # 添加绘图的部分
# plt.plot(range(1, num_epochs+1), accuracy_list)
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.title('BERT Model Accuracy')
# plt.show()

# 测试模型
model.eval()
test_warm_accuracy = 0.0
test_competence_accuracy = 0.0
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing'):
        input_ids, attention_mask, warm_labels, competence_labels = [b.to(device) for b in batch]

        warm_logits, competence_logits = model(input_ids, attention_mask)

        test_warm_accuracy += calculate_accuracy(warm_logits, warm_labels.long())
        test_competence_accuracy += calculate_accuracy(competence_logits, competence_labels.long())

# 计算平均准确率
avg_test_warm_accuracy = test_warm_accuracy / len(test_loader)
avg_test_competence_accuracy = test_competence_accuracy / len(test_loader)

print(f'Test Warm Accuracy: {avg_test_warm_accuracy:.4f}, Test Competence Accuracy: {avg_test_competence_accuracy:.4f}')

print('Training and evaluation complete.')
