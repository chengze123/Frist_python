import pandas as pd
import random

# 打开csv文件，并转换为列表
pd_all = pd.read_csv(r'C:\Users\cat\Desktop\NLPtest\Bruce-Bert-Text-Classification\simplifyweibo_4_moods\data\simplifyweibo_4_moods.csv')
pd_all_1ist = pd_all.values.tolist()

contents = []
for i in pd_all_1ist:
    label, content = i
    contents.append(str(content+ '\t'+ str(label)))

# 打乱数据集
random.shuffle(contents)

# 分成训练，验证，测试 三个列表
train_list = contents[0:int(len(pd_all_1ist)*0.8-1)]
dev_list   = contents[int(len(pd_all_1ist)*0.8):int(len(pd_all_1ist)*0.9-1)]
test_list  = contents[int(len(pd_all_1ist)*0.9):len(pd_all_1ist)-1]

# 写入训练，验证，测试 三个txt中
with open(r'C:\Users\cat\Desktop\NLPtest\Bruce-Bert-Text-Classification\simplifyweibo_4_moods\data\train.txt',\
          "w",encoding='UTF-8') as f:
    for item in train_list:
        f.writelines(str(item))
        f.writelines('\n')
    f.close()

with open(r'C:\Users\cat\Desktop\NLPtest\Bruce-Bert-Text-Classification\simplifyweibo_4_moods\data\dev.txt',\
          "w",encoding='UTF-8') as f:
    for item in dev_list:
        f.writelines(str(item))
        f.writelines('\n')
    f.close()

with open(r'C:\Users\cat\Desktop\NLPtest\Bruce-Bert-Text-Classification\simplifyweibo_4_moods\data\test.txt',\
          "w",encoding='UTF-8') as f:
    for item in test_list:
        f.writelines(str(item))
        f.writelines('\n')
    f.close()
