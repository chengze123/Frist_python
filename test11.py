contents = []
with open(r'C:\Users\cat\Desktop\NLPtest\Bruce-Bert-Text-Classification\NLPCC2017\data.txt',\
          'r',encoding='UTF-8') as f:
    for line in f:
        line = line.strip() #去除开头结尾的空白
        if not line: #防止空行
            continue
        label, content = line.split('\t')
        label_dict = {'history':0, 'military':1, 'baby':2, 'world':3, 'tech':4, 'game':5,
                      'society':6, 'sports':7, 'travel':8, 'car':9, 'food':10, 'entertainment':11,
                      'finance':12, 'fashion':13, 'discovery':14, 'story':15, 'regimen':16, 'essay':17
                      }
        label = label_dict[label]
        contents.append(str(content + '\t' + str(label)))

    f.close()

import random
random.shuffle(contents)

# 分成训练，验证，测试 三个列表
train_list = contents[0:int(len(contents)*0.8-1)]
dev_list   = contents[int(len(contents)*0.8):int(len(contents)*0.9-1)]
test_list  = contents[int(len(contents)*0.9):len(contents)-1]

with open(r'C:\Users\cat\Desktop\NLPtest\Bruce-Bert-Text-Classification\NLPCC2017\data\train.txt',\
          "w",encoding='UTF-8') as f:
    for item in train_list:
        f.writelines(item)
        f.writelines('\n')
    f.close()
    
with open(r'C:\Users\cat\Desktop\NLPtest\Bruce-Bert-Text-Classification\NLPCC2017\data\dev.txt',\
          "w",encoding='UTF-8') as f:
    for item in dev_list:
        f.writelines(item)
        f.writelines('\n')
    f.close()

with open(r'C:\Users\cat\Desktop\NLPtest\Bruce-Bert-Text-Classification\NLPCC2017\data\test.txt',\
          "w",encoding='UTF-8') as f:
    for item in test_list:
        f.writelines(item)
        f.writelines('\n')
    f.close()