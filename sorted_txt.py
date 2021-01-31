import re

list = []
with open(r'C:\Users\cat\Desktop\NLPtest\Bruce-Bert-Text-Classification\TOUTIAONews\data\toutiao_cat_data.txt',\
          'r',encoding='UTF-8') as f:
    for line in f:
        line = line.strip() #去除开头结尾的空白
        if not line: #防止空行
            continue
        _, label, __, content, ___ = line.split('_!_', 4)

        # 这坑爹数据集标签从100-116缺了105和111
        if int(label) < 105:
            label = int(label) - 100
        elif int(label) < 111:
            label = int(label) - 101
        else:
            label = int(label) - 102

        list.append(str(content+'\t'+str(label)))
    f.close()

with open(r'C:\Users\cat\Desktop\NLPtest\Bruce-Bert-Text-Classification\TOUTIAONews\data\toutiao_cat_data_sorted.txt',\
          "w",encoding='UTF-8') as f:
    for item in sorted(list, key = lambda i:int(re.findall('\t(\d+)',i)[0])):
        f.writelines(item)
        f.writelines('\n')
    f.close()

# list = []
# with open(r'C:\Users\cat\Desktop\NLPtest\Bruce-Bert-Text-Classification\TOUTIAONews\data\test_text.txt', 'r',encoding='UTF-8') as f:
#     for line in f:
#         list.append(line.strip())
#
# print(sorted(list, key=lambda i:i[22:24]))