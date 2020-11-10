import pandas as pd
from sklearn.utils import shuffle  # 用于数据的随机排列，也可不用

if __name__ == '__main__':
    # 此处是读取中文数据，如果是英文数据，编码可能是'ISO 8859-1'
    #pd_all = pd.read_csv("test.csv", sep=',', encoding='utf-8',error_bad_lines=False)
    pd_all = pd.read_csv("train.csv", sep=',', encoding='utf-8', error_bad_lines=False)
    pd_all = pd.read_csv("dev.csv", sep=',', encoding='utf-8', error_bad_lines=False)
    # 打乱数据
    pd_all = shuffle(pd_all)
    # 保存为tsv文件，当然也可以保存为csv文件，二者区别在于sep为'\t'还是','
    #pd_all.to_csv("test.tsv", index=False, sep='\t', encoding='utf-8')
    pd_all.to_csv("train.tsv", index=False, sep='\t', encoding='utf-8')
    pd_all.to_csv("dev.tsv", index=False, sep='\t', encoding='utf-8')