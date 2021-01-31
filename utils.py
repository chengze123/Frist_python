#!/usr/bin/python
# -*- coding: UTF-8 -*-
#个人微信 wibrce
#Author 杨博
from tqdm import tqdm
import torch
import time
from datetime import timedelta
import pickle as pkl
import os

PAD, CLS = '[PAD]', '[CLS]'

#返回列表，每个元素为元组(token_ids,label,seq_len,mask)
def load_dataset(file_path, config):
    """
    返回结果 4个list ids, lable, ids_len, mask
    :param file_path:
    :param seq_len:
    :return:
    """
    contents = []
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            line = line.strip() #去除开头结尾的空白
            if not line:
                continue
            content, lable = line.split('\t')
            token = config.tokenizer.tokenize(content) #切成字
            token = [CLS] +token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token) #按内置字典编码

            pad_size = config.pad_size

            #token_id以pad_size为标准，多截少补。 mask以有token_ids的地方为1，补出来地方为0
            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids = token_ids + ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((token_ids, int(lable), seq_len, mask))
    return contents

def bulid_dataset(config):
    """
    返回值 train, dev ,test
    :param config:
    :return:
    """
    # 不要每次都加载数据，加载过一次之后就保存为pkl数据，之后就直接用，具体见62课
    if os.path.exists(config.datasetpkl):
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        train = dataset['train']
        dev = dataset['dev']
        test = dataset['test']
    else:
        train = load_dataset(config.train_path, config) #(token_ids, int(lable), seq_len, mask)
        dev = load_dataset(config.dev_path, config)
        test = load_dataset(config.test_path, config)
        dataset = {}
        dataset['train'] = train
        dataset['dev'] = dev
        dataset['test'] = test
        pkl.dump(dataset, open(config.datasetpkl, 'wb'))
    return train, dev, test

class DatasetIterator(object):
    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size #128
        self.dataset = dataset
        self.n_batches = len(dataset) // batch_size  #整数除法，向下取整
        self.residue = False #记录batch数量是否为整数，residue应该是余数
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas): #将list[(token_ids, int(lable), seq_len, mask)]转换为tensor的x,y,seq_len,mask
        x = torch.LongTensor([item[0] for item in datas]).to(self.device) #样本数据ids
        y = torch.LongTensor([item[1] for item in datas]).to(self.device) #标签数据label

        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device) #每一个序列的真实长度
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)

        return (x, seq_len, mask), y

    def __next__(self):
        # 最后一组全输出来
        if self.residue and self.index == self.n_batches:
            # 这一行，加上下一行，代表舍弃最后一批次，因为这一批次可能不足一个batch_size，导致维度对不上
            self.index = 0
            raise StopIteration

            #下面四行，代表最后一批次有多少上多少，和上面两行必须注释掉一个
            # batches = self.dataset[self.index * self.batch_size : len(self.dataset)]
            # self.index += 1
            # batches = self._to_tensor(batches)
            # return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration

        else:
            batches = self.dataset[self.index * self.batch_size : (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def bulid_iterator(dataset, config):
    iter = DatasetIterator(dataset, config.batch_size, config.device) #注意，此处为实例化，不是函数调用
    return iter

def get_time_dif(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))