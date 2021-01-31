#!/usr/bin/python
# -*- coding: UTF-8 -*-
#个人微信 wibrce
#Author 杨博

import time
import torch
import numpy as np
from importlib import import_module
import argparse
import utils
import train
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Bruce-Bert-Text-Classsification')
# BruceERNIEDPCNN
parser.add_argument('--model', type=str, default='ERNIE_cls_si_fc1', help = 'choose a model：ERNIE_cls_fc1, ERNIE_reshape_fc1, ERNIE_reshape_fc0')
args = parser.parse_args()


if __name__ == '__main__':
    # dataset = 'THUCNews'
    # dataset = 'TOUTIAONews'
    # dataset = 'weibo_senti_100k'
    # dataset = 'simplifyweibo_4_moods'
    # dataset = 'Chinese_conversation_sentiment-master'
    # dataset = 'NLPCC2017'
    dataset = 'testtt'
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    np.random.seed(2)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True #保证每次运行结果一样

    start_time = time.time()
    print('加载数据集')
    train_data, dev_data, test_data = utils.bulid_dataset(config)
    train_iter = utils.bulid_iterator(train_data, config)
    dev_iter = utils.bulid_iterator(dev_data, config)
    test_iter = utils.bulid_iterator(test_data, config)

    time_dif = utils.get_time_dif(start_time)
    print("模型开始之前，准备数据时间：", time_dif)

    # 模型训练，评估与测试
    model = x.Model(config).to(config.device)
    train.train(config, model, train_iter, dev_iter, test_iter)
    # train.test(config, model, test_iter)