# coding: UTF-8
import warnings
warnings.filterwarnings("ignore")
import time
import torch
import numpy as np
from train_eval import train,eval, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
# from pytorch_pretrained_bert import BertTokenizer
from transformers import BertTokenizer
from model import Model


class Config(object):

    """配置参数"""
    def __init__(self,dataset):
        self.model_name = 'bert'
        self.train = True
        self.train_path = dataset + '/train_sentence.json'
        self.dev_path = dataset + '/dev_sentence.json'
        self.test_path = dataset + '/test_sentence.json'
        self.save_path = dataset +'/result/'

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 50                                            # epoch数
        self.batch_size =64                                           # mini-batch大小
        self.pad_size = 128                                             # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.rnn_hidden = 768
        self.num_layers = 1
        self.dropout = 0.1


if __name__ == '__main__':
    dataset = 'data'  # 数据集

    config=Config(dataset)

    # np.random.seed(11)
    # torch.manual_seed(11)
    # torch.cuda.manual_seed_all(11)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data = build_dataset(config.train_path,config)
    dev_data = build_dataset(config.dev_path, config)
    test_data = build_dataset(config.test_path, config)

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    
    if config.train:
        # train
        model = Model(config).to(config.device)
        train(config, model, train_iter, dev_iter)
    else:
        # test
        model=torch.load(config.save_path+'latest_model.pt')
        eval(model,test_iter,'test')
    
