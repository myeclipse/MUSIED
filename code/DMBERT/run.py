# coding: UTF-8
import json
import time
import torch
import random
import numpy as np
from train_eval import train,eval, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
from transformers import BertTokenizer
from models.dmbert import Model

# parser = argparse.ArgumentParser(description='Chinese Event Extraction')
# parser.add_argument('--model', type=str, default='dmbert', help='choose a model: Bert')
# args = parser.parse_args()




class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/customer/train_sentence_dmcnn.json'
        # self.dev_path = dataset + '/data/customer/test_sentence_dmcnn.json'
        self.test_path = dataset + '/customer/test_sentence_dmcnn.json'
        self.sen_id2_event = dataset + '/customer/test_sentence_dmcnn_id2event.json'
        self.save_path = dataset + '/result/customer'                               # 训练集
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

    # model_name = args.model  # bert
    # x = import_module('models.' + model_name)

    config = Config(dataset)

    with open(config.sen_id2_event,'r') as f:
        sent_id2_event=json.load(f)
    # np.random.seed(11)
    # torch.manual_seed(11)
    # torch.cuda.manual_seed_all(11)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    model = Model(config).to(config.device)
    model=torch.load('latest_model.pt')
    start_time = time.time()
    print("Loading data...")
    #test_data = build_dataset(config.test_path, config)
    train_data = build_dataset(config.train_path, config)
    # dev_data = build_dataset(config.dev_path, config)
    test_data = build_dataset(config.test_path, config)
    #random.shuffle(test_data)
    #print(len(test_data))
    #test_data=test_data[:100000]
    #with open('test','w',encoding='utf-8') as f:
        #json.dump(test_data,f,ensure_ascii=False)
    train_iter = build_iterator(train_data, config)
    # dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    #model = Model(config).to(config.device)
    #model=torch.load('latest_model.pt')
    #eval(model,test_iter,'test',sent_id2_event)
    train(config, model, train_iter, test_iter,sent_id2_event)
