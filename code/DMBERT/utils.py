# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import json
from const import TRIGGERS
PAD, CLS ,NONE= '[PAD]', '[CLS]' ,'NONE' # padding符号, bert中综合信息符号

def build_vocab(labels_trigger, BIO_tagging=False):
    all_labels = [PAD,NONE]
    for label in labels_trigger:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)

    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label

all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS,BIO_tagging=True)


def build_dataset(path,config):

    def load_dataset(path, pad_size=64):
        cut_off=pad_size
        contents = []

        with open(path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
            print(len(data))

            for item in data:
                senid=item['id']
                sentence=item['sentence'].strip()
                triggleL=item['start']
                triggleR=item['end']

                words=[ sentence[i] for i in range(len(sentence))]
                token=[]
                for w in words:
                    t = config.tokenizer.tokenize(w)
                    if t == []: t = ['[UNK]']
                    token.extend(t)

                textL = token[:triggleL]
                textR = token[triggleL:]
                maskL = [1.0 for i in range(0, len(textL) + 1)] + [0.0 for i in range(0, len(textR) )]
                maskR = [0.0 for i in range(0, len(textL) + 1)] + [1.0 for i in range(0, len(textR) )]

                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                        maskL = maskL + ([0.0] * (pad_size - len(token)))
                        maskR = maskR + ([0.0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                        maskL = maskL[:pad_size]
                        maskR = maskR[:pad_size]

                if triggleL<pad_size-1:
                    label=trigger2idx[item['type']]
                else:
                    label=trigger2idx['NONE']

                contents.append((token_ids,[label],seq_len,mask,token,maskL,maskR,senid))


        return contents
    train = load_dataset(path, config.pad_size)

    return train


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        mask = [_[3] for _ in datas]
        words=[_[4] for _ in datas]
        maskL = torch.FloatTensor([_[-3] for _ in datas]).to(self.device)
        maskR = torch.FloatTensor([_[-2] for _ in datas]).to(self.device)

        sen_ids=[_[-1] for _ in datas]

        return (x, seq_len, mask, words, maskL, maskR,sen_ids), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
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


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def find_triggers(labels):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    result_trigger = []
    # result_entities=[]
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result_trigger.append([i, i + 1, labels[i][1]])

    for item in result_trigger:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result_trigger]


def calc_metric(y_true, y_pred,flag):
    """
    :param y_true: [(tuple), ...]
    :param y_pred: [(tuple), ...]
    :return:
    """
    num_proposed = len(y_pred)
    num_gold = len(y_true)

    proposed_precidt={}
    correct_predict={}
    golden={}
    if flag:
        for item in y_true:
            e_type = item[3]
            if e_type not in golden:
                golden[e_type] = 0
            golden[e_type] += 1
        for item in y_pred:
            e_type = item[3]
            if e_type not in proposed_precidt:
                proposed_precidt[e_type] = 0
            proposed_precidt[e_type] += 1
    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1
            if flag:
                e_type=item[3]
                if e_type not in correct_predict:
                    correct_predict[e_type]=0
                correct_predict[e_type]+=1
    print(proposed_precidt)
    print(correct_predict)
    print(golden)
    
    
    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1

    print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1
