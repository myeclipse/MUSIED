# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
# from transformers import BertModel,BertTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer
from utils import all_triggers, trigger2idx, idx2trigger,find_triggers,vec_mat,word2id,id2word


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.config=config

        self.embedding_dim = config.hidden_size
        self.embedding = nn.Embedding(len(word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(vec_mat))
        self.embedding.weight.requires_grad = True


        self.cnn = nn.Conv1d(in_channels=config.hidden_size,
                            out_channels=config.hidden_size,
                            kernel_size=3,
                            stride=1,
                            padding=1)

        self.fc=nn.Sequential(nn.Linear(config.hidden_size*2, 256),
                      nn.Dropout(0.5),
                      nn.Linear(256, len(all_triggers)))
        
        self.device=config.device
        self.dropout = nn.Dropout(config.dropout)

        self.maxpooling = nn.MaxPool1d(128)


    def forward(self, x,label,train=True,condidate_entity=None):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        maskL=x[-3]
        maskR=x[-2]
        sent_ids=x[-1]

        triggers_y_2d = label
        encoder_out = self.embedding(context)
        encoder_out = encoder_out.permute(0, 2, 1)
        encoder_out=self.cnn(encoder_out)
        encoder_out= encoder_out.permute(0, 2, 1)

        # encoder_out=encoder_out[0]
        batchSize=encoder_out.shape[0]
        conved = encoder_out
        conved = conved.transpose(1, 2)
        conved = conved.transpose(0, 1)
        L = (conved * maskL).transpose(0, 1)
        R = (conved * maskR).transpose(0, 1)
        L = L + torch.ones_like(L)
        R = R + torch.ones_like(R)
        pooledL = self.maxpooling(L).contiguous().view(batchSize, self.config.hidden_size)
        pooledR = self.maxpooling(R).contiguous().view(batchSize, self.config.hidden_size)
        pooled = torch.cat((pooledL, pooledR), 1)
        pooled = pooled - torch.ones_like(pooled)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)

        trigger_hat_2d = logits.argmax(-1)
        

        return logits,trigger_hat_2d,triggers_y_2d,sent_ids
