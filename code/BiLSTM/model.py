# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer
from utils import all_triggers, trigger2idx, idx2trigger,find_triggers,vec_mat,word2id,id2word
from CRF import CRF


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()

        self.embedding_dim = config.hidden_size
        self.embedding = nn.Embedding(len(word2id), self.embedding_dim)
        self.embedding.weight.data.copy_(torch.from_numpy(vec_mat))
        self.embedding.weight.requires_grad = True

        self.fc=nn.Sequential(nn.Linear(config.hidden_size, 128),
                      nn.Dropout(0.5),
                      nn.Linear(128, len(all_triggers)))

        self.device=config.device

        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden//2, config.num_layers,
                            bidirectional=True, batch_first=True)
        kwargs = dict({'target_size': len(all_triggers), 'device': self.device})
        self.tri_CRF1 = CRF(**kwargs)

    def forward(self, x,label,train=True,condidate_entity=None):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]


        triggers_y_2d = label
        encoder_out=self.embedding(context)

        encoder_out, _ = self.lstm(encoder_out)
        out=self.fc(encoder_out)

        #trigger_loss = self.tri_CRF1.neg_log_likelihood_loss(feats=out, mask=torch.BoolTensor(mask).to(self.device), tags=triggers_y_2d)
        #_, trigger_hat_2d = self.tri_CRF1.forward(feats=out, mask=torch.BoolTensor(mask).to(self.device))

        trigger_hat_2d = out.argmax(-1)
        

        return out,trigger_hat_2d,triggers_y_2d
