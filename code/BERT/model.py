# coding: UTF-8
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
# from pytorch_pretrained import BertModel, BertTokenizer
from transformers import BertModel,BertTokenizer
from utils import all_triggers, trigger2idx, idx2trigger,find_triggers
from CRF import CRF




class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc=nn.Sequential(nn.Linear(config.hidden_size, 256),
                      nn.Dropout(0.5),
                      nn.Linear(256, len(all_triggers)))
        
        self.device=config.device

        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden//2, config.num_layers,
                            bidirectional=True, batch_first=True)
        kwargs = dict({'target_size': len(all_triggers), 'device': self.device})
        self.tri_CRF1 = CRF(**kwargs)

    def forward(self, x,label,train=True,condidate_entity=None):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        triggers_y_2d = label
        encoder_out= self.bert(context, attention_mask=torch.BoolTensor(mask).to(self.device))
        encoder_out = encoder_out[0]
        #encoder_out, _ = self.lstm(encoder_out)
        logits=self.fc(encoder_out)

       # trigger_loss = self.tri_CRF1.neg_log_likelihood_loss(feats=out, mask=torch.BoolTensor(mask).to(self.device), tags=triggers_y_2d)
       # _, trigger_hat_2d = self.tri_CRF1.forward(feats=out, mask=torch.BoolTensor(mask).to(self.device))

        trigger_hat_2d = logits.argmax(-1)
        #batch_size = encoder_out.shape[0]
        # argument_hidden,argument_keys = [],[]
        
        return logits,trigger_hat_2d,triggers_y_2d
