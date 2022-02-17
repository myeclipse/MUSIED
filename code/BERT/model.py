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

        #不需要论元信息
        # self.fc_argument = nn.Sequential(nn.Linear(config.hidden_size*2, 256),
        #                     nn.Dropout(0.5),
        #                     nn.Linear(256, len(all_arguments)))
        self.device=config.device

        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden//2, config.num_layers,
                            bidirectional=True, batch_first=True)
        kwargs = dict({'target_size': len(all_triggers), 'device': self.device})
        self.tri_CRF1 = CRF(**kwargs)

    def forward(self, x,label,train=True,condidate_entity=None):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        arguments_2d=x[-1]

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
        # for i in range(batch_size):
        #
        #     predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])

            # golden_entity_tensors = {}
            # for j in range(len(predicted_entities)):
            #     e_start, e_end, e_type_str = predicted_entities[j]
            #     golden_entity_tensors[predicted_entities[j]] = encoder_out[i, e_start:e_end, ].mean(dim=0)


            # for predicted_trigger in predicted_triggers:
            #     t_start, t_end, t_type_str = predicted_trigger
            #
            #     event_tensor = encoder_out[i, t_start:t_end, ].mean(dim=0)
            #     for j in range(len(predicted_entities)):
            #         e_start, e_end, e_type_str = predicted_entities[j]
            #         entity_tensor = golden_entity_tensors[predicted_entities[j]]

                    # argument_hidden.append(torch.cat([entity_tensor,event_tensor]))
                    # argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end, e_type_str))

        # if len(argument_keys) > 0:
        #     argument_hidden = torch.stack(argument_hidden)
        #     argument_hidden_logits = self.fc_argument(argument_hidden)
        #
        #     argument_hidden_hat_1d = argument_hidden_logits.argmax(-1)
        #
        #     arguments_y_1d = []
        #     for i, t_start, t_end, t_type_str, e_start, e_end, e_type_str in argument_keys:
        #         a_label = argument2idx['NONE']
        #         if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
        #             for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
        #                 if e_start == a_start and e_end == a_end:
        #                     a_label = a_type_idx
        #                     break
        #         arguments_y_1d.append(a_label)
        #
        #     arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)

            # batch_size = len(arguments_2d)
            # argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
            # for (i, st, ed, event_type_str, e_st, e_ed, entity_type), a_label in zip(argument_keys,argument_hidden_hat_1d.cpu().numpy()):
            #     if a_label == argument2idx['NONE']:
            #         continue
            #     if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
            #         argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            #     argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, a_label))

            # return trigger_loss,trigger_entities_hat_2d,triggers_y_2d,argument_hidden_logits,arguments_y_1d, argument_hidden_hat_1d, argument_hat_2d,argument_keys

        return logits,trigger_hat_2d,triggers_y_2d,None,None,None,None,None
