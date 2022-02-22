# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
# from sklearn import metrics
import time,os
from utils import get_time_dif,calc_metric
# from pytorch_pretrained_bert.optimization import BertAdam
from utils import all_triggers, trigger2idx, idx2trigger,find_triggers


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def eval(model, iterator, fname,sent_id2_event):
    model.eval()
    sent_id2_event_predict={}
    words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
    with torch.no_grad():
        # for i, batch in enumerate(iterator):
        for i, (test, labels) in enumerate(iterator):
            trigger_logits, trigger_hat_2d, triggers_y_2d, sent_ids= model(test, labels)


            # words_all.extend(test[3])
            triggers_all.extend(test[4])
            trigger_hat_2d=trigger_hat_2d.cpu().numpy().tolist()
            triggers_hat_all.extend(trigger_hat_2d)
            for trigger_hat,sen_id in zip(trigger_hat_2d,sent_ids):
                if sen_id not in sent_id2_event_predict:
                    sent_id2_event_predict[sen_id]={}
                    sent_id2_event_predict[sen_id]['seq']=[trigger_hat]
                    sent_id2_event_predict[sen_id]['words'] = test[3]
                else:
                    sent_id2_event_predict[sen_id]['seq'].append(trigger_hat)

    triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []

    for golden_item in sent_id2_event:
        id = golden_item['id']

        triggers_true_ = golden_item['event']
        triggers_true_=[tuple(item) for item in triggers_true_]

        triggers_hat=sent_id2_event_predict[id]['seq']
        triggers_hat = [idx2trigger[hat] for hat in triggers_hat]
        triggers_pred_ = find_triggers(triggers_hat)

        triggers_true.extend([(id, *item) for item in triggers_true_])
        triggers_pred.extend([(id, *item) for item in triggers_pred_])


    print('[trigger classification]')
    trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred,True)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p, trigger_r, trigger_f1))


    print('[trigger identification]')
    triggers_true = [(item[0], item[1], item[2]) for item in triggers_true]
    triggers_pred = [(item[0], item[1], item[2]) for item in triggers_pred]
    trigger_p_, trigger_r_, trigger_f1_ = calc_metric(triggers_true, triggers_pred,False)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p_, trigger_r_, trigger_f1_))

    metric = '[trigger classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p, trigger_r, trigger_f1)

    metric += '[trigger identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p_, trigger_r_, trigger_f1_)
  
    final = fname + ".P%.2f_R%.2f_F%.2f" % (trigger_p, trigger_r, trigger_f1)
    with open(final, 'w',encoding='utf-8') as fout:

        fout.write(metric)

    return metric,trigger_f1


def train(config, model, train_iter, dev_iter,sent_id2_event):
    criterion = nn.CrossEntropyLoss()
    model.train()
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=config.learning_rate,
    #                      warmup=0.05,
    #                      t_total=len(train_iter) * config.num_epochs)
    trigger_F1=0.46
    argument_F1=0

    for epoch in range(config.num_epochs):
        model.train()
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            model.zero_grad()
            trigger_logits, trigger_hat_2d, triggers_y_2d,sent_ids= model(trains,labels)


            trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
            trigger_loss = criterion(trigger_logits, triggers_y_2d.view(-1))

            loss=trigger_loss
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()

            optimizer.step()
            # if i % 100 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


        print(f"=========eval test at epoch={epoch}=========")
        metric_test, trigger_f1 = eval(model, dev_iter, config.save_path+'/'+str(epoch) + '_dev',sent_id2_event)


        if trigger_F1 < trigger_f1:
            trigger_F1 = trigger_f1
            torch.save(model, "latest_model.pt")
   
        print('best trigger F1:')
        print(trigger_F1)

