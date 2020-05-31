# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import os
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam
from keras.utils import plot_model


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


def train(config, model, train_iter, dev_iter):
    start_time = time.time()
    
    if os.path.exists(config.save_path):
        model.load_state_dict(torch.load(config.save_path)['model_state_dict'])

    model.train()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                    lr=config.learning_rate,
                    warmup=0.05,
                    t_total=len(train_iter) * config.num_epochs)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)
    
    if os.path.exists(config.save_path):
        optimizer.load_state_dict(torch.load(config.save_path)['optimizer_state_dict'])
    
    total_batch = 0  
    dev_best_loss = float('inf')
    dev_last_loss = float('inf')
    no_improve = 0
    flag = False

    model.train()
    # plot_model(model, to_file= config.save_dic+'.png')
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                train_loss = loss.item()
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    state = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    dev_best_loss = dev_loss

                    torch.save(state, config.save_dic + str(total_batch) + '.pth')
                    improve = '*'
                    del state
                else:
                    improve = ''

                if dev_last_loss > dev_loss:
                    no_improve = 0
                elif no_improve % 2 == 0:
                    no_improve += 1
                    scheduler.step()
                else:
                    no_improve += 1

                dev_last_loss = dev_loss
                
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, train_loss, train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if no_improve > config.require_improvement:
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break

def test(config, model, test_iter):
    # test
    checkpoint = torch.load(config.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    with open(config.save_dic + 'report.txt', 'w') as f:
        f.write(msg.format(test_loss, test_acc))
        f.write('\n')
        f.write("Precision, Recall and F1-Score...")
        f.write(str(test_report))
        f.write('\n')
        f.write("Confusion Matrix...\n")
        f.write(str(test_confusion))
    
def predict(config, model, data_iter):
    checkpoint = torch.load(config.save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    start_time = time.time()
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    predict_all_sec = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predic_sec = torch.topk(outputs.data, 2, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            if predict_all_sec.size == 0:
                predict_all_sec = predic_sec
            else:
                predict_all_sec = np.concatenate((predict_all_sec, predic_sec),axis=0)
    
    acc = metrics.accuracy_score(labels_all, predict_all)
    print("acc:", acc)
    labels_all = list(labels_all)
    predict_all_sec = list(predict_all_sec)
    total = len(labels_all)
    correct = 0
    for i, j in zip(labels_all, predict_all_sec):
        if i in j:
            correct += 1

    print('second acc', correct/total)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)