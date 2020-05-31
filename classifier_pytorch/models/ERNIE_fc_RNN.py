# coding: UTF-8
import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel, BertTokenizer

class Config(object):

    def __init__(self, use_words, dataset):
        self.model_name = 'ERNIE_fc_RNN'
        self.train_path = dataset + '/data/dataset_classes/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dataset_classes/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/dataset_classes/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/dataset_classes/class.txt').readlines()]                                # 类别名单
        dic = self.model_name + '+word/' if use_words else '+character/'
        self.save_path = dataset + '/saved_dict/classes/' + dic + self.model_name + '.pth'        # 模型训练结果
        self.save_dic = dataset + '/saved_dict/classes/' + dic + self.model_name + '_'      # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
            
        vocabfile = 'vocab_word.pkl' if use_words else 'vocab_char.pkl'
        self.vocab_path = dataset + '/data/dataset_classes/' + vocabfile
        self.require_improvement = 10000                               # 若超过10000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 70                                         # mini-batch大小
        self.pad_size = 30                                            # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4                                       # 学习率
        self.bert_path = 'C:/Users/USER/Documents/Capstone_Project/pretrain_model/ERNIE_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.dropout = 0.1
        self.rnn_hidden = 768
        self.num_layers = 2


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.fc_rnn = nn.Linear(config.rnn_hidden * 2, config.num_classes)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out_RNN, _ = self.lstm(encoder_out)
        out_RNN = self.dropout(out_RNN)
        out_RNN = self.fc_rnn(out_RNN[:, -1, :])  # 句子最后时刻的 hidden state
        out = out_RNN + self.fc(pooled)
        return out

