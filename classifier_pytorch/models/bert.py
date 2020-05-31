# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained_bert import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, use_words, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/dataset_classes/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dataset_classes/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/dataset_classes/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/dataset_classes/class.txt').readlines()]
        dic = self.model_name + '+word/' if use_words else '+character/'
        self.save_path = dataset + '/saved_dict/classes/' + dic + self.model_name + '.pth'        # 模型训练结果
        self.save_dic = dataset + '/saved_dict/classes/' + dic + self.model_name + '_'      # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        vocabfile = 'vocab_word.pkl' if use_words else 'vocab_char.pkl'
        self.vocab_path = dataset + '/data/dataset_classes/' + vocabfile
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 3                                             # epoch数
        self.batch_size = 84                                           # mini-batch大小
        self.pad_size = 32                                           # 每句话处理成的长度(短填长切)
        self.learning_rate = 5e-4                                       # 学习率
        self.bert_path = 'C:/Users/USER/Documents/Capstone_Project/pretrain_model/bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
