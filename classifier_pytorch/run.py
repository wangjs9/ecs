# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, test, predict, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
parser.add_argument('--do_train', type=bool, default=False, required=False, help='Train model')
parser.add_argument('--do_test', type=bool, default=False, required=False, help='Test model')
parser.add_argument('--do_predict', type=bool, default=False, required=False, help='Test model')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'C:/Users/USER/Documents/Capstone_Project/datalogs'  # 数据集

    model_name = args.model  # bert/renie
    do_train = args.do_train
    do_test = args.do_test
    do_predict = args.do_predict
    if (do_train or do_test or do_predict) == False:
        raise ValueError('At least one of `do_train` or `do_test` or `do_predict` must be True.')
    x = import_module('models.' + model_name)
    config = x.Config(args.word, dataset)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    train_data, dev_data, test_data = build_dataset(config, do_train, do_test or do_predict, args.word)
    
    if do_train:
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
    if do_test:
        test_iter = build_iterator(test_data, config)
    if do_predict:
        data_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    if do_train:
        train(config, model, train_iter, dev_iter)
    if do_test:
        test(config, model, test_iter)
    if do_predict:
        predict(config, model, data_iter)