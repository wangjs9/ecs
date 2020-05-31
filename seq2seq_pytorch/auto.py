import time
import torch.nn as nn
import torch
import numpy as np 
from train_eval import train, test
import argparse, sys, os
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif

def main(model_name, iters):
    dataset = 'C:/Users/USER/Documents/Capstone_Project/datalogs'  # 数据集

    p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    if p not in sys.path:
        sys.path.append(p)


    do_train = True
    do_test = False

    i = 0
    while i < iters:
        i += 1
        print(model_name, i)
        print("****************************************************************")

        x = import_module('models.{}'.format(model_name))
        config = x.Config(dataset)
        np.random.seed(156)
        torch.cuda.manual_seed_all(1024)
        torch.backends.cudnn.deterministic = True

        start_time = time.time()
        print('Loading data...')
        train_data, dev_data, test_data = build_dataset(config, do_train, do_test)

        if do_train:
            train_iter = build_iterator(train_data, config)
            dev_iter = build_iterator(dev_data, config, do_dev=True)

        time_dif = get_time_dif(start_time)

        model = x.Seq2SeqModel(config).to(config.device)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        train(config, model, train_iter, dev_iter)

def Test(model_name):
    dataset = 'C:/Users/USER/Documents/Capstone_Project/datalogs'  # 数据集

    p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    if p not in sys.path:
        sys.path.append(p)

    do_train = False
    do_test = True
    
    x = import_module('models.{}'.format(model_name))
    config = x.Config(dataset)
    np.random.seed(156)
    torch.cuda.manual_seed_all(1024)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print('Loading data...')
    train_data, dev_data, test_data = build_dataset(config, do_train, do_test)

    if do_test:
        test_iter = build_iterator(test_data, config, do_dev=True)

    time_dif = get_time_dif(start_time)

    model = x.Seq2SeqModel(config).to(config.device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if do_test:
        test(config, model, test_iter)

if __name__ == '__main__':
    Test('ECS')
