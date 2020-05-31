import time
import torch.nn as nn
import torch
import numpy as np 
from train_eval import train, test
import argparse, sys, os
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Chinese Dialogs Generation')
parser.add_argument('--model', type=str, required=True, help='choose a model: Transformer, RNN, Topic_Based')
parser.add_argument('--do_train', type=bool, default=False, required=False, help='Train model')
parser.add_argument('--do_test', type=bool, default=False, required=False, help='Test model')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = 'C:/Users/USER/Documents/Capstone_Project/datalogs'  # 数据集

    p = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    if p not in sys.path:
        sys.path.append(p)

    model_name = args.model
    do_train = args.do_train 
    do_test = args.do_test 
    if (do_train or do_test) == False:
        raise ValueError('At lest one of `do_train` or `do_test` muest be True.')
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
    if do_test:
        test_iter = build_iterator(test_data, config, do_dev=True)

    time_dif = get_time_dif(start_time)

    model = x.Seq2SeqModel(config).to(config.device)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    if do_train:
        train(config, model, train_iter, dev_iter)
    if do_test:
        test(config, model, test_iter)

