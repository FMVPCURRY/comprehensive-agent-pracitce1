# coding: UTF-8
import time
import torch
import numpy as np
import os
from train_eval import train, init_network, predict
from importlib import import_module
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Detection')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, FastText, TextRCNN, Transformer, Bert, Chinese_Bert')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
parser.add_argument('--mode', default='train', type=str, help='train or test')
parser.add_argument('--dataset-name', default='ChiFraudDialog', type=str, help='dataset name used for checkpoints and logs')
parser.add_argument('--train-path', default=None, type=str, help='training data path')
parser.add_argument('--dev-path', default=None, type=str, help='validation data path')
parser.add_argument('--test-path', default=None, type=str, help='test data path')
parser.add_argument('--class-path', default=None, type=str, help='class label file path')
parser.add_argument('--bert-path', default=None, type=str, help='pretrained model path')
parser.add_argument('--batch-size', default=None, type=int, help='override batch size')
parser.add_argument('--num-epochs', default=None, type=int, help='override epochs')
parser.add_argument('--pad-size', default=None, type=int, help='override max token length')
parser.add_argument('--learning-rate', default=None, type=float, help='override learning rate')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = args.dataset_name
    model_name = args.model
    embedding = args.embedding
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    elif model_name == 'Bert':
        from utils_bert import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    elif model_name == 'Chinese_Bert':
        from utils_chinesebert import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    if args.train_path:
        config.train_path = args.train_path
    if args.dev_path:
        config.dev_path = args.dev_path
    if args.test_path:
        config.test_path = args.test_path
    if args.class_path:
        config.class_list = [item.strip() for item in open(args.class_path, encoding='utf-8').readlines()]
    if args.bert_path and hasattr(config, 'bert_path'):
        config.bert_path = args.bert_path
        if hasattr(config, 'reload_tokenizer'):
            config.reload_tokenizer()
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.pad_size:
        config.pad_size = args.pad_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    config.num_classes = len(config.class_list)
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    os.makedirs(config.log_path, exist_ok=True)
    os.makedirs('./result', exist_ok=True)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer' and model_name != 'Bert' and model_name != 'Chinese_Bert':
        init_network(model)
    print(model.parameters)
    if args.mode == 'train':
        train(config, model, train_iter, dev_iter, test_iter)
    elif args.mode == 'test':
        model.load_state_dict(torch.load(config.save_path))
        predict(config, model, test_iter)
