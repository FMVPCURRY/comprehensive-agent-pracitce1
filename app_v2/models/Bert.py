# coding: UTF-8
import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class Config(object):

    def __init__(self, dataset, embedding):
        self.model_name = 'Bert'
        self.train_path = './dataset/dialogue_binary/train.tsv'
        self.dev_path = './dataset/dialogue_binary/dev.tsv'
        self.test_path = './dataset/dialogue_binary/test.tsv'
        self.class_list = [x.strip() for x in open(
            './dataset/dialogue_binary/class.txt', encoding='utf-8').readlines()]
        self.save_path = os.path.join('./saved_dict', dataset, self.model_name + '.ckpt')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 5000
        self.eval_steps = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 8
        self.batch_size = 32
        self.pad_size = 512
        self.learning_rate = 5e-5
        self.bert_path = 'bert-base-chinese'
        self.tokenizer = None
        self.reload_tokenizer()
        self.hidden_size = 768
        self.vocab_path = './new_vocab.txt'
        self.log_path = os.path.join('./log', dataset, self.model_name)

    def reload_tokenizer(self):
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path, local_files_only=True)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, local_files_only=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        outputs = self.bert(input_ids=context, attention_mask=mask)
        pooled = outputs.pooler_output
        out = self.fc(pooled)
        return out
