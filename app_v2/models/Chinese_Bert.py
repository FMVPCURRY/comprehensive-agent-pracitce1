# coding: UTF-8
import json
import os
import torch
import torch.nn as nn
from transformers import BertConfig
from models.modeling_glycebert import GlyceBertForSequenceClassification
from tokenizers import BertWordPieceTokenizer

class Config(object):

    def __init__(self, dataset, embedding):
        self.model_name = 'Chinese_Bert'
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
        self.batch_size = 16
        self.pad_size = 512
        self.learning_rate = 5e-5
        self.bert_path = './pretrained/ChineseBERT-base'
        self.hidden_size = 768
        self.log_path = os.path.join('./log', dataset, self.model_name)
        self.tokenizer = None
        self.pinyin_dict = None
        self.id2pinyin = None
        self.pinyin2tensor = None
        self.reload_tokenizer()

    def reload_tokenizer(self):
        vocab_file = os.path.join(self.bert_path, "vocab.txt")
        config_dir = os.path.join(self.bert_path, "config")
        self.tokenizer = BertWordPieceTokenizer(vocab_file)
        with open(os.path.join(config_dir, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        with open(os.path.join(config_dir, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        with open(os.path.join(config_dir, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.bert_dir = config.bert_path
        self.bert_config = BertConfig.from_pretrained(self.bert_dir,
                                                      output_hidden_states=False,
                                                      num_labels=config.num_classes)
        self.model = GlyceBertForSequenceClassification.from_pretrained(self.bert_dir,
                                                                        config=self.bert_config)
        
    def forward(self, trains):
        input_ids, pinyin_ids = trains[0][0], trains[0][1]
        attention_mask = (input_ids != 0).long()
        outputs = self.model(input_ids, pinyin_ids, attention_mask=attention_mask)
        y_logits = outputs[0].view(-1, self.config.num_classes)
        return y_logits
