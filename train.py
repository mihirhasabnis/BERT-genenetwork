import config
import os 
import re 
from collections import defaultdict
import dataset 
import torch 
import re
import numpy
import random 
import math 
from random import *
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split 
import model 
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import engine

def run():
    with open(config.training_data) as f:
        random_walk_text = f.read().split("\n")

    sentences = []
    for i in range(len(random_walk_text)):
        sentences.append(re.sub("[.,!?\\-]", '',str(random_walk_text[i][1:-2])))
    
    
    gene_dic = defaultdict(int)
    for i in range(len(sentences)):
        for j in range(len(sentences[i].split(" "))):
            gene_dic[sentences[i].split(" ")[j]] = gene_dic.get(sentences[i].split(" ")[j],0)+1

    word_dict = defaultdict(int)
    word_dict['[PAD]'] = 0
    word_dict['[CLS]'] = 1
    word_dict['[SEP]'] = 2
    word_dict['[MASK]'] = 3
    for idx,gene in enumerate(gene_dic):
        word_dict[gene] = idx+4

    vocab_size = len(word_dict)
    #dic mapping index to word/gene
    number_dict = {i: w for i, w in enumerate(word_dict)}
    word_list = list(set(" ".join(sentences).split()))
    #token list  = [[tokens_s1], [tokens_s2],....]
    token_list = list()
    for sentence in sentences:
        arr = [word_dict[s] for s in sentence.split()]
        token_list.append(arr)

    train_data  = dataset.RandomWalkDataset(sentences, token_list,word_dict,number_dict,vocab_size,config.maxlen,config.max_pred)
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True)
    
    #setup 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_1 = model.BERT()

    param_optimizer = list(model_1.named_parameters())
    #no_decay = ["bias","LayerNorm.bias","LayerNorm.weight"]
    # optimizer_parameters = [
    #     {'params':[p for n,p in param_optimizer]}
    # ]
    num_train_steps = int(len(sentences)/config.batch_size*config.epochs)
    optimizer = AdamW(model_1.parameters(), lr = 0.0001)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps = 0,
    #     num_training_steps = num_train_steps
    # )

    for epoch in range(config.epochs):
        engine.train_fn(train_loader, model_1,optimizer,device,epoch)


if __name__ == '__main__':
    run()