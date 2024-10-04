import math
import re
import random
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
import os
from pathlib import Path
#import torch.utils.data import Dataset  

class RandomWalkDataset(Dataset):
  def __init__(self,sentences,token_list,word_dict,number_dict,vocab_size,maxlen,max_pred):
    self.sentences= sentences
    self.token_list = token_list
    self.word_dict = word_dict
    self.number_dict = number_dict
    self.vocab_size = vocab_size
    self.maxlen = maxlen
    self.max_pred = max_pred
  def __len__(self):
    return len(self.sentences)
  def __getitem__(self,idx):
    #get random sentence pair
    tokens_a_index =  randrange(len(self.sentences)) #randomly pick a sentence for A
    tokens_b_index= randrange(len(self.sentences)) #randomly pick a sentence for B
    #get the token lists corr to A and B
    tokens_a =  self.token_list[tokens_a_index] #get list of tokens for corr A sentence
    tokens_b= self.token_list[tokens_b_index] #get list of tokens for corr B sentence
    input_ids = [self.word_dict['[CLS]']] + tokens_a + [self.word_dict['[SEP]']] + tokens_b + [self.word_dict['[SEP]']]
    #input ids and segment ids containining tokens pertaining to each sentence, 0 for A and 1 for B
    segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
    n_pred =  min(self.max_pred, max(1, int(round(len(input_ids) * 0.15)))) # 15 % of tokens in one sentence, will always be 2 in this case

    cand_masked_pos = [i for i, token in enumerate(input_ids)
                      if token != self.word_dict['[CLS]'] and token != self.word_dict['[SEP]']]  #candidates for masking, cant be CLS or SEP
    shuffle(cand_masked_pos)    #shuffles the list in-place
    masked_tokens, masked_pos, output_labels = [], [], []
    for pos in cand_masked_pos[:n_pred]:
      masked_pos.append(pos) #stores pos within input ids of token to be masked
      masked_tokens.append(input_ids[pos]) #stores the actual token which is masked...used for comparison/loss
      if random() < 0.8:  # 80% of the time we create a mask at pos
        #output_labels.append(input_ids[pos])
        input_ids[pos] = self.word_dict['[MASK]'] # make mask
      elif random() < 0.1:  # 10%
        index = randint(0, self.vocab_size - 1) # random index in vocabulary
        #output_labels.append(input_ids[pos])
        input_ids[pos] = self.word_dict[self.number_dict[index]] # we intentionally replace token at pos with a wrong token
      else:
        pass
    # Zero Paddings, add padding where necessary to have sentences of uniform length
    n_pad = self.maxlen - len(input_ids)
    input_ids.extend([0] * n_pad)
    segment_ids.extend([0] * n_pad)
    # Zero Padding (100% - 15%) tokens
    if self.max_pred > n_pred:
      n_pad = self.max_pred - n_pred
      masked_tokens.extend([0] * n_pad)
      masked_pos.extend([0] * n_pad)
    if tokens_a_index + 1 == tokens_b_index: #if B comes directly after A, nsp = True
      is_next_label = 1 # IsNext
    elif tokens_a_index + 1 != tokens_b_index:
      is_next_label = 0 # NotNext
    output = {"bert_input": input_ids,
              "bert_label": masked_tokens,
              "segment_label": segment_ids,
              "is_next": is_next_label,
              "masked_pos":masked_pos}

    return {key: torch.tensor(value) for key, value in output.items()}