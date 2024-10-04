# -*- coding: utf-8 -*-
"""Working_Copy of BERT for gene_regulatory_networks.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EmrtSLhgmN6VqvGx-0utEuGUSjF1gEqE
"""

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

with open(r"C:\Downloads\LLM_Sridhar\BERT-genenetwork\rand_walks_samplewise\rand_walk_1.txt") as f:
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

print(list(word_dict.items())[:10])

vocab_size = len(word_dict)
#dic mapping index to word/gene
number_dict = {i: w for i, w in enumerate(word_dict)}
word_list = list(set(" ".join(sentences).split()))
#token list  = [[tokens_s1], [tokens_s2],....]
token_list = list()
for sentence in sentences:
    arr = [word_dict[s] for s in sentence.split()]
    token_list.append(arr)

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
    tokens_a =  token_list[tokens_a_index] #get list of tokens for corr A sentence
    tokens_b= token_list[tokens_b_index] #get list of tokens for corr B sentence
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

print("\n")
train_data = RandomWalkDataset(sentences, token_list,word_dict,number_dict,vocab_size,maxlen=15,max_pred=5)
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
sample_data = next(iter(train_loader))
print(sample_data)

maxlen = 15 # maximum length
batch_size = 6
max_pred = 5  # max tokens of prediction
n_layers = 6 # number of Encoder of Encoder Layer
n_heads = 2 # number of heads in Multi-Head Attention
d_model = 32 # Embedding Size
d_ff = 32 * 4  # 4*d_model, FeedForward dimension
d_k = d_v = 16  # dimension of K(=Q), V, tpyically d_model//n_heads
n_segments = 2

# def make_batch():
#     batch = []
#     positive = 0
#     negative = 0
#     while positive != batch_size/2 or negative != batch_size/2:
#         tokens_a_index =  randrange(len(sentences)) #randomly pick a sentence for A
#         tokens_b_index= randrange(len(sentences)) #randomly pick a sentence for B
#         tokens_a =  token_list[tokens_a_index] #get list of tokens for corr A sentence
#         tokens_b= token_list[tokens_b_index] #get list of tokens for corr B sentence

#         input_ids = [word_dict['[CLS]']] + tokens_a + [word_dict['[SEP]']] + tokens_b + [word_dict['[SEP]']] #list containining tokens pertaining to each sentence

#         segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1) #init to list of zeros, size corr to length of input_ids

#         #MASK LM
#         n_pred =  min(max_pred, max(1, int(round(len(input_ids) * 0.15)))) # 15 % of tokens in one sentence, will always be 2 in this case

#         cand_maked_pos = [i for i, token in enumerate(input_ids)
#                           if token != word_dict['[CLS]'] and token != word_dict['[SEP]']]  #candidates for masking, cant be CLS or SEP
#         shuffle(cand_maked_pos)    #shuffles the list in-place
#         masked_tokens, masked_pos = [], []
#         for pos in cand_maked_pos[:n_pred]:
#             masked_pos.append(pos)
#             masked_tokens.append(input_ids[pos])
#             if random() < 0.8:  # 80% of the time we create a mask at pos
#                 input_ids[pos] = word_dict['[MASK]'] # make mask
#             elif random() < 0.2:  # 10%
#                 index = randint(0, vocab_size - 1) # random index in vocabulary
#                 input_ids[pos] = word_dict[number_dict[index]] # we intentionally replace token at pos with a wrong token

#         # Zero Paddings, add padding where necessary to have sentences of uniform length
#         n_pad = maxlen - len(input_ids)
#         input_ids.extend([0] * n_pad)
#         segment_ids.extend([0] * n_pad)

#         # Zero Padding (100% - 15%) tokens
#         if max_pred > n_pred:
#             n_pad = max_pred - n_pred
#             masked_tokens.extend([0] * n_pad)
#             masked_pos.extend([0] * n_pad)

#         # if positive < batch_size/2:
#         #   batch.append([input_ids, segment_ids, masked_tokens, masked_pos])
#         #   positive += 1
#         if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2: #if B comes directly after A, nsp = True
#             batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True]) # IsNext
#             positive += 1
#         elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
#             batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False]) # NotNext
#             negative += 1
#     return batch

def get_attn_pad_mask(seq_q, seq_k):  ##masking for PAD tokens ie 0 tokens
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k, just creates a new view of the tensor with singleton dim expanded to specified size

def gelu(x): #gaussian activiation function, known to improve performance in transformer models
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)  # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # (seq_len,) -> (batch_size, seq_len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return scores,context, attn

# emb = Embedding()
# embeds = emb(input_ids, segment_ids)

# attenM = get_attn_pad_mask(input_ids, input_ids)

# SDPA= ScaledDotProductAttention()(embeds, embeds, embeds, attenM)

# S,C, A = SDPA

# print('Masks',attenM[0][0])
# # print()
# print('Scores: ', S[0][0],'\n\nAttention M: ', A[0][0])

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores,context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn # output: [batch_size x len_q x d_model]

# emb = Embedding()
# embeds = emb(input_ids, segment_ids)

# attenM = get_attn_pad_mask(input_ids, input_ids)

# MHA = MultiHeadAttention()(embeds, embeds, embeds, attenM)

# Output, A = MHA

# A[0][0]

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn

class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]
        # it will be decided by first token(CLS)
        h_pooled = self.activ1(self.fc(output[:, 0])) # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled) # [batch_size, 2]

        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        # get masked position from final output of transformer.
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_clsf

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = BERT()
model.train()
model.to(device)

train_data = RandomWalkDataset(sentences, token_list,word_dict,number_dict,vocab_size,maxlen=15,max_pred=5)
train_loader = DataLoader(train_data, batch_size=6, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)
epochs = 2
for epoch in range(epochs):
    loop = tqdm(train_loader,leave= True)
    for batch in loop:
      optimizer.zero_grad()
      input_ids = batch['bert_input'].to(device)
      segment_ids = batch['segment_label'].to(device)
      labels = batch['bert_label'].to(device)
      isNext = batch['is_next'].to(device)
      masked_pos = batch['masked_pos'].to(device)
      logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
      loss_lm = criterion(logits_lm.transpose(1, 2), labels) # for masked LM
      loss_lm = (loss_lm.float()).mean()
      loss_clsf = criterion(logits_clsf, isNext) # for sentence classification
      loss = loss_lm + loss_clsf
      loop.set_description(f"epoch{epoch}")
      loop.set_postfix(loss = loss.item())
      # if (epoch + 1) % 10 == 0:
      #   print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
      loss.backward()
      optimizer.step()



# #Predict mask tokens ans isNext
# input_ids, segment_ids, masked_tokens, masked_pos, isNext = map(torch.LongTensor, zip(batch[3]))
# #print(text)
# print([number_dict[w.item()] for w in input_ids[0] if number_dict[w.item()] != '[PAD]'])

# logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
# logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
# print('masked tokens list : ',[pos.item() for pos in masked_tokens[0] if pos.item() != 0])
# print('predict masked tokens list : ',[pos for pos in logits_lm if pos != 0])

# logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
# # print('isNext : ', True if isNext else False)
# # print('predict isNext : ',True if logits_clsf else False)



