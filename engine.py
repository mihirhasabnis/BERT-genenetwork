import torch
import config 
import torch.nn as nn
from pathlib import Path 
from tqdm import tqdm
import numpy as np 
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F



def train_fn(data_loader, model, optimizer, device,epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loop = tqdm(data_loader,leave= True)
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
        loss.backward()
        optimizer.step()
        #scheduler.step()


# def eval_fn(data_loader,model, device):
#     model.eval()
#     with torch.no_grad():
        



