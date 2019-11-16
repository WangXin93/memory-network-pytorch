import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from babi_loader import BabiDataset, pad_collate
from model import DMNPlus
import argparse

parser = argparse.ArgumentParser(description="DMN")
parser.add_argument("--use_cuda", action="store_true")
parser.add_argument("--lr", default=0.001)
parser.add_argument("--batch_size", default=100)
parser.add_argument("--max_hops", default=3)
parser.add_argument("--embedding_dim", default=20)
parser.add_argument("--hidden_size", default=30)
parser.add_argument("--dataset_dir", default="../bAbI/tasks_1-20_v1-2/en-10k/")
parser.add_argument("--task", type=int, default=1)
parser.add_argument("--random_state", default=2033)
parser.add_argument("--weight_decay", default=0.001)
parser.add_argument("--epochs", default=100)
args = parser.parse_args()

dset = BabiDataset(dataset_dir=args.dataset_dir, task_id=args.task)

vocab_size = len(dset.QA.VOCAB)
torch.manual_seed(args.random_state)
model = DMNPlus(args.hidden_size, vocab_size=vocab_size, num_pass=args.max_hops, qa=dset.QA).cuda()

optims = [
    torch.optim.AdamW([p for name, p in model.named_parameters() if 'embedding' not in name], weight_decay=args.weight_decay),
    torch.optim.SparseAdam([p for name, p in model.named_parameters() if 'embedding' in name])
]

for epoch in range(1, args.epochs+1):
    # train single epoch
    model.train()
    total_acc = 0
    cnt = 0
    dset.set_mode('train')
    train_loader = DataLoader(dset, batch_size=100, shuffle=True, collate_fn=pad_collate)
    for step, (story, query, answer) in enumerate(train_loader):
        story, query, answer = story.long().cuda(), query.long().cuda(), answer.long().cuda()
        for opt in optims: opt.zero_grad()
        loss, acc = model.loss(story, query, answer)
        loss.backward()
        for opt in optims: opt.step()
        total_acc += acc * 100
        cnt += 100
    print("Epoch: {} Loss: {:.4f} Train Acc:{:.4f}".format(epoch, loss.item(), total_acc / cnt), end=" ")

    # evaluate single epoch
    model.eval()
    dset.set_mode('test')
    test_loader = DataLoader( dset, batch_size=100, shuffle=False, collate_fn=pad_collate)
    total_acc = 0
    cnt = 0
    for step, (story, query, answer) in enumerate(test_loader):
        with torch.no_grad():
            story, query, answer = story.long().cuda(), query.long().cuda(), answer.long().cuda()
            loss, acc = model.loss(story, query, answer)
            total_acc += acc * 100
            cnt += 100
    print("Test Acc: {:.4f}".format(total_acc / cnt))
