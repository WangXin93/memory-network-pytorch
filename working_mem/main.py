import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from babi_loader import BabiDataset, pad_collate
from model import WMN
import argparse

parser = argparse.ArgumentParser(description="Working Memory Network")
parser.add_argument("--use_cuda", action="store_true")
parser.add_argument("--lr", default=0.001)
parser.add_argument("--batch_size", default=100)
parser.add_argument("--max_hops", default=3)
parser.add_argument("--num_head", default=4)
parser.add_argument("--embed_size", default=30)
parser.add_argument("--hidden_size", default=60)
parser.add_argument("--dataset_dir", default="../bAbI/tasks_1-20_v1-2/en-10k/")
parser.add_argument("--task", type=int, default=1)
parser.add_argument("--random_state", default=2033)
parser.add_argument("--epochs", default=100)
parser.add_argument("--weight_decay", default=0.001)
args = parser.parse_args()

device=torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")

dset = BabiDataset(dataset_dir=args.dataset_dir, task_id=args.task)
vocab_size = len(dset.QA.VOCAB)

torch.manual_seed(args.random_state)
model = WMN(vocab_size=vocab_size, embed_size=args.embed_size, hidden_size=args.hidden_size,
    max_hops=args.max_hops, num_head=args.num_head, seqend_idx=1, pad_idx=0).cuda()
criterion = nn.CrossEntropyLoss(reduction='sum')
optims = [
    torch.optim.AdamW([p for name, p in model.named_parameters() if 'embedding' not in name], weight_decay=args.weight_decay),
    torch.optim.SparseAdam([p for name, p in model.named_parameters() if 'embedding' in name]),
]

for epoch in range(1, args.epochs+1):
    # train single epoch
    correct = 0
    cnt = 0
    model.train()
    dset.set_mode('train')
    train_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    for step, (stories, questions, answers) in enumerate(train_loader):
        stories, questions, answers = stories.long().cuda(), questions.long().cuda(), answers.long().cuda()
        for opt in optims: opt.zero_grad()
        logits = model(stories, questions)
        loss = criterion(logits, answers)
        loss.backward()
        for opt in optims: opt.step()
        preds = logits.argmax(dim=1)
        correct += torch.sum(preds == answers).item()
        cnt += args.batch_size
    acc = correct / cnt
    print("Epoch: {} Loss: {:.4f} Train Acc:{:.4f}".format(epoch, loss.item(), acc), end=" ")

    # evaluate single epoch
    correct = 0
    cnt = 0
    model.eval()
    dset.set_mode('test')
    test_loader = DataLoader(dset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
    for step, (stories, questions, answers) in enumerate(test_loader):
        with torch.no_grad():
            stories, questions, answers = stories.long().cuda(), questions.long().cuda(), answers.long().cuda()
            logits = model(stories, questions)
            preds = logits.argmax(dim=1)
            correct += torch.sum(preds == answers).item()
            cnt += args.batch_size
    acc = correct / cnt
    print("Test Acc: {:.4f}".format(acc))
