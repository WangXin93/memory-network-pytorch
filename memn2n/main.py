import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import bAbIDataset
from model import MemN2N
import argparse

parser = argparse.ArgumentParser(description="MemN2N")
parser.add_argument("--use_cuda", action="store_true")
parser.add_argument("--lr", default=0.01)
parser.add_argument("--decay_ratio", default=0.5)
parser.add_argument("--decay_interval", default=25)
parser.add_argument("--max_grad_norm", default=40.)
parser.add_argument("--batch_size", default=32)
parser.add_argument("--max_hops", default=3)
parser.add_argument("--embedding_dim", default=20)
parser.add_argument("--memory_size", default=50)
parser.add_argument("--dataset_dir", default="../bAbI/tasks_1-20_v1-2/en/")
parser.add_argument("--task", type=int, default=1)
parser.add_argument("--random_state", default=2033)
parser.add_argument("--epochs", default=200)
args = parser.parse_args()

def gradient_noise_and_clip(parameters, noise_stddev=1e-3, max_clip=40.0):
    """ Adding Gradient Noise Improves Learning for Very Deep Networks
    https://arxiv.org/abs/1511.06807

    Note: It helps the training a lot. Without gradient noise, loss could
        be **nan**.
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    device = parameters[0].device
    nn.utils.clip_grad_norm_(parameters, max_clip)
    for p in parameters:
        noise = torch.randn(p.size()) * noise_stddev
        noise = noise.to(device)
        p.grad.data.add_(noise)

def decay_learning_rate(opt, epoch, lr, decay_interval, decay_ratio):
    decay_count = max(0, epoch // decay_interval)
    lr = lr * (decay_ratio ** decay_count)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

device=torch.device("cuda:0" if args.use_cuda and torch.cuda.is_available() else "cpu")

train_data = bAbIDataset(args.dataset_dir, args.task)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_data = bAbIDataset(args.dataset_dir, args.task, train=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
print("Task: {} Train datset size: {}, Test dataset size: {}".format(
        args.task, len(train_data), len(test_data)))

settings = {
    "device": device,
    "num_vocab": train_data.num_vocab,
    "embedding_dim": args.embedding_dim,
    "sentence_size": train_data.sentence_size,
    "max_hops": args.max_hops
}
print("Longest sentence length", train_data.sentence_size)
print("Longest story length", train_data.max_story_size)
print("Average story length", train_data.mean_story_size)
print("Number of vocab", train_data.num_vocab)

torch.manual_seed(args.random_state)
mem_n2n = MemN2N(settings)
criterion = nn.CrossEntropyLoss(reduction='sum')
opt = torch.optim.SGD(mem_n2n.parameters(), lr=args.lr)
print(mem_n2n)

mem_n2n = mem_n2n.to(device)

for epoch in range(1, args.epochs+1):
    # train single epoch
    total_loss = 0.
    correct = 0
    for step, (story, query, answer) in enumerate(train_loader):
        story, query, answer = story.to(device), query.to(device), answer.to(device)
        logits = mem_n2n(story, query)
        preds = logits.argmax(dim=1)
        correct += torch.sum(preds == answer).item()
        opt.zero_grad()
        loss = criterion(logits, answer)
        loss.backward()
        gradient_noise_and_clip(mem_n2n.parameters(),
                                noise_stddev=1e-3, max_clip=args.max_grad_norm)
        opt.step()
        total_loss += loss.item()
    acc = correct / len(train_data)
    print("Epoch: {} Loss: {:.4f} Train Acc:{:.4f}".format(epoch, total_loss, acc), end=" ")
    lr = decay_learning_rate(opt, epoch, args.lr, args.decay_interval, args.decay_ratio)


    # evaluate single epoch
    correct = 0
    for step, (story, query, answer) in enumerate(test_loader):
        with torch.no_grad():
            story, query, answer = story.to(device), query.to(device), answer.to(device)
            logits = mem_n2n(story, query)
            preds = logits.argmax(dim=1)
            correct += torch.sum(preds == answer).item()
    acc = correct / len(test_data)
    print("Test Acc: {:.4f}".format(acc))
