"""
Data util codes based on https://github.com/domluna/memn2n
"""
import os
import re
import random
from itertools import chain
import numpy as np
import torch
import torch.utils.data as data
from six.moves import range, reduce
import numpy as np

def load_task(data_dir, task_id, only_supporting=False):
    """
    Load the nth task. There are 20 tasks in total.
    Returns a tuple containing the training and testing data for the task.
    """
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = "qa{}_".format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data


def tokenize(sent):
    """
    Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    """
    return [x.strip() for x in re.split("(\W+)?", sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    """
    Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    """
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(" ", 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if "\t" in line: # question
            q, a, supporting = line.split("\t")
            q = tokenize(q)
            # a = tokenize(a)
            # answer is one vocab word even if it's actually multiple words
            a = [a]
            substory = None

            # remove question marks
            if q[-1] == "?":
                q = q[:-1]

            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]

            data.append((substory, q, a))
            story.append("")
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    """
    Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length
    tokens will be discarded.
    """
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)


def vectorize_data(data, word_idx, sentence_size, memory_size):
    """
    Vectorize stories and queries.
    If a sentence length < sentence_size, the sentence will be padded with 0's.
    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.
    The answer array is returned as a one-hot encoding.
    """
    S, Q, A = [], [], []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # Make the last word of each sentence the time 'word' which
        # corresponds to vector of lookup table
        for i in range(len(ss)):
            ss[i][-1] = len(word_idx) - memory_size - i + len(ss)

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss); Q.append(q); A.append(y)
    return np.array(S), np.array(Q), np.array(A)

class bAbIDataset(data.Dataset):
    def __init__(self, dataset_dir, task_id=1, memory_size=50, train=True):
        self.train = train
        self.task_id = task_id
        self.dataset_dir = dataset_dir

        train_data, test_data = load_task(self.dataset_dir, task_id)
        data = train_data + test_data

        self.vocab = set()
        for story, query, answer in data:
            self.vocab = self.vocab | set(list(chain.from_iterable(story))+query+answer)
        self.vocab = sorted(self.vocab)
        word_idx = dict((word, i+1) for i, word in enumerate(self.vocab))

        self.max_story_size = max([len(story) for story, _, _ in data])
        self.query_size = max([len(query) for _, query, _ in data])
        self.sentence_size = max([len(row) for row in \
            chain.from_iterable([story for story, _, _ in data])])
        self.memory_size = min(memory_size, self.max_story_size)

        # Add time words/indexes
        for i in range(self.memory_size):
            word_idx["time{}".format(i+1)] = "time{}".format(i+1)

        self.num_vocab = len(word_idx) + 1 # +1 for nil word
        self.sentence_size = max(self.query_size, self.sentence_size) # for the position
        self.sentence_size += 1  # +1 for time words
        self.word_idx = word_idx

        self.mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))

        if train:
            story, query, answer = vectorize_data(train_data, self.word_idx,
                self.sentence_size, self.memory_size)
        else:
            story, query, answer = vectorize_data(test_data, self.word_idx,
                self.sentence_size, self.memory_size)

        self.data_story = torch.LongTensor(story)
        self.data_query = torch.LongTensor(query)
        self.data_answer = torch.LongTensor(np.argmax(answer, axis=1))

    def __getitem__(self, idx):
        return self.data_story[idx], self.data_query[idx], self.data_answer[idx]

    def __len__(self):
        return len(self.data_story)

