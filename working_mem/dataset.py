import os
import re
import torch
import numpy as np
from collections import defaultdict
from functools import reduce
from itertools import chain
from torch.utils.data import DataLoader, Dataset

def load_task(data_dir, task_id, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''
    assert task_id > 0 and task_id < 21

    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    s = 'qa{}_'.format(task_id)
    train_file = [f for f in files if s in f and 'train' in f][0]
    test_file = [f for f in files if s in f and 'test' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    out = []
    for x in re.split('(\w+)?', sent):
        if x is not None and x.strip():
            out.append(x.strip())
    return out


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = str.lower(line)
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line: # question
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            q.append("<EOS>")
            #a = tokenize(a)
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
            story.append('')
        else: # regular sentence
            # remove periods
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            sent.append("<EOS>")
            story.append(sent)
    return data


def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
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
    S = []
    Q = []
    A = []
    for story, query, answer in data:
        ss = []
        for i, sentence in enumerate(story, 1):
            ls = max(0, sentence_size - len(sentence))
            ss.append([word_idx[w] for w in sentence] + [0] * ls)

        # take only the most recent sentences that fit in memory
        ss = ss[::-1][:memory_size][::-1]

        # pad to memory_size
        lm = max(0, memory_size - len(ss))
        for _ in range(lm):
            ss.append([0] * sentence_size)

        lq = max(0, sentence_size - len(query))
        q = [word_idx[w] for w in query] + [0] * lq

        y = np.zeros(len(word_idx) + 1) # 0 is reserved for nil word
        for a in answer:
            y[word_idx[a]] = 1

        S.append(ss)
        Q.append(q)
        A.append(y)
    return np.array(S), np.array(Q), np.array(A)


class bAbIDataset(Dataset):
    """ dataset for bAbI

    Compare to memn2n, this dataset features:
        1. Remove time words in memnen.
        2. Add "<EOS>" at each end of sentence.
        3. Unknown works will be recognized as "<UNK>"
    """
    def __init__(self, dataset_dir, task_id=1, memory_size=50, train=True):
        self.train = train
        self.task_id = task_id
        self.dataset_dir = dataset_dir

        train_data, test_data = load_task(self.dataset_dir, task_id)
        data = train_data + test_data

        # Build dictionary
        self.vocab = set()
        for story, query, answer in data:
            self.vocab = self.vocab | set(list(chain.from_iterable(story))+query+answer)
        self.vocab = sorted(self.vocab)
        self.vocab = ['<PAD>' + '<EOS>', '<UNK>'] + self.vocab
        self.word2idx = defaultdict(lambda : 2)
        self.idx2word = defaultdict(lambda : '<UNK>')
        for i, w in enumerate(self.vocab):
            self.word2idx[w] = i
            self.idx2word[i] = w

        # Get size of the dataset
        self.max_story_size = max([len(story) for story, _, _ in data])
        self.query_size = max([len(query) for _, query, _ in data])
        self.sentence_size = max([len(row) for row in \
            chain.from_iterable([story for story, _, _ in data])])
        self.memory_size = min(memory_size, self.max_story_size)
        self.num_vocab = len(self.word2idx)
        self.sentence_size = max(self.query_size, self.sentence_size) # for the position
        self.mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))

        if train:
            story, query, answer = vectorize_data(train_data, self.word2idx,
                self.sentence_size, self.memory_size)
        else:
            story, query, answer = vectorize_data(test_data, self.word2idx,
                self.sentence_size, self.memory_size)

        self.data_story = torch.LongTensor(story)
        self.data_query = torch.LongTensor(query)
        self.data_answer = torch.LongTensor(np.argmax(answer, axis=1))

    def __getitem__(self, idx):
        return self.data_story[idx], self.data_query[idx], self.data_answer[idx]

    def __len__(self):
        return len(self.data_story)


if __name__ == "__main__":
    # train_data, test_data = load_task("./data/tasks_1-20_v1-2/en/", task_id=20)
    # vocab = sorted(reduce(lambda x, y: x | y, (set(sum(s, []) + q + a) for s, q, a in train_data)))
    # # 0 for pad
    # # 1 for end of sentence
    # # 2 for unknown word
    # vocab = ['<PAD>' + '<EOS>', '<UNK>'] + vocab
    # word2idx = defaultdict(lambda : 2)
    # idx2word = defaultdict(lambda : '<UNK>')
    # for i, w in enumerate(vocab):
    #     word2idx[w] = i
    #     idx2word[i] = w

    # max_story_size = max(map(len, (s for s, _, _ in train_data)))
    # mean_story_size = int(np.mean([ len(s) for s, _, _ in train_data ]))
    # sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in train_data)))
    # query_size = max(map(len, (q for _, q, _ in train_data)))
    # memory_size = max_story_size
    # trainS, trainQ, trainA =  vectorize_data(train_data, word2idx, sentence_size, memory_size)
    train_d = bAbIDataset("./data/tasks_1-20_v1-2/en/", task_id=2, train=True)
    loader = DataLoader(train_d, batch_size=4, shuffle=False)
