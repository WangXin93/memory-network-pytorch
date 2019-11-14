import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda")

class WMN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_hops, num_head, seqend_idx, pad_idx):
        super(WMN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size= hidden_size
        self.max_hops = max_hops
        self.embed_size = embed_size
        self.seqend_idx = seqend_idx
        self.pad_idx = pad_idx

        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.input_gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.w_m = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.Tanh())
        self.max_hops = max_hops
        self.num_head = num_head
        self.attn_ctl = MultiHeadAttention(h=self.num_head, d_model=self.hidden_size)
        self.f_t = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size))
        self.RM = ReasonModule(hidden_size)
        self.V = nn.Linear(hidden_size*3, vocab_size, bias=False)

        self.init_weight()

    def init_hidden(self, batch_size):
        '''GRU的初始hidden。单层单向'''
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        hidden = hidden.to(self.embed.weight.device)
        return hidden

    def init_weight(self):
        nn.init.xavier_uniform_(self.embed.weight)
        components = [self.embed, self.input_gru, self.w_m, self.f_t, self.V, self.RM]
        for component in components:
            for name, param in component.state_dict().items():
                if 'weight' in name:
                    nn.init.xavier_normal_(param)

    def forward(self, story, query):
        """
        Args:
            story: [32, 50, 7]
            query: [32, 7]
        Return:
            answer: [32, num_vocab]
        """
        batch_size, memory_size, sen_size = story.size()
        _, query_size = query.size()

        # Short memory storage
        story_embs = self.embed(story) # [4, 50, 7, 32]
        hidden = self.init_hidden(memory_size)
        memory_slots = []
        for bi, story_emb in enumerate(story_embs):
            outputs, hidden = self.input_gru(story_emb, hidden)
            real_output = []
            for mi, o in enumerate(outputs):
                lst = story[bi][mi].tolist()
                if self.seqend_idx not in lst:
                    real_output.append(o[-1])
                else:
                    real_output.append(o[lst.index(self.seqend_idx)])
            real_output = torch.stack(real_output, dim=0) # [50, 32]
            memory_slots.append(real_output)
        memory_slots = torch.stack(memory_slots, dim=0) # [4, 50, 20]

        # Encode query
        query_embs = self.embed(query)
        hidden = self.init_hidden(batch_size)
        outputs, hidden = self.input_gru(query_embs, hidden)
        real_query = []
        for bi, o in enumerate(outputs):
            lst = query[bi].tolist()
            real_query.append(o[lst.index(self.seqend_idx)])
        real_query = torch.stack(real_query) # [4, 20]

        # Attentional Controller
        o_k = self.f_t(real_query)
        memory_key = memory_slots
        memory_value = memory_slots
        memory_buffers = []
        for hopn in range(self.max_hops):
            memory_buffer = self.attn_ctl(o_k, memory_key, memory_value) # [4, 20]
            memory_buffers.append(memory_buffer)
            o_k = self.f_t(memory_buffer)
        memory_buffers = torch.stack(memory_buffers, dim=1)

        # Reasoning Module
        reasoning = self.RM(memory_buffers, real_query) # [4, 20]

        a_hat = self.V(reasoning)
        return a_hat

class ReasonModule(nn.Module):
    def __init__(self, hidden_size):
        """TODO: to be defined. """
        super(ReasonModule,self).__init__()
        self.g = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size*3),
            nn.ReLU(),
            nn.Linear(hidden_size*3, hidden_size*3)
        )

    def forward(self, inp, question):
        """
        Args:
            inp: [4, 3, 20]
            question: [4, 20]
        """
        batch_size, num_obj, emb_size = inp.size()
        left = inp.unsqueeze(1).repeat(1, num_obj, 1, 1).view(batch_size, num_obj*num_obj, emb_size)
        right = inp.unsqueeze(2).repeat(1, 1, num_obj, 1).view(batch_size, num_obj*num_obj, emb_size)
        question = question.unsqueeze(1).expand_as(left)
        combined = torch.cat([left, right, question], dim=-1) # [4, 9, 60]
        out = F.relu(self.g(combined))
        out = out.sum(dim=1) # [4, 60]
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        Args:
            query: [4, 20]
            key: [4, 50, 20]
            value: [4, 50, 20]
            mask: [4, 50]
        Return:
            out: [4, 20]
            attn_weight: [4, 50]
        """
        d = query.size()[-1]
        scores = torch.matmul(key, query.unsqueeze(-1)) / np.sqrt(d)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            scores = scores.masked_fill(mask == 0, 1e-9)
        attn_weight = torch.softmax(scores, dim=-2)
        if dropout is not None:
            attn_weight = dropout(attn_weight)
        out = torch.sum((attn_weight * value), dim=-2)
        return out, attn_weight.squeeze()

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)

        query = self.linears[0](query).view(nbatches, self.h, self.d_k)
        key = self.linears[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linears[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        out, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        out = out.reshape(nbatches, self.h*self.d_k)
        out = self.linears[3](out)
        return out


if __name__ == "__main__":
    from dataset import bAbIDataset
    from torch.utils.data import DataLoader
    train_d = bAbIDataset("./data/tasks_1-20_v1-2/en/", task_id=2, train=True)
    loader = DataLoader(train_d, batch_size=4, shuffle=False)

    device = torch.device("cuda:0")
    story, query, answer = next(iter(loader))
    story, query = story.to(device), query.to(device)
    m = WMN(vocab_size=train_d.num_vocab, embed_size=32, hidden_size=20, max_hops=3, num_head=4, seqend_idx=2, pad_idx=0)
    m = m.to(device)
    out = m(story, query)

