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

        # self.que_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.input_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0, sparse=True)
        nn.init.uniform_(self.input_embedding.state_dict()['weight'], a=-(3**0.5), b=3**0.5)

        self.question_gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.input_gru = nn.GRU(embed_size, hidden_size, batch_first=True)

        self.num_head = num_head
        self.max_hops = max_hops
        self.attn_ctl = nn.ModuleList()
        for _ in range(self.max_hops):
            self.attn_ctl.append(AttentionController(num_head, hidden_size))

        self.f_t = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.g = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size*3),
            nn.ReLU(),
            nn.Linear(hidden_size*3, hidden_size*3),
            nn.ReLU()
        )
        self.V = nn.Linear(hidden_size*3, vocab_size, bias=False)

        def init_weight(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: m.bias.data.fill_(0.01)

        for name, p in self.question_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_uniform_(p)

        for name, p in self.input_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_uniform_(p)

        self.f_t.apply(init_weight)
        self.g.apply(init_weight)
        self.V.apply(init_weight)

        self.dropout = nn.Dropout(0.1)


    def init_hidden(self, batch_size):
        '''GRU的初始hidden。单层单向'''
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        hidden = hidden.to(self.input_embedding.weight.device)
        return hidden

    def RM(self, working_buffers, question_embs):
        """ Reasoning Module
        Args:
            working_buffers: [N, H, D]
            question_embs: [N, H]
        """
        batch_size, num_obj, emb_size = working_buffers.size()

        left = working_buffers.unsqueeze(1).repeat(1, num_obj, 1, 1).view(
            batch_size, num_obj*num_obj, emb_size)
        right = working_buffers.unsqueeze(2).repeat(1, 1, num_obj, 1).view(
            batch_size, num_obj*num_obj, emb_size)
        question_embs = question_embs.unsqueeze(1).expand_as(left)
        combined = torch.cat([left, right], dim=-1) # [4, 9, 60]
        combined = self.dropout(combined)
        combined = torch.cat([combined, question_embs], dim=-1) # [4, 9, 60]
        out = self.g(combined)
        out = out.sum(dim=1) # [4, 60]
        return out

    def input_module(self, stories):
        """ Short memory storage
        """
        batch_size, memory_size, sen_size = stories.size()

        story_embs = self.input_embedding(stories) # [4, 50, 7, 32]
        story_embs = self.dropout(story_embs)
        hidden = self.init_hidden(memory_size)
        memory_storage = []
        for bi, story_emb in enumerate(story_embs):
            outputs, hidden = self.input_gru(story_emb, hidden)
            memory_storage.append(hidden.squeeze())
        memory_storage = torch.stack(memory_storage, dim=0) # [4, 50, 20]
        return memory_storage

    def question_module(self, questions):
        """ Encode question
        """
        batch_size, quz_size = questions.size()

        que_word_embs = self.input_embedding(questions)
        hidden = self.init_hidden(batch_size)
        outputs, hidden = self.question_gru(que_word_embs, hidden)
        return hidden.squeeze(0)

    def forward(self, stories, questions):
        """
        Args:
            stories: [N, T, W]
            questions: [N, T]
        Return:
            answer: [N, num_vocab]
        """
        memory_storage = self.input_module(stories)

        question_embs = self.question_module(questions)

        # Attentional Controller
        o_k = self.f_t(question_embs)
        memory_key = memory_storage
        memory_value = memory_storage
        memory_buffers = []
        for hopn in range(self.max_hops):
            memory_buffer = self.attn_ctl[hopn](o_k, memory_key, memory_value) # [4, 20]
            memory_buffers.append(memory_buffer)
            o_k = self.f_t(memory_buffer)
        memory_buffers = torch.stack(memory_buffers, dim=1) # [4, 8, 20]

        # Reasoning Module
        reasoning = self.RM(memory_buffers, question_embs) # [4, 20]

        a_hat = self.V(reasoning)
        return a_hat

class AttentionController(nn.Module):
    def __init__(self, num_head, hidden_size):
        super(AttentionController, self).__init__()
        self.num_head = num_head
        self.hidden_size = hidden_size
        
        def init_weight(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: m.bias.data.fill_(0.01)

        self.W_m = nn.ModuleList()
        for _ in range(self.num_head):
            W = nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.Tanh()
            )
            W.apply(init_weight)
            self.W_m.append(W)

        self.W_o = nn.Linear(num_head*hidden_size, hidden_size)
        self.W_o.apply(init_weight)

        self.dropout = nn.Dropout(0.1)

    def forward(self, o_k, memory_key, memory_value):
        """
        Args:
            o_k: [N, D]
            memory_key: [N, T, D]
            memory_value: [N, T, D]
        Return:
            out: [N, D]
        """
        o_k = o_k.unsqueeze(-1)
        out = []
        for headn in range(self.num_head):
            projected = self.W_m[headn](memory_key)
            scores = torch.matmul(projected, o_k) # [N, T, 1]
            scores = torch.softmax(scores, dim=-2)
            rep = torch.sum(scores * memory_value, dim=-2) # [N, D]
            out.append(rep)
        out = torch.cat(out, dim=-1)
        # out = self.dropout(out)
        out = self.W_o(out)
        return out

# class MultiHeadAttention(nn.Module):
#     def __init__(self, h, d_model, dropout=0.1):
#         super(MultiHeadAttention, self).__init__()
#         assert d_model % h == 0
#         self.d_k = d_model // h
#         self.h = h
#         self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
#         self.attn = None
#         self.dropout = nn.Dropout(p=dropout)

#     @staticmethod
#     def attention(query, key, value, mask=None, dropout=None):
#         """
#         Args:
#             query: [4, 20]
#             key: [4, 50, 20]
#             value: [4, 50, 20]
#             mask: [4, 50]
#         Return:
#             out: [4, 20]
#             attn_weight: [4, 50]
#         """
#         d = query.size()[-1]
#         scores = torch.matmul(key, query.unsqueeze(-1)) / np.sqrt(d)
#         if mask is not None:
#             mask = mask.unsqueeze(-1)
#             scores = scores.masked_fill(mask == 0, 1e-9)
#         attn_weight = torch.softmax(scores, dim=-2)
#         if dropout is not None:
#             attn_weight = dropout(attn_weight)
#         out = torch.sum((attn_weight * value), dim=-2)
#         return out, attn_weight.squeeze()

#     def forward(self, query, key, value, mask=None):
#         nbatches = query.size(0)
#         if mask is not None:
#             mask = mask.unsqueeze(1)

#         query = self.linears[0](query).view(nbatches, self.h, self.d_k)
#         key = self.linears[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
#         value = self.linears[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

#         out, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
#         out = out.reshape(nbatches, self.h*self.d_k)
#         out = self.linears[3](out)
#         return out


if __name__ == "__main__":
    from babi_loader import BabiDataset, pad_collate
    from torch.utils.data import DataLoader
    
    dset = BabiDataset(dataset_dir="../bAbI/tasks_1-20_v1-2/en/", task_id=2)
    vocab_size = len(dset.QA.VOCAB)
    dset.set_mode('train')
    loader = DataLoader(dset, batch_size=2, shuffle=False, collate_fn=pad_collate)

    s, q, a = next(iter(loader))
    s, q, a = s.long().cuda(), q.long().cuda(), a.long().cuda()

    wmn = WMN(vocab_size=vocab_size, embed_size=32, hidden_size=32, max_hops=3, num_head=8, seqend_idx=1, pad_idx=0).cuda()
    ans = wmn(s, q)

