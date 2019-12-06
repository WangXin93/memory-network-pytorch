import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_dim+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    return np.transpose(encoding)

class MemN2N(nn.Module):
    def __init__(self, settings):
        super(MemN2N, self).__init__()

        device = settings["device"]
        num_vocab = settings["num_vocab"]
        embedding_dim = settings["embedding_dim"]
        sentence_size = settings["sentence_size"]
        self.max_hops = settings["max_hops"]

        self.C = nn.ModuleList()
        for hop in range(self.max_hops+1):
            m = nn.Embedding(num_vocab, embedding_dim, padding_idx=0)
            m.weight.data.normal_(0, 0.1)
            self.C.append(m)

        self.T_k = nn.Linear(embedding_dim, embedding_dim)

        self.encoding = Variable(torch.FloatTensor(
            position_encoding(sentence_size, embedding_dim)), requires_grad=False)

        self.encoding = self.encoding.to(device)

    def forward(self, story, query):
        u = list()
        query_embed = self.C[0](query)
        # weired way to perform reduce_dot
        encoding = self.encoding.unsqueeze(0).expand_as(query_embed)
        u.append(torch.sum(query_embed*encoding, 1))
        
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story) # [32, 10, 7, 20]
       
            encoding = self.encoding.unsqueeze(0).unsqueeze(1).expand_as(embed_A)
            m_A = torch.sum(embed_A*encoding, 2) # [32, 10, 20]
       
            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob   = torch.softmax(torch.sum(m_A*u_temp, 2), dim=-1)
        
            embed_C = self.C[hop+1](story) # [32, 10, 7, 20]
            m_C  = torch.sum(embed_C*encoding, 2) # [32, 10, 20]
       
            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k  = torch.sum(m_C*prob, 1) # [32, 20]
       
            t_k = torch.sigmoid(self.T_k(u[-1]))
            u_k = (1 - t_k) * u[-1] + o_k * t_k
            u.append(u_k)
       
        a_hat = u[-1]@self.C[self.max_hops].weight.transpose(0, 1)
        return a_hat
