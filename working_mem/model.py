import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda")

class WMN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, seqend_idx, pad_idx):
        super(WMN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size= hidden_size
        self.embed_size = embed_size
        self.seqend_idx = seqend_idx
        self.pad_idx = pad_idx

        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.input_gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.w_m = nn.Linear(hidden_size, hidden_size, bias=False)
        num_head = 3
        self.max_hops = 3
        self.attn_ctl = Attention()
        self.f_t = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.RM = ReasonModule(hidden_size)
        self.V = nn.Linear(hidden_size, vocab_size, bias=False)

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
                    nn.init.xavier_normal(param)

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
        o_k = real_query
        memory_key = self.w_m(memory_slots)
        memory_value = memory_slots
        memory_buffers = []
        for hopn in range(self.max_hops):
            memory_buffer, _ = self.attn_ctl(o_k, memory_key, memory_value) # [4, 20]
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
        self.f = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size*3),
            nn.ReLU(),
            nn.Linear(hidden_size*3, hidden_size)
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
        out = self.f(out) # [4, 60]
        return out

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask=None, dropout=None):
        """
        Args:
            query: [4, 20]
            key: [4, 50, 20]
            value: [4, 50, 20]
        Return:
            out: [4, 20]
            attn_weight: [4, 50]
        """
        d = query.size()[-1]
        scores = torch.matmul(key, query.unsqueeze(-1)) / np.sqrt(d)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 1e-9)
        attn_weight = torch.softmax(scores, dim=1)
        if dropout is not None:
            attn_weight = dropout(attn_weight)
        out = (attn_weight * value).sum(dim=1)
        return out, attn_weight.squeeze()

class DMN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, padding_idx, seqbegin_id, dropout_p=0.1):
        '''
        Args:
            vocab_size -- 词汇表大小
            embed_size -- 词嵌入维数
            hidden_size -- GRU的输出维数
            padding_idx -- pad标记的wordid
            seqbegin_id -- 句子起始的wordid
            dropout_p -- dropout比率
        '''
        super(DMN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seqbegin_id = seqbegin_id
        
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.input_gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.question_gru = nn.GRU(embed_size, hidden_size, batch_first=True)    
        self.gate = nn.Sequential(
                        nn.Linear(hidden_size * 4, hidden_size),
                        nn.Tanh(),
                        nn.Linear(hidden_size, 1),
                        nn.Sigmoid()
                    )
        self.attention_grucell = nn.GRUCell(hidden_size, hidden_size)
        self.memory_grucell = nn.GRUCell(hidden_size, hidden_size)
        self.answer_grucell = nn.GRUCell(hidden_size * 2, hidden_size)
        self.answer_fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        
        self.init_weight()

    def init_hidden(self, batch_size):
        '''GRU的初始hidden。单层单向'''
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        hidden = get_variable(hidden)
        return hidden

    def init_weight(self):
        nn.init.xavier_uniform(self.embed.state_dict()['weight'])
        components = [self.input_gru, self.question_gru, self.gate, self.attention_grucell,
                     self.memory_grucell, self.answer_grucell]
        for component in components:
            for name, param in component.state_dict().items():
                if 'weight' in name:
                    nn.init.xavier_normal(param)
        nn.init.xavier_uniform(self.answer_fc.state_dict()['weight'])
        self.answer_fc.bias.data.fill_(0)

    def forward(self, allfacts, allfacts_mask, questions, questions_mask, alen, n_episode=3):
        '''
        Args:
            allfacts -- [b, n_fact, flen]，输入的多个句子
            allfacts_mask -- [b, n_fact, flen]，mask=1表示是pad的，否则不是
            questions -- [b, qlen]，问题
            questions_mask -- [b, qlen]，mask=1：pad
            alen -- Answer len
            seqbegin_id -- 句子开始标记的wordid
            n_episodes -- 
        Returns:
            preds -- [b * alen,  vocab_size]，预测的句子。b*alen合在一起方便后面算交叉熵
        '''
        # 0. 计算常用的信息，batch_size，一条数据nfact条句子，每个fact长度为flen，每个问题长度为qlen
        bsize = allfacts.size(0)
        nfact = allfacts.size(1)
        flen = allfacts.size(2)
        qlen = questions.size(1)
        
        # 1. 输入模块，用RNN编码输入的句子
        # TODO 两层循环，待优化
        encoded_facts = []
        # 对每一条数据，计算facts编码
        for facts, facts_mask in zip(allfacts, allfacts_mask):
            facts_embeds = self.embed(facts)
            facts.embeds = self.dropout(facts_embeds)
            hidden = self.init_hidden(nfact)
            # 1.1 把输入(多条句子)给到GRU
            # b=nf, [nf, flen, h], [1, nf, h]
            outputs, hidden = self.input_gru(facts_embeds, hidden)
            # 1.2 每条句子真正结束时(real_len)对应的输出，作为该句子的hidden。GRU：ouput=hidden
            real_hiddens = []

            for i, o in enumerate(outputs):
                real_len = facts_mask[i].data.tolist().count(0)
                real_hiddens.append(o[real_len - 1])
            # 1.3 把所有单个fact连接起来，unsqueeze(0)是为了后面的所有batch的cat
            hiddens = torch.cat(real_hiddens).view(nfact, -1).unsqueeze(0)
            encoded_facts.append(hiddens)
        # [b, nfact, h]
        encoded_facts = torch.cat(encoded_facts)

        # 2. 问题模块，对问题使用RNN编码
        questions_embeds = self.embed(questions)
        questions_embeds = self.dropout(questions_embeds)
        hidden = self.init_hidden(bsize)
        # [b, qlen, h], [1, b, h]
        outputs, hidden = self.question_gru(questions_embeds, hidden)
        real_questions = []
        for i, o in enumerate(outputs):
            real_len = questions_mask[i].data.tolist().count(0)
            real_questions.append(o[real_len - 1])
        encoded_questions = torch.cat(real_questions).view(bsize, -1)
        
        # 3. Memory模块
        memory = encoded_questions
        for i in range(n_episode):
            # e
            e = self.init_hidden(bsize).squeeze(0)
            # [nfact, b, h]
            encoded_facts_t = encoded_facts.transpose(0, 1)
            # 根据memory, episode，计算每一时刻的e。最终的e和memory来计算新的memory
            for t in range(nfact):
                # [b, h]
                bfact = encoded_facts_t[t]
                # TODO 计算4个特征，论文是9个
                f1 = bfact * encoded_questions
                f2 = bfact * memory
                f3 = torch.abs(bfact - encoded_questions)
                f4 = torch.abs(bfact - memory)
                z = torch.cat([f1, f2, f3, f4], dim=1)
                # [b, 1] 对每个fact的注意力
                gt = self.gate(z)
                e = gt * self.attention_grucell(bfact, e) + (1 - gt) * e
            # 每一轮的e和旧memory计算新的memory
            memory = self.memory_grucell(e, memory)
        
        # 4. Answer模块
        # [b, h]
        answer_hidden = memory
        begin_tokens = get_variable(torch.LongTensor([self.seqbegin_id]*bsize))
        # [b, h]
        last_word = self.embed(begin_tokens)
        preds = []
        for i in range(alen):
            inputs = torch.cat([last_word, encoded_questions], dim=1)
            answer_hidden = self.answer_grucell(inputs, answer_hidden)
            # to vocab_size
            probs = self.answer_fc(answer_hidden)
            # [b, v]
            probs = F.log_softmax(probs.float())
            _, indics = torch.max(probs, 1)
            last_word = self.embed(indics)
            # for cross entropy
            preds.append(probs.view(bsize, 1, -1))
            #preds.append(indics.view(bsize, -1))
        #print (preds[0].data.shape)
        preds = torch.cat(preds, dim=1)
        #print (preds.data.shape)
        return preds.view(bsize * alen, -1)

if __name__ == "__main__":
    from dataset import bAbIDataset
    from torch.utils.data import DataLoader
    train_d = bAbIDataset("./data/tasks_1-20_v1-2/en/", task_id=2, train=True)
    loader = DataLoader(train_d, batch_size=4, shuffle=False)

    device = torch.device("cuda:0")
    story, query, answer = next(iter(loader))
    story, query = story.to(device), query.to(device)
    m = WMN(vocab_size=train_d.num_vocab, embed_size=32, hidden_size=20, seqend_idx=2, pad_idx=0)
    m = m.to(device)
    out = m(story, query)

