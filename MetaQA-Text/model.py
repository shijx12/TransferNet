import os
import torch
import torch.nn as nn
import pickle
import math
import random

from utils.BiGRU import GRU, BiGRU
from IPython import embed

class TransferNet(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.max_active = args.max_active
        self.ent_act_thres = args.ent_act_thres
        dim_word = args.dim_word
        dim_hidden = args.dim_hidden
        
        with open(os.path.join(args.input_dir, 'wiki.pt'), 'rb') as f:
            self.kb_pair = torch.LongTensor(pickle.load(f))
            self.kb_range = torch.LongTensor(pickle.load(f))
            self.kb_desc = torch.LongTensor(pickle.load(f))

        num_words = len(vocab['word2id'])
        num_entities = len(vocab['entity2id'])
        self.num_steps = args.num_steps

        self.desc_encoder = BiGRU(dim_word, dim_hidden, num_layers=1, dropout=0.2)
        self.question_encoder = BiGRU(dim_word, dim_hidden, num_layers=1, dropout=0.2)
        
        self.word_embeddings = nn.Embedding(num_words, dim_word)
        self.word_dropout = nn.Dropout(0.2)
        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.Tanh(),
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)
        self.rel_classifier = nn.Linear(dim_hidden, 1)

        self.q_classifier = nn.Linear(dim_hidden, num_entities)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def follow(self, e, pair, p):
        """
        Args:
            e [num_ent]: entity scores
            pair [rsz, 2]: pairs that are taken into consider
            p [rsz]: transfer probabilities of each pair
        """
        sub, obj = pair[:, 0], pair[:, 1]
        obj_p = e[sub] * p
        out = torch.index_add(torch.zeros_like(e), 0, obj, obj_p)
        return out
        

    def forward(self, questions, e_s, answers = None):
        question_lens = questions.size(1) - questions.eq(0).long().sum(dim=1) # 0 means <PAD>
        q_word_emb = self.word_dropout(self.word_embeddings(questions)) # [bsz, max_q, dim_hidden]
        q_word_h, q_embeddings, q_hn = self.question_encoder(q_word_emb, question_lens) # [bsz, max_q, dim_h], [bsz, dim_h], [num_layers, bsz, dim_h]


        device = q_word_h.device
        bsz, dim_h = q_embeddings.size()
        last_e = e_s
        word_attns = []
        ent_probs = [e_s]
        
        path_infos = [] # [bsz, num_steps]
        for i in range(bsz):
            path_infos.append([])
            for j in range(self.num_steps):
                path_infos[i].append(None)

        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_embeddings) # [bsz, dim_h]
            q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [bsz, max_q]
            q_dist = torch.softmax(q_logits, 1).unsqueeze(1) # [bsz, 1, max_q]
            word_attns.append(q_dist.squeeze(1))
            ctx_h = (q_dist @ q_word_h).squeeze(1) # [bsz, dim_h]
            ctx_h = ctx_h + cq_t

            e_stack = []
            cnt_trunc = 0
            for i in range(bsz):
                # e_idx = torch.topk(last_e[i], k=1, dim=0)[1].tolist() + \
                #         last_e[i].gt(self.ent_act_thres).nonzero().squeeze(1).tolist()
                # DOING
                if self.training and t > 0 and random.random() < 0.05:
                    e_idx = last_e[i].gt(0).nonzero().squeeze(1).tolist()
                    random.shuffle(e_idx)
                else:
                    sort_score, sort_idx = torch.sort(last_e[i], dim=0, descending=True)
                    e_idx = sort_idx[sort_score.gt(self.ent_act_thres)].tolist()
                    e_idx = set(e_idx) - set([0])
                    if len(e_idx) == 0:
                        # print('no active entity at step {}'.format(t))
                        pad = sort_idx[0].item()
                        if pad == 0:
                            pad = sort_idx[1].item()
                        e_idx = set([pad])

                rg = []
                for j in e_idx:
                    rg.append(torch.arange(self.kb_range[j,0], self.kb_range[j,1]).long().to(device))
                rg = torch.cat(rg, dim=0) # [rsz,]
                # print(len(e_idx), len(rg))
                if len(rg) > self.max_active: # limit the number of next-hop
                    rg = rg[:self.max_active]
                    # DOING
                    # rg = rg[torch.randperm(len(rg))[:self.max_active]]
                    cnt_trunc += 1
                    # print('trunc: {}'.format(cnt_trunc))

                # print('step {}, desc number {}'.format(t, len(rg)))
                pair = self.kb_pair[rg] # [rsz, 2]
                desc = self.kb_desc[rg] # [rsz, max_desc]
                desc_lens = desc.size(1) - desc.eq(0).long().sum(dim=1)
                desc_word_emb = self.word_dropout(self.word_embeddings(desc))
                desc_word_h, desc_embeddings, _ = self.desc_encoder(desc_word_emb, desc_lens) # [rsz, dim_h]
                d_logit = self.rel_classifier(ctx_h[i:i+1] * desc_embeddings).squeeze(1) # [rsz,]
                d_prob = torch.sigmoid(d_logit) # [rsz,]
                # transfer probability
                e_stack.append(self.follow(last_e[i], pair, d_prob))

                # collect path
                act_idx = d_prob.gt(0.9)
                act_pair = pair[act_idx].tolist()
                act_desc = [' '.join([self.vocab['id2word'][w] for w in d if w > 0]) for d in desc[act_idx].tolist()]
                path_infos[i][t] = [(act_pair[_][0], act_desc[_], act_pair[_][1]) for _ in range(len(act_pair))]

            new_e = torch.stack(e_stack, dim=0)

            # reshape >1 scores to 1 in a differentiable way
            m = new_e.gt(1).float()
            z = (m * new_e + (1-m)).detach()
            new_e = new_e / z

            last_e = new_e

            # Specifically for MetaQA: reshape cycle entities to 0, because A-r->B-r_inv->A is not allowed
            if t > 0:
                ent_m = torch.zeros_like(last_e)
                for i in range(bsz):
                    prev_inv = set()
                    for (s, r, o) in path_infos[i][t-1]:
                        prev_inv.add((o, r.replace('__subject__', 'obj').replace('__object__', 'sub'), s))
                    for (s, r, o) in path_infos[i][t]:
                        element = (s, r.replace('__subject__', 'sub').replace('__object__', 'obj'), o)
                        if r != '__self_rel__' and element in prev_inv:
                            ent_m[i, o] = 1
                            # print('block cycle: {}'.format(' ---> '.join(list(map(str, element)))))
                last_e = (1-ent_m) * last_e

            ent_probs.append(last_e.detach())

        # Specifically for MetaQA: for 2-hop questions, topic entity is excluded from answer
        hop2m = torch.zeros((bsz,)).to(device)
        for i in range(bsz):
            self_cnt = 0
            for t in range(self.num_steps):
                cnt = len([r for (s, r, o) in path_infos[i][t] if r == '__self_rel__'])
                if len(path_infos[i][t]) > 0 and cnt/len(path_infos[i][t]) > 0.5:
                    self_cnt += 1
            if self_cnt == 1:
                hop2m[i] = 1
                # print('block 2-hop topic')
        ent_m = hop2m.unsqueeze(1) * e_s
        last_e = (1-ent_m) * last_e

        # question mask, incorporate language bias
        q_mask = torch.sigmoid(self.q_classifier(q_embeddings))
        last_e = last_e * q_mask


        ent_probs[-1] = last_e.detach() # use the newest last_e
        if answers is None:
            return {
                'e_score': last_e,
                'word_attns': word_attns,
                'ent_probs': ent_probs,
                'path_infos': path_infos
            }

        # Distance loss
        weight = answers * 9 + 1
        loss_score = torch.mean(weight * torch.pow(last_e - answers, 2))

        return {'loss_score': loss_score}
