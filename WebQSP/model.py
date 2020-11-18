import torch
import torch.nn as nn
import math
from transformers import AutoModel
from utils.BiGRU import GRU, BiGRU

class TransferNet(nn.Module):
    def __init__(self, args, ent2id, rel2id, triples):
        super().__init__()
        self.args = args
        self.num_steps = 2
        num_relations = len(rel2id)
        # self.triples = triples

        Tsize = len(triples)
        Esize = len(ent2id)
        idx = torch.LongTensor([i for i in range(Tsize)])
        self.Msubj = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:,0])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
        self.Mobj = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:,2])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
        self.Mrel = torch.sparse.FloatTensor(
            torch.stack((idx, triples[:,1])), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, num_relations]))
        print('triple size: {}'.format(Tsize))

        self.bert_encoder = AutoModel.from_pretrained(args.bert_name, return_dict=True)
        dim_hidden = self.bert_encoder.config.hidden_size

        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.Tanh()
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)

        self.rel_classifier = nn.Linear(dim_hidden, num_relations)

        self.hop_selector = nn.Linear(dim_hidden, self.num_steps)


    def follow(self, e, r):
        x = torch.sparse.mm(self.Msubj, e.t()) * torch.sparse.mm(self.Mrel, r.t())
        return torch.sparse.mm(self.Mobj.t(), x).t() # [bsz, Esize]

    def forward(self, heads, questions, answers=None, entity_range=None):
        q = self.bert_encoder(**questions)
        q_embeddings, q_word_h = q.pooler_output, q.last_hidden_state # (bsz, dim_h), (bsz, len, dim_h)

        device = heads.device
        last_e = heads
        word_attns = []
        rel_probs = []
        ent_probs = []
        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_embeddings) # [bsz, dim_h]
            q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [bsz, max_q]
            q_dist = torch.softmax(q_logits, 1) # [bsz, max_q]
            q_dist = q_dist * questions['attention_mask'].float()
            q_dist = q_dist / (torch.sum(q_dist, dim=1, keepdim=True) + 1e-6) # [bsz, max_q]
            word_attns.append(q_dist)
            ctx_h = (q_dist.unsqueeze(1) @ q_word_h).squeeze(1) # [bsz, dim_h]

            rel_logit = self.rel_classifier(ctx_h) # [bsz, num_relations]
            # rel_dist = torch.softmax(rel_logit, 1) # bad
            rel_dist = torch.sigmoid(rel_logit)
            rel_probs.append(rel_dist)

            # sub, rel, obj = self.triples[:,0], self.triples[:,1], self.triples[:,2]
            # sub_p = last_e[:, sub] # [bsz, #tri]
            # rel_p = rel_dist[:, rel] # [bsz, #tri]
            # obj_p = sub_p * rel_p
            # last_e = torch.index_add(torch.zeros_like(last_e), 1, obj, obj_p)

            last_e = self.follow(last_e, rel_dist) # faster than index_add

            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            ent_probs.append(last_e)

        hop_res = torch.stack(ent_probs, dim=1) # [bsz, num_hop, num_ent]
        hop_attn = torch.softmax(self.hop_selector(q_embeddings), dim=1).unsqueeze(2) # [bsz, num_hop, 1]
        last_e = torch.sum(hop_res * hop_attn, dim=1) # [bsz, num_ent]

        if not self.training:
            return {
                'e_score': last_e,
                'word_attns': word_attns,
                'rel_probs': rel_probs,
                'ent_probs': ent_probs,
                'hop_attn': hop_attn.squeeze(2)
            }
        else:
            weight = answers * 99 + 1
            loss = torch.sum(entity_range * weight * torch.pow(last_e - answers, 2)) / torch.sum(entity_range * weight)

            return {'loss': loss}
