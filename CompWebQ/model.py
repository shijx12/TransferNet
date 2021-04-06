import torch
import torch.nn as nn
import math
from transformers import AutoModel
from utils.BiGRU import GRU, BiGRU

class TransferNet(nn.Module):
    def __init__(self, args, ent2id, rel2id):
        super().__init__()
        num_relations = len(rel2id)
        self.num_ents = len(ent2id)
        self.num_steps = args.num_steps
        self.num_ways = args.num_ways

        self.bert_encoder = AutoModel.from_pretrained(args.bert_name, return_dict=True)
        dim_hidden = self.bert_encoder.config.hidden_size

        self.step_encoders = {}
        self.hop_selectors = {}
        self.rel_classifiers = {}
        for i in range(self.num_ways):
            for j in range(self.num_steps):
                m = nn.Sequential(
                    nn.Linear(dim_hidden*2, dim_hidden),
                    nn.Tanh()
                )
                name = 'way_{}_step_{}'.format(i, j)
                self.step_encoders[name] = m
                self.add_module(name, m)

            m = nn.Linear(dim_hidden, self.num_steps)
            self.hop_selectors['way_{}'.format(i)] = m
            self.add_module('hop-way_{}'.format(i), m)

            m = nn.Linear(dim_hidden, num_relations)
            self.rel_classifiers['way_{}'.format(i)] = m
            self.add_module('rel-way_{}'.format(i), m)
        


    def forward(self, heads, questions, answers=None, triples=None, entity_range=None):
        q = self.bert_encoder(**questions)
        q_embeddings, q_word_h = q.pooler_output, q.last_hidden_state # (bsz, dim_h), (bsz, len, dim_h)
        bsz = len(heads)
        device = heads.device

        e_score = []
        last_h = torch.zeros_like(q_embeddings)
        for w in range(self.num_ways):
            last_e = heads
            word_attns = []
            rel_probs = []
            ent_probs = []
            for t in range(self.num_steps):
                cq_t = self.step_encoders['way_{}_step_{}'.format(w, t)](
                    torch.cat((q_embeddings, last_h), dim=1) # consider history
                ) # [bsz, dim_h]
                q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [bsz, max_q]
                q_dist = torch.softmax(q_logits, 1) # [bsz, max_q]
                q_dist = q_dist * questions['attention_mask'].float()
                q_dist = q_dist / (torch.sum(q_dist, dim=1, keepdim=True) + 1e-6) # [bsz, max_q]
                word_attns.append(q_dist)
                ctx_h = (q_dist.unsqueeze(1) @ q_word_h).squeeze(1) # [bsz, dim_h]
                ctx_h = ctx_h + cq_t
                last_h = ctx_h

                rel_logit = self.rel_classifiers['way_{}'.format(w)](ctx_h) # [bsz, num_relations]
                # rel_dist = torch.softmax(rel_logit, 1) # bad
                rel_dist = torch.sigmoid(rel_logit)
                rel_probs.append(rel_dist)

                new_e = []
                for b in range(bsz):
                    sub, rel, obj = triples[b][:,0], triples[b][:,1], triples[b][:,2]
                    sub_p = last_e[b:b+1, sub] # [1, #tri]
                    rel_p = rel_dist[b:b+1, rel] # [1, #tri]
                    obj_p = sub_p * rel_p
                    new_e.append(
                        torch.index_add(torch.zeros(1, self.num_ents).to(device), 1, obj, obj_p))
                last_e = torch.cat(new_e, dim=0)

                # reshape >1 scores to 1 in a differentiable way
                m = last_e.gt(1).float()
                z = (m * last_e + (1-m)).detach()
                last_e = last_e / z

                ent_probs.append(last_e)

            hop_res = torch.stack(ent_probs, dim=1) # [bsz, num_hop, num_ent]
            hop_logit = self.hop_selectors['way_{}'.format(w)](q_embeddings)
            hop_attn = torch.softmax(hop_logit, dim=1).unsqueeze(2) # [bsz, num_hop, 1]
            last_e = torch.sum(hop_res * hop_attn, dim=1) # [bsz, num_ent]

            e_score.append(last_e)

        e_score = torch.prod(torch.stack(e_score), dim=0)

        if not self.training:
            return {
                'e_score': e_score,
                'word_attns': word_attns,
                'rel_probs': rel_probs,
                'ent_probs': ent_probs,
                # 'hop_attn': hop_attn.squeeze(2)
            }
        else:
            weight = answers * 9 + 1
            loss = torch.sum(entity_range * weight * torch.pow(last_e - answers, 2)) / torch.sum(entity_range * weight)

            return {'loss': loss}
