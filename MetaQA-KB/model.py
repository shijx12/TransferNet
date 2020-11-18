import torch
import torch.nn as nn
import math

from utils.BiGRU import GRU, BiGRU
from .Knowledge_graph import KnowledgeGraph

class TransferNet(nn.Module):
    def __init__(self, args, dim_word, dim_hidden, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.kg = KnowledgeGraph(args, vocab)
        num_words = len(vocab['word2id'])
        num_entities = len(vocab['entity2id'])
        num_relations = len(vocab['relation2id'])
        self.num_steps = args.num_steps
        self.aux_hop = args.aux_hop

        self.question_encoder = BiGRU(dim_word, dim_hidden, num_layers=1, dropout=0.2)
        
        self.word_embeddings = nn.Embedding(num_words, dim_word)
        self.word_dropout = nn.Dropout(0.2)
        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.Tanh()
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)
        self.rel_classifier = nn.Linear(dim_hidden, num_relations)
        # self.q_classifier = nn.Linear(dim_hidden, num_entities)

        self.hop_selector = nn.Linear(dim_hidden, self.num_steps)


    def follow(self, e, r):
        x = torch.sparse.mm(self.kg.Msubj, e.t()) * torch.sparse.mm(self.kg.Mrel, r.t())
        return torch.sparse.mm(self.kg.Mobj.t(), x).t() # [bsz, Esize]


    def forward(self, questions, e_s, answers=None, hop=None):
        question_lens = questions.size(1) - questions.eq(0).long().sum(dim=1) # 0 means <PAD>
        q_word_emb = self.word_dropout(self.word_embeddings(questions)) # [bsz, max_q, dim_hidden]
        q_word_h, q_embeddings, q_hn = self.question_encoder(q_word_emb, question_lens) # [bsz, max_q, dim_h], [bsz, dim_h], [num_layers, bsz, dim_h]

        device = q_word_h.device
        bsz = q_word_h.size(0)
        dim_h = q_word_h.size(-1)
        last_e = e_s
        word_attns = []
        rel_probs = []
        ent_probs = []
        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_embeddings) # [bsz, dim_h]
            q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [bsz, max_q]
            q_dist = torch.softmax(q_logits, 1).unsqueeze(1) # [bsz, 1, max_q]
            word_attns.append(q_dist.squeeze(1))
            ctx_h = (q_dist @ q_word_h).squeeze(1) # [bsz, dim_h]
            rel_dist = torch.softmax(self.rel_classifier(ctx_h), 1) # [bsz, num_relations]
            rel_probs.append(rel_dist)

            last_e = self.follow(last_e, rel_dist)

            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            # Specifically for MetaQA: reshape cycle entities to 0, because A-r->B-r_inv->A is not allowed
            if t > 0:
                prev_rel = torch.argmax(rel_probs[-2], dim=1)
                curr_rel = torch.argmax(rel_probs[-1], dim=1)
                prev_prev_ent_prob = ent_probs[-2] if len(ent_probs)>=2 else e_s
                # in our vocabulary, indices of inverse relations are adjacent. e.g., director:0, director_inv:1
                m = torch.zeros((bsz,1)).to(device)
                m[(torch.abs(prev_rel-curr_rel)==1) & (torch.remainder(torch.min(prev_rel,curr_rel),2)==0)] = 1
                ent_m = m.float() * prev_prev_ent_prob.gt(0.9).float()
                last_e = (1-ent_m) * last_e

            ent_probs.append(last_e)

        hop_res = torch.stack(ent_probs, dim=1) # [bsz, num_hop, num_ent]
        hop_logit = self.hop_selector(q_embeddings)
        hop_attn = torch.softmax(hop_logit, dim=1) # [bsz, num_hop]
        last_e = torch.sum(hop_res * hop_attn.unsqueeze(2), dim=1) # [bsz, num_ent]

        # Specifically for MetaQA: for 2-hop questions, topic entity is excluded from answer
        m = hop_attn.argmax(dim=1).eq(1).float().unsqueeze(1) * e_s
        last_e = (1-m) * last_e

        # question mask, incorporate language bias
        # q_mask = torch.sigmoid(self.q_classifier(q_embeddings))
        # last_e = last_e * q_mask

        if not self.training:
            return {
                'e_score': last_e,
                'word_attns': word_attns,
                'rel_probs': rel_probs,
                'ent_probs': ent_probs
            }
        else:
            # Distance loss
            weight = answers * 9 + 1
            loss_score = torch.mean(weight * torch.pow(last_e - answers, 2))
            loss = {'loss_score': loss_score}

            if self.aux_hop:
                loss_hop = nn.CrossEntropyLoss()(hop_logit, hop-1)
                loss['loss_hop'] = 0.01 * loss_hop

            return loss
