import os
import torch
import torch.nn as nn
import pickle
import math
import random

from utils.BiGRU import GRU, BiGRU
from utils.misc import idx_to_one_hot
from transformers import AutoModel
from IPython import embed

class TransferNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_act = args.max_act
        dim_hidden = args.dim_hidden

        self.bert_encoder = AutoModel.from_pretrained(args.bert_type, return_dict=True)
        dim_hidden = self.bert_encoder.config.hidden_size

        self.num_steps = args.num_steps
        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.Tanh(),
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)
        
        self.rel_classifier = nn.Linear(dim_hidden, 1)

        self.q_classifier = nn.Linear(dim_hidden, 1)
        self.binary_indicator = nn.Linear(dim_hidden, 1)
        self.hop_selector = nn.Linear(dim_hidden, self.num_steps)


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
        

    def forward(self, origin_entity, entity, kb_pair, kb_desc, kb_range, question, topic_ent_idxs=None, answer_idx=None, gold_answer=None):
        kb_lens = kb_range[:,1]-kb_range[:,0]
        # print(kb_lens.min().item(), kb_lens.float().mean().item(), kb_lens.max().item())

        # ent_emb = self.bert_encoder(**entity).pooler_output # (num_ent, dim_h)
        q = self.bert_encoder(**question)
        q_emb, q_word_h = q.pooler_output, q.last_hidden_state # (1, dim_h), (1, len, dim_h)
        # print(kb_desc['input_ids'].size())
        # desc_emb = self.bert_encoder(**kb_desc).pooler_output # (num_kb, dim_h)

        # def seq_encode(seq, encoder):
        #     seq_len = seq.size(1) - seq.eq(0).long().sum(dim=1) # 0 means <PAD>
        #     word_emb = self.word_dropout(self.word_embeddings(seq)) # [bsz, max_q, dim_hidden]
        #     word_h, seq_emb, _ = encoder(word_emb, seq_len) # [bsz, max_q, dim_h], [bsz, dim_h], [num_layers, bsz, dim_h]
        #     return word_h, seq_emb

        # _, ent_emb = seq_encode(entity, self.entity_encoder)
        # q_word_h, q_emb = seq_encode(question, self.question_encoder)
        # _, desc_emb = seq_encode(kb_desc, self.desc_encoder)

        device = q_emb.device
        # last_e = None
        last_e = idx_to_one_hot(topic_ent_idxs, len(origin_entity)).to(device)
        word_attns = []
        ent_probs = []
        path_infos = [None]*self.num_steps # [num_steps]

        can_reach = False
        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_emb) # [1, dim_h]
            q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [1, len]
            q_dist = torch.softmax(q_logits, 1).unsqueeze(1) # [1, 1, len]
            word_attns.append(q_dist.squeeze(1))
            ctx_h = (q_dist @ q_word_h).squeeze(1) # [1, dim_h]
            ctx_h = ctx_h + cq_t

            # if t == 0:
            #     last_e = torch.softmax(torch.sum(ent_emb * ctx_h, 1), 0) # (num_ent)

            #     if not self.training:
            #         path_infos[t] = [origin_entity[last_e.argmax(0).item()]]
            # else:
            # d_logit = self.rel_classifier(ctx_h * desc_emb).squeeze(1) # (num_kb,)
            # d_prob = torch.sigmoid(d_logit) # (num_kb,)
            # # transfer probability
            # last_e = self.follow(last_e, kb_pair, d_prob)

            sort_score, sort_idx = torch.sort(last_e, dim=0, descending=True)
            e_idx = sort_idx[sort_score.gt(0.5)].tolist()
            if len(e_idx) == 0:
                e_idx = sort_idx[:self.max_act//2].tolist()
            rg = []
            for j in e_idx:
                rg.append(torch.arange(kb_range[j,0], kb_range[j,1]).long().to(device))
            rg = torch.cat(rg, dim=0) # [rsz,]
            if len(rg) > self.max_act:
                rg = rg[:self.max_act]
            pair = kb_pair[rg]
            if answer_idx in set(pair[:,1].tolist()):
                can_reach = True
            desc = {k:v[rg] for k,v in kb_desc.items()}
            desc_emb = self.bert_encoder(**desc).pooler_output # (rsz, dim_h)
            d_logit = self.rel_classifier(ctx_h * desc_emb).squeeze(1)
            d_prob = torch.sigmoid(d_logit)
            last_e = self.follow(last_e, pair, d_prob)

            # if not self.training:
            #     # collect path
            #     act_idx = d_prob.gt(0.8)
            #     act_pair = kb_pair[act_idx].tolist()
            #     act_desc = [' '.join([self.vocab['id2word'][w] for w in d if w > 0]) for d in kb_desc[act_idx].tolist()]
            #     path_infos[t] = [
            #         '{} ---> {} ---> {}: {:.3f}'.format(
            #             origin_entity[act_pair[_][0]], act_desc[_], origin_entity[act_pair[_][1]], d_prob[act_idx][_].item()) 
            #         for _ in range(len(act_pair))]


            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            ent_probs.append(last_e)


        hop_res = torch.stack(ent_probs, dim=0) # [num_hop, num_ent]
        hop_logit = self.hop_selector(q_emb) # [1, num_hop]
        hop_attn = torch.softmax(hop_logit, dim=1) # [1, num_hop]
        last_e = torch.mm(hop_attn, hop_res).squeeze(0) # [num_ent]

        # question mask, incorporate language bias
        # q_mask = torch.sigmoid(self.q_classifier(q_emb * ent_emb)).squeeze(1)
        # last_e = last_e * q_mask

        # whether the answer should be yes/no
        bin_feat = q_emb.detach()
        binary_prob = torch.sigmoid(self.binary_indicator(bin_feat)).squeeze(1) # (1,)
        
        ent_probs[-1] = last_e # use the newest last_e
        if not self.training:
            # if binary_prob.item() > 0.5:
            if False:
                total_attn = last_e.sum().item()
                if total_attn >= 1:
                    pred_answer = 'yes'
                else:
                    pred_answer = 'no'
            else:
                pred_answer = origin_entity[last_e.argmax().item()]
            return {
                'pred': pred_answer,
                'word_attns': word_attns,
                'ent_probs': ent_probs,
                'path_infos': path_infos
            }
        else:
            if gold_answer in {'yes', 'no'}:
                total_attn = last_e.sum()
                if total_attn.item() > 1:
                    total_attn = total_attn / total_attn.item()
                target = 1 if gold_answer == 'yes' else 0
                ans_loss = torch.abs(total_attn - target)
                bin_target = 1
            else:
                answer_onehot = idx_to_one_hot(answer_idx, len(last_e)).to(device)
                weight = answer_onehot * 9 + 1
                ans_loss = torch.sum(weight * torch.pow(last_e - answer_onehot, 2)) / torch.sum(weight)
                # print(last_e, answer_idx, last_e[answer_idx], ans_loss)
                # if not can_reach:
                #     print(gold_answer, '|', origin_entity[answer_idx])
                bin_target = 0

            bin_loss = nn.BCELoss()(binary_prob, torch.zeros_like(binary_prob).fill_(bin_target))

            return {'ans_loss': ans_loss, 'bin_loss': bin_loss}
            # return {'ans_loss': ans_loss}
