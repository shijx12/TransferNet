import os
import torch
import torch.nn as nn
import pickle
import math
import random

from utils.BiGRU import GRU, BiGRU
from utils.misc import idx_to_one_hot
from transformers import AutoModel, AutoTokenizer
from IPython import embed

class TransferNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_act = args.max_act
        dim_hidden = args.dim_hidden

        self.bert_encoder = AutoModel.from_pretrained(args.bert_type, return_dict=True)
        # self.bert_encoder_for_desc = AutoModel.from_pretrained(args.bert_type, return_dict=True)
        dim_hidden = self.bert_encoder.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(args.bert_type)

        self.num_steps = args.num_steps
        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.GELU(),
                nn.Linear(dim_hidden, dim_hidden),
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)
        
        self.type_classifier = nn.Sequential(
                nn.Linear(dim_hidden, 256),
                nn.ReLU(),
                nn.Linear(256, 3)
            )
        self.yes_no_classifier = nn.Sequential(
                nn.Linear(dim_hidden, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )
        self.select_classifier = nn.Sequential(
                nn.Linear(dim_hidden, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            )
        # self.rel_classifier = nn.Sequential(
        #         nn.Linear(dim_hidden, 256),
        #         nn.ReLU(),
        #         nn.Linear(256, 1)
        #     )
        self.rel_classifier = nn.Sequential(
                nn.Linear(dim_hidden, 256),
                nn.GELU(),
                nn.Linear(256, 1)
            )
        self.hop_selector = nn.Sequential(
                nn.Linear(dim_hidden, 256),
                nn.GELU(),
                nn.Linear(256, args.num_steps)
            )


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
        

    def forward(self, origin_entity, entity, kb_pair, kb_desc, kb_range, 
                question, question_type=-1, topic_ent_idxs=None, topic_ent_desc=None,
                answer_idx=None, gold_answer=None):
        kb_lens = kb_range[:,1]-kb_range[:,0]
        # print(kb_lens.min().item(), kb_lens.float().mean().item(), kb_lens.max().item())

        q = self.bert_encoder(**question)
        q_emb, q_word_h = q.pooler_output, q.last_hidden_state # (1, dim_h), (1, len, dim_h)
        device = q_emb.device

        # predict question type
        type_logit = self.type_classifier(q_emb) # (1, 3)
        if self.training:
            assert question_type > -1
            type_loss = 0.01 * nn.CrossEntropyLoss()(type_logit, torch.LongTensor([question_type]).to(device))
        else:
            pass
            # question_type = type_logit.argmax(dim=1).item()
        
        if question_type == 0: # predict yes/no, the question may be both/same/different
            e = self.bert_encoder(**topic_ent_desc)
            ent_emb, emb_word_h = e.pooler_output, e.last_hidden_state # (#ent, dim_h), (#ent, len, dim_h)
            feat = torch.sum(ent_emb*q_emb, dim=0, keepdim=True) #(1, dim_h)
            logit = self.yes_no_classifier(feat) # (1, 2)
            if self.training:
                target = 0 if gold_answer == 'yes' else 1
                ans_loss = nn.CrossEntropyLoss()(logit, torch.LongTensor([target]).to(device))
            else:
                prediction = 'yes' if logit.argmax(dim=1).item()==0 else 'no'
                vis = None

        elif question_type == 1: # select one from topic entities
            e = self.bert_encoder(**topic_ent_desc)
            ent_emb, emb_word_h = e.pooler_output, e.last_hidden_state # (#ent, dim_h), (#ent, len, dim_h)
            logit = self.select_classifier(ent_emb*q_emb).view(1, -1) # (1, #ent)
            if self.training:
                ans_loss = nn.CrossEntropyLoss()(logit, torch.LongTensor([answer_idx]).to(device))
            else:
                i = topic_ent_idxs[logit.argmax(dim=1)]
                prediction = origin_entity[i]
                vis = None

        elif question_type == 2: # multi-hop
            # last_e = None
            last_e = idx_to_one_hot(topic_ent_idxs, len(origin_entity)).to(device)
            ent_probs = []
            word_attns = [None]*self.num_steps
            path_infos = [None]*self.num_steps # [num_steps]

            can_reach = False
            for t in range(self.num_steps):
                cq_t = self.step_encoders[t](q_emb) # [1, dim_h]
                q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [1, len]
                q_dist = torch.softmax(q_logits, 1).unsqueeze(1) # [1, 1, len]
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

                if not self.training:
                    path_infos[t] = [
                        '{:.3f} {} ---> {} : {}'.format(
                            d_prob[i].item(),
                            origin_entity[pair[i][0].item()], origin_entity[pair[i][1].item()],
                            self.tokenizer.decode(desc['input_ids'][i], skip_special_tokens=True)
                        ) for i in range(len(d_prob))]
                    word_attns[t] = [
                        '{}: {:.3f}'.format(self.tokenizer.decode(i), a.item())
                        for i, a in zip(question['input_ids'].squeeze(), q_dist.squeeze())
                    ]


                # reshape >1 scores to 1 in a differentiable way
                m = last_e.gt(1).float()
                z = (m * last_e + (1-m)).detach()
                last_e = last_e / z

                ent_probs.append(last_e)


            hop_res = torch.stack(ent_probs, dim=0) # [num_hop, num_ent]
            hop_logit = self.hop_selector(q_emb) # [1, num_hop]
            hop_attn = torch.softmax(hop_logit, dim=1) # [1, num_hop]
            last_e = torch.mm(hop_attn, hop_res).squeeze(0) # [num_ent]

            if self.training:
                answer_onehot = idx_to_one_hot(answer_idx, len(last_e)).to(device)
                weight = answer_onehot * 9 + 1
                ans_loss = torch.sum(weight * torch.pow(last_e - answer_onehot, 2)) / torch.sum(weight)
            else:
                prediction = origin_entity[last_e.argmax().item()]
                vis = {'ent_probs': ent_probs, 'hop_attn': hop_attn, 'word_attns': word_attns, 'path_infos': path_infos}
        
        # print(question_type, ans_loss.item())
        if self.training:
            return { 'ans_loss': ans_loss, 'type_loss': type_loss }
        else:
            return { 'pred': prediction, 'vis': vis }
