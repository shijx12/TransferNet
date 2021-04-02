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
        self.tokenizer = args.tokenizer

        self.bert_encoder = AutoModel.from_pretrained(args.bert_type, return_dict=True)
        self.bert_encoder.resize_token_embeddings(len(self.tokenizer)) # add special vocab
        dim_hidden = self.bert_encoder.config.hidden_size

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

        self.merge_two_side = nn.Sequential(
                nn.Linear(dim_hidden*2, dim_hidden),
                nn.GELU(),
                nn.Linear(dim_hidden, dim_hidden),
            )
        self.pair_encoder = nn.Sequential(
                nn.Linear(dim_hidden*4, dim_hidden),
                nn.GELU(),
                nn.Linear(dim_hidden, dim_hidden),
            )
        
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
        self.sim_classifier = nn.Sequential(
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
    
    def extract_entity_feature(self, para_token_emb, ent_pos):
        """
        Args:
            para_token_emb : (#para, max_len, dim_h)
            ent_pos : list of Tensor
        Return:
            ent_feat : (#ent, dim_h)
        """
        ent_feat = []
        for i in range(len(ent_pos)):
            left = ent_pos[i][:, :2]
            right = ent_pos[i][:, ::2]
            left_feat = para_token_emb[left[:,0], left[:,1]]
            right_feat = para_token_emb[right[:,0], right[:,1]]

            ent_f = torch.cat((left_feat, right_feat), dim=1) # (#occur, 2*dim_h)
            ent_f = torch.mean(ent_f, dim=0) # (2*dim_h)
            ent_feat.append(ent_f)
        ent_feat = torch.stack(ent_feat, dim=0) # (#ent, 2*dim_h)
        ent_feat = self.merge_two_side(ent_feat) # (#ent, dim_h)
        return ent_feat

    def extract_pair_feature(self, para_token_emb, pair_pos):
        """
        Return:
            pair_feat : (#pair, dim_h)
        """
        sub_l = para_token_emb[pair_pos[:,0], pair_pos[:,1]]
        sub_r = para_token_emb[pair_pos[:,0], pair_pos[:,2]]
        obj_l = para_token_emb[pair_pos[:,0], pair_pos[:,3]]
        obj_r = para_token_emb[pair_pos[:,0], pair_pos[:,4]]

        sub_feat = self.merge_two_side(torch.cat((sub_l, sub_r), dim=1)) # (#pair, dim_h)
        obj_feat = self.merge_two_side(torch.cat((obj_l, obj_r), dim=1))
        pair_inp_feat = torch.cat((
            sub_feat, obj_feat, obj_feat-sub_feat, sub_feat*obj_feat), dim=1)
        pair_feat = self.pair_encoder(pair_inp_feat)
        return pair_feat



    def forward(self, entity, ent_pos, pair_so, pair_pos, paragraphs,
                question, question_type=-1, topic_ent_idxs=None, topic_ent_desc=None,
                answer_idx=None, gold_answer=None):
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
                prediction = entity[i]
                vis = None

        elif question_type == 2: # multi-hop
            para_token_emb = self.bert_encoder(**paragraphs).last_hidden_state # (#para, len, dim_h)
            ent_emb = self.extract_entity_feature(para_token_emb, ent_pos)
            pair_emb = self.extract_pair_feature(para_token_emb, pair_pos)

            last_e = None
            ent_probs = []
            word_attns = [None]*self.num_steps
            path_infos = [None]*self.num_steps # [num_steps]

            for t in range(self.num_steps):
                cq_t = self.step_encoders[t](q_emb) # [1, dim_h]
                q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [1, len]
                q_dist = torch.softmax(q_logits, 1).unsqueeze(1) # [1, 1, len]
                ctx_h = (q_dist @ q_word_h).squeeze(1) # [1, dim_h]
                ctx_h = ctx_h + cq_t

                if t == 0:
                    last_e = torch.softmax(self.sim_classifier(ent_emb * ctx_h).squeeze(1), dim=0) # (#ent)
                else:
                    pair_prob = torch.sigmoid(self.sim_classifier(pair_emb * ctx_h).squeeze(1)) # (#pair,)
                    last_e = self.follow(last_e, pair_so, pair_prob)

                if not self.training:
                    if t == 0:
                        path_infos[t] = [
                            '{:.3f} {}'.format(last_e[i].item(), entity[i])
                            for i in range(len(last_e))]
                    else:
                        path_infos[t] = [
                            '{:.3f} {} ---> {} : {}'.format(
                                pair_prob[i].item(),
                                entity[pair_so[i][0].item()], entity[pair_so[i][1].item()],
                                self.tokenizer.decode(paragraphs['input_ids'][pair_pos[i][0].item()], skip_special_tokens=True)
                            ) for i in range(len(pair_prob))]
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
            hop_attn = torch.softmax(self.hop_selector(q_emb), dim=1).squeeze() # [num_hop]
            # hop_attn = hop_attn * torch.Tensor([0,1,1]).to(device) # mask the 0-th hop
            # hop_attn = hop_attn / hop_attn.sum()
            last_e = torch.mm(hop_attn.view(1,-1), hop_res).squeeze(0) # [num_ent]

            if self.training:
                answer_onehot = idx_to_one_hot(answer_idx, len(last_e)).to(device)
                weight = answer_onehot * 9 + 1
                ans_loss = torch.sum(weight * torch.pow(last_e - answer_onehot, 2)) / torch.sum(weight)
            else:
                prediction = entity[last_e.argmax().item()]
                vis = {'ent_probs': ent_probs, 'hop_attn': hop_attn, 'word_attns': word_attns, 'path_infos': path_infos}
        
        # print(question_type, ans_loss.item())
        if self.training:
            return { 'ans_loss': ans_loss, 'type_loss': type_loss }
        else:
            return { 'pred': prediction, 'vis': vis }
