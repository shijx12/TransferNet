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
    def __init__(self, args, vocab):
        super().__init__()
        self.args = args
        dim_word = args.dim_word
        dim_hidden = args.dim_hidden
        num_words = len(vocab)

        # self.bert_encoder = AutoModel.from_pretrained(args.bert_type, return_dict=True)
        # dim_hidden = self.bert_encoder.config.hidden_size

        self.word_embeddings = nn.Embedding(num_words, dim_word)
        self.word_dropout = nn.Dropout(0.2)

        self.desc_encoder = BiGRU(dim_word, dim_hidden, num_layers=1, dropout=0.2)
        self.question_encoder = BiGRU(dim_word, dim_hidden, num_layers=1, dropout=0.2)
        self.entity_encoder = BiGRU(dim_word, dim_hidden, num_layers=1, dropout=0.2)

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
        

    def forward(self, origin_entity, entity, kb_pair, kb_desc, kb_range, question, answer_idx=None, gold_answer=None):
        # ent_emb = self.bert_encoder(**entity).pooler_output # (num_ent, dim_h)
        # q = self.bert_encoder(**question)
        # q_emb, q_word_h = q.pooler_output, q.last_hidden_state # (1, dim_h), (1, len, dim_h)
        # print(kb_desc['input_ids'].size())
        # desc_emb = self.bert_encoder(**kb_desc).pooler_output # (num_kb, dim_h)

        def seq_encode(seq, encoder):
            seq_len = seq.size(1) - seq.eq(0).long().sum(dim=1) # 0 means <PAD>
            word_emb = self.word_dropout(self.word_embeddings(seq)) # [bsz, max_q, dim_hidden]
            word_h, seq_emb, _ = encoder(word_emb, seq_len) # [bsz, max_q, dim_h], [bsz, dim_h], [num_layers, bsz, dim_h]
            return word_h, seq_emb

        _, ent_emb = seq_encode(entity, self.entity_encoder)
        q_word_h, q_emb = seq_encode(question, self.question_encoder)
        _, desc_emb = seq_encode(kb_desc, self.desc_encoder)

        last_e = None
        word_attns = []
        ent_probs = []

        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_emb) # [1, dim_h]
            q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [1, len]
            q_dist = torch.softmax(q_logits, 1).unsqueeze(1) # [1, 1, len]
            word_attns.append(q_dist.squeeze(1))
            ctx_h = (q_dist @ q_word_h).squeeze(1) # [1, dim_h]
            ctx_h = ctx_h + cq_t

            if t == 0:
                last_e = torch.softmax(torch.sum(ent_emb * ctx_h, 1), 0) # (num_ent)
            else:
                d_logit = self.rel_classifier(ctx_h * desc_emb).squeeze(1) # (num_kb,)
                d_prob = torch.sigmoid(d_logit) # (num_kb,)
                # transfer probability
                last_e = self.follow(last_e, kb_pair, d_prob)

                # collect path
                # act_idx = d_prob.gt(0.9)
                # act_pair = pair[act_idx].tolist()
                # act_desc = [' '.join([self.vocab['id2word'][w] for w in d if w > 0]) for d in desc[act_idx].tolist()]
                # path_infos[i][t] = [(act_pair[_][0], act_desc[_], act_pair[_][1]) for _ in range(len(act_pair))]


            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            ent_probs.append(last_e.detach())

        # question mask, incorporate language bias
        q_mask = torch.sigmoid(self.q_classifier(q_emb * ent_emb)).squeeze(1)
        last_e = last_e * q_mask

        # whether the answer should be yes/no
        binary_prob = torch.sigmoid(self.binary_indicator(q_emb)).squeeze(1) # (1,)
        
        ent_probs[-1] = last_e.detach() # use the newest last_e
        if gold_answer is None:
            if binary_prob.item() > 0.5:
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
            }
        else:
            if gold_answer in {'yes', 'no'}:
                total_attn = last_e.sum()
                target = 1 if gold_answer == 'yes' else 0
                ans_loss = torch.abs(total_attn - target)
                bin_target = 1
            else:
                answer_onehot = idx_to_one_hot(answer_idx, len(last_e)).to(last_e.device)
                weight = answer_onehot * 9 + 1
                ans_loss = torch.mean(weight * torch.pow(last_e - answer_onehot, 2))
                bin_target = 0

            bin_loss = torch.abs(binary_prob - bin_target)

            return {'ans_loss': ans_loss, 'bin_loss': bin_loss}
