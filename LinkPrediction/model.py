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
        # num_words = len(vocab['word2id'])
        num_entities = len(vocab['entity2id'])
        num_relations = len(vocab['relation2id'])
        self.num_steps = args.num_steps
        self.path_encoder = GRU(dim_hidden, dim_hidden, num_layers=3, dropout = 0.3)
        self.dim_hidden = dim_hidden
        # self.question_encoder = BiGRU(dim_word, dim_hidden, num_layers=1, dropout=0.2)
        
        # self.word_embeddings = nn.Embedding(num_words, dim_word)
        # self.word_dropout = nn.Dropout(0.3)
        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.Tanh()
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)
        # self.rel_classifier = nn.Linear(dim_hidden, num_relations)
        self.cq_linear = nn.Linear(2 * dim_hidden, dim_hidden)
        # self.ca_linear = nn.Linear(dim_hidden, 1)

        self.relation_embeddings = nn.Embedding(num_relations, dim_hidden)
        self.entity_embeddings = nn.Parameter(torch.FloatTensor(num_entities, dim_hidden))
        self.entity_bias = nn.Parameter(torch.FloatTensor(num_entities))
        nn.init.normal_(self.entity_embeddings, mean=0, std=1/math.sqrt(dim_hidden))
        self.entity_bias.data.zero_()
        nn.init.normal_(self.relation_embeddings.weight, mean=0, std=1/math.sqrt(dim_hidden))

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def follow(self, e, r):
        x = torch.sparse.mm(self.kg.Msubj, e.t()) * torch.sparse.mm(self.kg.Mrel, r.t())
        return torch.sparse.mm(self.kg.Mobj.t(), x).t() # [bsz, Esize]
    

    def hist_transfer(self, questions, e_s, answers):
        '''
        Args:
            questions [bsz, 1]: r id for each query
            e_s [bsz, num_entities]: one hot vector of h for each query
            answers [bsz, num_entities]: one hot vector of t in training, to mask the ground truth entity in kg 
        '''
        device = questions.device
        bsz = questions.size(0)
        q_emb = self.relation_embeddings(questions.squeeze()) # [bsz, dim_h]
        last_e = e_s
        rel_probs = []
        ent_probs = [e_s]
        last_h = torch.zeros((3, bsz, self.dim_hidden)).float().to(device)
        last_r_emb = torch.stack([self.relation_embeddings.weight.mean(0, True)] * bsz, 0) # [bsz, 1, dim_h]
        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_emb) # [bsz, dim_h]
            # print(last_r_emb.size())
            # print(last_h.size())
            hist_emb, last_h = self.path_encoder.forward_one_step(last_r_emb, last_h) # [bsz, 1, dim_h] [num_layers, bsz, dim_h]
            hist_emb = hist_emb.squeeze(1)
            rel_logits = self.cq_linear(torch.cat([cq_t, hist_emb], dim = -1)) @ self.relation_embeddings.weight.t() # [bsz, num_relations]
            rel_dist = torch.softmax(rel_logits, 1) # [bsz, num_relations]
            last_r_emb = rel_dist @ self.relation_embeddings.weight # [bsz, dim_h]
            last_r_emb = last_r_emb.unsqueeze(1) # [bsz, 1, dim_h]
            # hist_emb = hist_emb + rel_dist @ self.relation_embeddings.weight # [bsz, dim_h]
            last_e = self.follow(last_e, rel_dist)
            if t == 0 and answers != None:
                gt_mask = torch.gather(rel_dist, 1, questions) # [bsz, 1]
                gt_mask = answers * gt_mask # [bsz, num_entities]
                last_e = last_e - gt_mask
            rel_probs.append(rel_dist)
            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            # Specifically for MetaQA: reshape cycle entities to 0, because A-r->B-r_inv->A is not allowed
            if t > 0:
                prev_rel = torch.argmax(rel_probs[-2], dim=1)
                curr_rel = torch.argmax(rel_probs[-1], dim=1)
                prev_prev_ent_prob = ent_probs[-2]
                # in our vocabulary, indices of inverse relations are adjacent. e.g., director:0, director_inv:1
                m = torch.zeros((bsz,1)).to(device)
                m[(torch.abs(prev_rel-curr_rel)==1) & (torch.remainder(torch.min(prev_rel,curr_rel),2)==0)] = 1
                ent_m = m.float() * prev_prev_ent_prob.gt(0.9).float()
                last_e = (1-ent_m) * last_e

            # Specifically for MetaQA: for 2-hop questions, topic entity is excluded from answer
            if t == self.num_steps-1:
                stack_rel_probs = torch.stack(rel_probs, dim=1) # [bsz, num_step, num_rel]
                stack_rel = torch.argmax(stack_rel_probs, dim=2) # [bsz, num_step]
                num_self = stack_rel.eq(self.vocab['relation2id']['<SELF_REL>']).long().sum(dim=1) # [bsz,]
                m = num_self.eq(1).float().unsqueeze(1) * e_s
                last_e = (1-m) * last_e

            ent_probs.append(last_e.detach())
        return {
            'last_e': last_e,
            'rel_probs': rel_probs,
            'ent_probs': ent_probs
        }

    def transfer(self, questions, e_s, answers):
        '''
        Args:
            questions [bsz, 1]: r id for each query
            e_s [bsz, num_entities]: one hot vector of h for each query
            answers [bsz, num_entities]: one hot vector of t in training, to mask the ground truth entity in kg 
        '''
        device = questions.device
        bsz = questions.size(0)
        q_emb = self.relation_embeddings(questions.squeeze()) # [bsz, dim_h]
        last_e = e_s
        rel_probs = []
        ent_probs = [e_s]

        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_emb) # [bsz, dim_h]
            rel_logits = cq_t @ self.relation_embeddings.weight.t() # [bsz, num_relations]
            rel_dist = torch.softmax(rel_logits, 1) # [bsz, num_relations]
            last_e = self.follow(last_e, rel_dist)
            if t == 0 and answers != None:
                gt_mask = torch.gather(rel_dist, 1, questions) # [bsz, 1]
                gt_mask = answers * gt_mask # [bsz, num_entities]
                last_e = last_e - gt_mask
            rel_probs.append(rel_dist)
            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            # Specifically for MetaQA: reshape cycle entities to 0, because A-r->B-r_inv->A is not allowed
            if t > 0:
                prev_rel = torch.argmax(rel_probs[-2], dim=1)
                curr_rel = torch.argmax(rel_probs[-1], dim=1)
                prev_prev_ent_prob = ent_probs[-2]
                # in our vocabulary, indices of inverse relations are adjacent. e.g., director:0, director_inv:1
                m = torch.zeros((bsz,1)).to(device)
                m[(torch.abs(prev_rel-curr_rel)==1) & (torch.remainder(torch.min(prev_rel,curr_rel),2)==0)] = 1
                ent_m = m.float() * prev_prev_ent_prob.gt(0.9).float()
                last_e = (1-ent_m) * last_e

            # Specifically for MetaQA: for 2-hop questions, topic entity is excluded from answer
            if t == self.num_steps-1:
                stack_rel_probs = torch.stack(rel_probs, dim=1) # [bsz, num_step, num_rel]
                stack_rel = torch.argmax(stack_rel_probs, dim=2) # [bsz, num_step]
                num_self = stack_rel.eq(self.vocab['relation2id']['<SELF_REL>']).long().sum(dim=1) # [bsz,]
                m = num_self.eq(1).float().unsqueeze(1) * e_s
                last_e = (1-m) * last_e

            ent_probs.append(last_e.detach())
        return {
            'last_e': last_e,
            'rel_probs': rel_probs,
            'ent_probs': ent_probs
        }



    def forward(self, questions, e_s, answers = None):
        # question_lens = questions.size(1) - questions.eq(0).long().sum(dim=1) # 0 means <PAD>
        # q_word_emb = self.word_dropout(self.word_embeddings(questions)) # [bsz, max_q, dim_hidden]
        # q_word_h, q_embeddings, q_hn = self.question_encoder(q_word_emb, question_lens) # [bsz, max_q, dim_h], [bsz, dim_h], [num_layers, bsz, dim_h]
        if self.args.hist:
            outputs = self.hist_transfer(questions, e_s, answers)
        else:
            outputs = self.transfer(questions, e_s, answers) # [bsz, num_entities]

        e_score = outputs['last_e']
        normed_e_score = e_score / (e_score.sum(1, True) + 1e-6) # [bsz, num_entities]
        # detach e_score, so we just learn the classifier by cross entropy loss
        pred_e = normed_e_score.detach() @ self.entity_embeddings # [bsz, dim_hidden]
        pred_e = pred_e @ self.entity_embeddings.t() + self.entity_bias.unsqueeze(0) # [bsz, num_entities]

        if answers is None:
            return {
                'e_score': e_score,
                'pred_e': pred_e,
                'rel_probs': outputs['rel_probs'],
                'ent_probs': outputs['ent_probs']
            }
        # # CrossEntropy with most possible entity
        # pos_pred_e = pred_e * answers
        # idx = torch.argmax(pos_pred_e, 1) # [bsz]
        # if (idx==0).sum().item() > 0:
        #     idx[idx==0] = torch.argmax(answers[idx==0], 1)
        # loss_prob = nn.CrossEntropyLoss()(pred_e, idx)
        # print(answers.size())
        idx = torch.argmax(answers, 1)
        # print(idx.size())
        loss_prob = nn.CrossEntropyLoss()(pred_e, idx)

        # Distance loss
        weight = answers * 9 + 1
        # weight = answers * 0 + 1
        loss_score = torch.mean(weight * torch.pow(e_score - answers, 2))

        return {'loss_score': loss_score, 'loss_prob': loss_prob}
