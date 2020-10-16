import torch
import torch.nn as nn
import math
import os
import pickle

from utils.BiGRU import GRU, BiGRU

class TransferNet(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.vocab = vocab
        self.max_active = args.max_active
        self.dim_hidden = args.dim_hidden
        dim_hidden = args.dim_hidden
        self.num_steps = args.num_steps

        with open(os.path.join(args.input_dir, 'kb.pt'), 'rb') as f:
            self.kb_triple = torch.LongTensor(pickle.load(f))
            self.kb_range = torch.LongTensor(pickle.load(f))

        num_entities = len(vocab['entity2id'])
        self.num_entities = num_entities
        num_relations = len(vocab['relation2id'])
        
        self.path_encoder = GRU(dim_hidden, dim_hidden, num_layers=1, dropout = 0.2)

        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.Tanh()
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)

        self.rel_classifier = nn.Linear(dim_hidden, 1)

        self.relation_embeddings = nn.Embedding(num_relations, dim_hidden)
        self.entity_embeddings = nn.Embedding(num_entities, dim_hidden)
        nn.init.normal_(self.entity_embeddings.weight, mean=0, std=1/math.sqrt(dim_hidden))
        nn.init.normal_(self.relation_embeddings.weight, mean=0, std=1/math.sqrt(dim_hidden))

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

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


    def forward(self, start, query, answer = None):
        '''
        Args:
            start [bsz, num_entities]: one hot vector of h for each query
            query [bsz]: r id for each query
        '''
        device = query.device
        bsz = query.size(0)
        last_e = start
        rel_probs = []
        ent_probs = [start]
        start_idx = torch.argmax(start, dim=1)
        q_emb = self.relation_embeddings(query)
        hist_emb = torch.zeros((bsz, self.num_entities, self.dim_hidden)).to(device)

        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_emb) # [bsz, dim_h]
            e_stack = []
            hist_stack = []
            for i in range(bsz):
                topk = 1 if t == 0 else 3
                e_idx = torch.topk(last_e[i], k=topk, dim=0)[1].tolist() + \
                        last_e[i].gt(0.9).nonzero().squeeze(1).tolist()
                rg = []
                for j in set(e_idx):
                    rg.append(torch.arange(self.kb_range[j,0], self.kb_range[j,1]).long().to(device))
                rg = torch.cat(rg, dim=0) # [rsz,]

                if len(rg) > self.max_active: # limit the number of next-hop
                    rg = rg[torch.randperm(len(rg))[:self.max_active]]

                # candidate triples to transfer
                triples = self.kb_triple[rg] # [rsz, 3]
                if self.training:
                    # remove the direct target link
                    sub, rel = triples[:,0], triples[:,2]
                    mask = sub.eq(start_idx[i]) & rel.eq(query[i])
                    triples = triples[~mask]
                sub, obj, rel = triples[:,0], triples[:,1], triples[:,2]

                sub_feat = torch.index_select(hist_emb[i], 0, sub) # [rsz, dim_h]
                rel_feat = self.relation_embeddings(rel) # [rsz, dim_h]
                trans_feat = self.path_encoder.forward_one_step(
                    rel_feat.unsqueeze(1), 
                    sub_feat.unsqueeze(0))[0].squeeze(1) # [rsz, dim_h]
                trans_prob = torch.sigmoid(self.rel_classifier(trans_feat * cq_t[i:i+1])).squeeze(1) # [rsz,]

                # transfer probability
                obj_p = last_e[i][sub] * trans_prob
                obj_ep = torch.index_add(torch.zeros_like(last_e[i]), 0, obj, obj_p)
                e_stack.append(obj_ep)

                # update history
                obj_feat = trans_feat * obj_p.unsqueeze(1) # [rsz, dim_h]
                # obj_feat = trans_feat
                new_hist = torch.index_add(torch.zeros_like(hist_emb[i]), 0, obj, obj_feat)
                hist_stack.append(new_hist)

            last_e = torch.stack(e_stack, dim=0)
            hist_emb = torch.stack(hist_stack, dim=0)

            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            ent_probs.append(last_e.detach())

        if answer is None:
            return {
                'e_score': last_e,
                'ent_probs': ent_probs
            }

        # weight = answer * 9 + 1
        weight = torch.zeros_like(answer)
        weight[answer==1] = 10
        neg_ratio = 0.01
        weight[answer==0] = (weight[answer==0].random_(0, to=1000)/1000).le(neg_ratio).float() # negative sample
        # from IPython import embed; embed()
        loss = torch.mean(weight * torch.pow(last_e - answer, 2))

        return loss
