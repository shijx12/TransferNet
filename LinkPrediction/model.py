import torch
import torch.nn as nn
import math
import os
import pickle

from utils.BiGRU import GRU, BiGRU
from .Knowledge_graph import KnowledgeGraph
from IPython import embed

class TransferNet(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.vocab = vocab
        self.ent_act_thres = args.ent_act_thres
        self.max_active = args.max_active
        self.dim_hidden = args.dim_hidden
        dim_hidden = args.dim_hidden
        self.num_steps = args.num_steps
        self.kg = KnowledgeGraph(args, vocab)

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
        # self.rel_classifier = nn.Linear(dim_hidden, num_relations)

        self.relation_embeddings = nn.Embedding(num_relations, dim_hidden)
        self.entity_embeddings = nn.Embedding(num_entities, dim_hidden)
        nn.init.normal_(self.entity_embeddings.weight, mean=0, std=1/math.sqrt(dim_hidden))
        nn.init.normal_(self.relation_embeddings.weight, mean=0, std=1/math.sqrt(dim_hidden))

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

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

        path_infos = [] # [bsz, num_steps]
        for i in range(bsz):
            path_infos.append([])
            for j in range(self.num_steps):
                path_infos[i].append(None)

        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_emb) # [bsz, dim_h]

            
            e_stack = []
            hist_stack = []
            for i in range(bsz):
                # e_idx = torch.topk(last_e[i], k=1, dim=0)[1].tolist() + \
                #         last_e[i].gt(self.ent_act_thres).nonzero().squeeze(1).tolist()
                # DOING
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
                for j in set(e_idx):
                    rg.append(torch.arange(self.kb_range[j,0], self.kb_range[j,1]).long().to(device))
                rg = torch.cat(rg, dim=0) # [rsz,]

                # print(len(e_idx), len(rg))
                if len(rg) > self.max_active: # limit the number of next-hop
                    # DOING
                    rg = rg[:self.max_active]
                    # rg = rg[torch.randperm(len(rg))[:self.max_active]]

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
                trans_prob = torch.sigmoid(
                    self.rel_classifier(
                        trans_feat * cq_t[i:i+1]
                        # trans_feat * q_emb[i:i+1]
                        )
                    ).squeeze(1) # [rsz,]

                # transfer probability
                obj_p = last_e[i][sub] * trans_prob
                obj_ep = torch.index_add(torch.zeros_like(last_e[i]), 0, obj, obj_p)
                e_stack.append(obj_ep)

                # update history
                obj_feat = trans_feat * obj_p.unsqueeze(1) # [rsz, dim_h]
                # obj_feat = trans_feat
                new_hist = torch.index_add(torch.zeros_like(hist_emb[i]), 0, obj, obj_feat)
                new_hist = new_hist / (obj_ep.unsqueeze(1) + 1e-6)
                hist_stack.append(new_hist)
                # if t == 2:
                #     print(obj_ep)
                #     print(new_hist)
                #     embed()

                # collect path
                act_idx = obj_p.gt(0.9)
                act_sub, act_rel, act_obj = sub[act_idx], rel[act_idx], obj[act_idx]
                path_infos[i][t] = [(act_sub[i].item(), act_rel[i].item(), act_obj[i].item()) for i in range(len(act_sub))]

            last_e = torch.stack(e_stack, dim=0)
            hist_emb = torch.stack(hist_stack, dim=0)
            

            '''
            rel_dist = torch.sigmoid(self.rel_classifier(cq_t)) # [bsz, num_relations]
            e_stack = []
            for i in range(bsz):
                triples = self.kb_triple
                if self.training:
                    # remove the direct target link
                    sub, rel = triples[:,0], triples[:,2]
                    mask = sub.eq(start_idx[i]) & rel.eq(query[i])
                    triples = triples[~mask]

                sub, obj, rel = triples[:,0], triples[:,1], triples[:,2]
                obj_p = last_e[i][sub] * rel_dist[i][rel]
                obj_ep = torch.index_add(torch.zeros_like(last_e[i]), 0, obj, obj_p)
                e_stack.append(obj_ep)

                path_infos[i][t] = [r for r in range(len(rel_dist[0])) if rel_dist[i,r].item() > 0.8]

            last_e = torch.stack(e_stack, dim=0)
            # print(rel_dist[0])
            '''


            # rel_dist = torch.sigmoid(self.rel_classifier(cq_t)) # [bsz, num_relations]
            # x = torch.sparse.mm(self.kg.Msubj, last_e.t()) * torch.sparse.mm(self.kg.Mrel, rel_dist.t())
            # last_e = torch.sparse.mm(self.kg.Mobj.t(), x).t()
            # for i in range(bsz):
            #     path_infos[i][t] = [r for r in range(len(rel_dist[0])) if rel_dist[i,r].item() > 0.9]


            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            ent_probs.append(last_e.detach())

        if answer is None:
            return {
                'e_score': last_e,
                'ent_probs': ent_probs,
                'path_infos': path_infos
            }

        weight = answer * 9 + 1
        # weight = torch.zeros_like(answer)
        # weight[answer==1] = 10
        # neg_ratio = 1
        # weight[answer==0] = (weight[answer==0].random_(0, to=1000)/1000).le(neg_ratio).float() # negative sample
        # from IPython import embed; embed()
        loss = torch.mean(weight * torch.pow(last_e - answer, 2))

        return loss
