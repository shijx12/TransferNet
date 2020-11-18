import collections
import os
import pickle
from collections import defaultdict
import torch
import torch.nn as nn
from utils.misc import *
import numpy as np

class KnowledgeGraph(nn.Module):
    def __init__(self, args, vocab):
        super(KnowledgeGraph, self).__init__()
        self.args = args
        self.entity2id, self.id2entity = vocab['entity2id'], vocab['id2entity']
        self.relation2id, self.id2relation = vocab['relation2id'], vocab['id2relation']
        Msubj = torch.from_numpy(np.load(os.path.join(args.input_dir, 'Msubj.npy'))).long()
        Mobj = torch.from_numpy(np.load(os.path.join(args.input_dir, 'Mobj.npy'))).long()
        Mrel = torch.from_numpy(np.load(os.path.join(args.input_dir, 'Mrel.npy'))).long()
        Tsize = Msubj.size()[0]
        Esize = len(self.entity2id)
        Rsize = len(self.relation2id)
        self.Msubj = torch.sparse.FloatTensor(Msubj.t(), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
        self.Mobj = torch.sparse.FloatTensor(Mobj.t(), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Esize]))
        self.Mrel = torch.sparse.FloatTensor(Mrel.t(), torch.FloatTensor([1] * Tsize), torch.Size([Tsize, Rsize]))
        # self.num_relations = len(self.relation2id)
        self.num_entities = len(self.entity2id)
        # self.relation_embeddings = nn.Embedding(self.num_relations, args.dim_hidden)
        # self.entity_embeddings = nn.Embedding(self.num_entities, args.dim_hidden)
        # nn.init.xavier_normal_(self.relation_embeddings.weight)
        # if args.rel:
        #     # print('Loading pretrained relation embeddings')
        #     pretrained = torch.load('/data/csl/exp/AI_project/Bucket/pretrained_rel/transe_metaqa.ckpt')
        #     pretrained = pretrained['rel_embeddings.weight'].cpu().numpy()
        #     # self.relation_embeddings.weight.data[:1].fill_(0)
        #     self.relation_embeddings.weight.data.copy_(torch.from_numpy(pretrained))
    
    @property
    def dummy_e(self):
        return DUMMY_ENTITY_ID
    
    # @property
    # def dummy_r(self):
    #     return DUMMY_RELATION_ID
