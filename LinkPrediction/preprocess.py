import os
import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
import collections
from collections import Counter
from itertools import chain
from tqdm import tqdm
from utils.misc import *
import re

def encode_kb(args, vocab):
    with open(os.path.join(args.input_dir, 'train.triples')) as f:
        kb = f.readlines()
    triples = []
    Msubj = []
    Mobj = []
    Mrel = []
    idx = 0
    for line in tqdm(kb):
        s, o, r = line.strip().split('\t')
        r_inv = r + '_inv'
        add_item_to_x2id(s, vocab['entity2id'])
        add_item_to_x2id(o, vocab['entity2id'])
        add_item_to_x2id(r, vocab['relation2id'])
        add_item_to_x2id(r_inv, vocab['relation2id'])
        s_id, r_id, o_id, r_inv_id = vocab['entity2id'][s], vocab['relation2id'][r], vocab['entity2id'][o], vocab['relation2id'][r_inv]
        triples.append((s_id, o_id, r_id))
        triples.append((o_id, s_id, r_inv_id))

        Msubj.append([idx, s_id])
        Mobj.append([idx, o_id])
        Mrel.append([idx, r_id])
        idx += 1
        Msubj.append([idx, o_id])
        Mobj.append([idx, s_id])
        Mrel.append([idx, r_inv_id])
        idx += 1
        
    # self relation
    r = '<SELF_REL>'
    add_item_to_x2id(r, vocab['relation2id'])
    r_id = vocab['relation2id'][r]
    for i in vocab['entity2id'].values():
        triples.append((i, i, r_id))

        Msubj.append([idx, i])
        Mobj.append([idx, i])
        Mrel.append([idx, r_id])
        idx += 1

    print('{} entities, {} relations, {} triples (including reverse and self)'.format(len(vocab['entity2id']), len(vocab['relation2id']), len(triples)))

    Msubj = np.array(Msubj, dtype = np.int32)
    Mobj = np.array(Mobj, dtype = np.int32)
    Mrel = np.array(Mrel, dtype = np.int32)
    np.save(os.path.join(args.output_dir, 'Msubj.npy'), Msubj)
    np.save(os.path.join(args.output_dir, 'Mobj.npy'), Mobj)
    np.save(os.path.join(args.output_dir, 'Mrel.npy'), Mrel)

    triples = sorted(triples)
    # [start, end) of each entity
    knowledge_range = np.full((len(vocab['entity2id']), 2), -1)
    start = 0
    for i in range(len(triples)):
        if i > 0 and triples[i][0] != triples[i-1][0]:
            idx = triples[i-1][0]
            knowledge_range[idx] = (start, i)
            start = i
    idx = triples[-1][0]
    knowledge_range[idx] = (start, len(triples))

    triples = np.asarray(triples, dtype=np.int64)
    knowledge_range = np.asarray(knowledge_range, dtype=np.int64)

    with open(os.path.join(args.output_dir, 'kb.pt'), 'wb') as f:
        pickle.dump(triples, f)
        pickle.dump(knowledge_range, f)

    with open(os.path.join(args.output_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f)


def encode_qa(args, vocab):
    datasets = []
    for dataset in ['train', 'test', 'dev']:
        data = defaultdict(list)
        with open(os.path.join(args.input_dir, '%s.triples'%(dataset))) as f:
            qas = f.readlines()
            print('original lines: {}'.format(len(qas)))
            for qa in qas:
                sub, obj, rel = qa.strip().split('\t')
                sub = vocab['entity2id'].get(sub, 0)
                obj = vocab['entity2id'].get(obj, 0)
                rel = vocab['relation2id'].get(rel, 0)
                data[(sub, rel)].append(obj)
        query, answer = [], []
        for k,v in data.items():
            query.append(k)
            answer.append(v)

        # pad answer
        max_ans = max(len(a) for a in answer)
        for i in range(len(answer)):
            while len(answer[i]) < max_ans:
                answer[i].append(0)

        query = np.asarray(query, dtype=np.int32)
        answer = np.asarray(answer, dtype=np.int32)
        print(query.shape)
        print(answer.shape)

        print('unseen numbers: {}'.format(np.sum(query==0)))
        with open(os.path.join(args.output_dir, '{}.pt'.format(dataset)), 'wb') as f:
            pickle.dump(query, f)
            pickle.dump(answer, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required = True, type = str)
    parser.add_argument('--output_dir', required = True, type = str)
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    print('Init vocabulary')
    vocab = {
        'entity2id': {
            '<UNK>': 0
        },
        'relation2id': {
            '<UNK>': 0
        }
    }

    print('Encode kb')
    encode_kb(args, vocab)

    print('Encode qa')
    encode_qa(args, vocab)

if __name__ == '__main__':
    main()
