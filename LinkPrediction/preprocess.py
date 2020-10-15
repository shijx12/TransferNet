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
        Msubj.append([idx, i])
        Mobj.append([idx, i])
        Mrel.append([idx, r_id])
        idx += 1


    Tsize = len(Msubj)
    Esize = len(vocab['entity2id'])
    Rsize = len(vocab['relation2id'])
    assert Tsize == len(kb) * 2 + Esize
    Msubj = np.array(Msubj, dtype = np.int32)
    Mobj = np.array(Mobj, dtype = np.int32)
    Mrel = np.array(Mrel, dtype = np.int32)
    assert len(Msubj) == Tsize
    assert len(Mobj) == Tsize
    assert len(Mrel) == Tsize
    np.save(os.path.join(args.output_dir, 'Msubj.npy'), Msubj)
    np.save(os.path.join(args.output_dir, 'Mobj.npy'), Mobj)
    np.save(os.path.join(args.output_dir, 'Mrel.npy'), Mrel)


    # Sanity check
    print('Sanity check: {} entities'.format(len(vocab['entity2id'])))
    print('Sanity check: {} relations'.format(len(vocab['relation2id'])))
    print('Sanity check: {} triples'.format(len(kb)))

def encode_qa(args, vocab):
    datasets = []
    for dataset in ['train', 'test', 'dev']:
        data = []
        with open(os.path.join(args.input_dir, '%s.triples'%(dataset))) as f:
            qas = f.readlines()
            for qa in qas:
                topic_entity, answer, question = qa.strip().split('\t')
                answers = [answer]
                data.append({'question':question, 'topic_entity':topic_entity, 'answers':answers, 'hop':int(0)})
        datasets.append(data)
        json.dump(data, open(os.path.join(args.output_dir, '%s.json'%(dataset)), 'w'))
    train_set, test_set, val_set = datasets[0], datasets[1], datasets[2]
    print('size of training data: {}'.format(len(train_set)))
    print('size of test data: {}'.format(len(test_set)))
    print('size of valid data: {}'.format(len(val_set)))
    with open(os.path.join(args.output_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=2)

    for name, dataset in zip(('train', 'val', 'test'), (train_set, val_set, test_set)):
        print('Encode {} set'.format(name))
        outputs = encode_dataset(vocab, dataset)
        print('shape of questions, topic_entities, answers, hops:')
        with open(os.path.join(args.output_dir, '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)

def encode_dataset(vocab, dataset):
    questions = []
    topic_entities = []
    answers = []
    hops = []
    for qa in tqdm(dataset):
        assert len(qa['topic_entity']) > 0
        questions.append([vocab['relation2id'].get(qa['question'], vocab['relation2id']['<UNK>'])])
        # questions.append([vocab['word2id'].get(w, vocab['word2id']['<UNK>']) for w in word_tokenize(qa['question'].lower())])
        topic_entities.append([vocab['entity2id'].get(qa['topic_entity'], vocab['entity2id']['<UNK>'])])
        # topic_entities.append([vocab['entity2id'][qa['topic_entity']]])
        answers.append([vocab['entity2id'].get(answer, vocab['entity2id']['<UNK>']) for answer in qa['answers']])
        hops.append(qa['hop'])
        
    # question padding
    # max_len = max(len(q) for q in questions)
    # print('max question length:{}'.format(max_len))
    # for q in questions:
    #     while len(q) < max_len:
    #         q.append(vocab['word2id']['<PAD>'])
    questions = np.asarray(questions, dtype=np.int32)
    topic_entities = np.asarray(topic_entities, dtype=np.int32)
    max_len = max(len(a) for a in answers)
    print('max answer length:{}'.format(max_len))
    for a in answers:
        while len(a) < max_len:
            a.append(DUMMY_ENTITY_ID)
    answers = np.asarray(answers, dtype=np.int32)
    hops = np.asarray(hops, dtype=np.int8)
    return questions, topic_entities, answers, hops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required = True, type = str)
    parser.add_argument('--output_dir', required = True, type = str)
    parser.add_argument('--min_cnt', type=int, default=1)
    parser.add_argument('--stop_thresh', type=int, default=1000)
    parser.add_argument('--num_hop', type = str, default = '1, 2, 3')
    parser.add_argument('--replace_es', type = int, default = 1)
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
        },
        'topic_entity': {}
    }

    print('Encode kb')
    encode_kb(args, vocab)

    print('Encode qa')
    encode_qa(args, vocab)

if __name__ == '__main__':
    main()