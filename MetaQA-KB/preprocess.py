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
    with open(os.path.join(args.input_dir, 'kb/kb.txt')) as f:
        kb = f.readlines()
    
    Msubj = []
    Mobj = []
    Mrel = []
    idx = 0
    for line in tqdm(kb):
        s, r, o = line.strip().split('|')
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
    # r = '<SELF_REL>'
    # add_item_to_x2id(r, vocab['relation2id'])
    # r_id = vocab['relation2id'][r]
    # for i in vocab['entity2id'].values():
    #     Msubj.append([idx, i])
    #     Mobj.append([idx, i])
    #     Mrel.append([idx, r_id])
    #     idx += 1


    Tsize = len(Msubj)
    Esize = len(vocab['entity2id'])
    Rsize = len(vocab['relation2id'])
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
    pattern = re.compile(r'\[(.*?)\]')
    hops = ['%d-hop'%((int)(num)) for num in args.num_hop.split(',')]
    datasets = []
    for dataset in ['train', 'test', 'dev']:
        data = []
        for hop in hops:
            with open(os.path.join(args.input_dir, (hop + '/vanilla/qa_%s.txt'%(dataset)))) as f:
                qas = f.readlines()
                for qa in qas:
                    question, answers = qa.strip().split('\t')
                    topic_entity = re.search(pattern, question).group(1)
                    if args.replace_es:
                        question = re.sub(r'\[.*\]', 'E_S', question)
                    else:
                        question = question.replace('[', '').replace(']', '')
                    answers = answers.split('|')
                    assert topic_entity in vocab['entity2id']
                    for answer in answers:
                        assert answer in vocab['entity2id']
                    data.append({'question':question, 'topic_entity':topic_entity, 'answers':answers, 'hop':int(hop[0])})
        datasets.append(data)
        json.dump(data, open(os.path.join(args.output_dir, '%s.json'%(dataset)), 'w'))

    train_set, test_set, val_set = datasets[0], datasets[1], datasets[2]
    print('size of training data: {}'.format(len(train_set)))
    print('size of test data: {}'.format(len(test_set)))
    print('size of valid data: {}'.format(len(val_set)))
    print('Build question vocabulary')
    word_counter = Counter()
    for qa in tqdm(train_set):
        tokens = word_tokenize(qa['question'].lower())
        word_counter.update(tokens)
    stopwords = set()
    for w, c in word_counter.items():
        if w and c >= args.min_cnt:
            add_item_to_x2id(w, vocab['word2id'])
        if w and c >= args.stop_thresh:
            stopwords.add(w)
    print('number of stop words (>={}): {}'.format(args.stop_thresh, len(stopwords)))
    print('number of word in dict: {}'.format(len(vocab['word2id'])))
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
        questions.append([vocab['word2id'].get(w, vocab['word2id']['<UNK>']) for w in word_tokenize(qa['question'].lower())])
        topic_entities.append([vocab['entity2id'][qa['topic_entity']]])
        answers.append([vocab['entity2id'][answer] for answer in qa['answers']])
        hops.append(qa['hop'])
        
    # question padding
    max_len = max(len(q) for q in questions)
    print('max question length:{}'.format(max_len))
    for q in questions:
        while len(q) < max_len:
            q.append(vocab['word2id']['<PAD>'])
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
    parser.add_argument('--input_dir', default = '/data/csl/resources/KBQA_datasets/MetaQA', type = str)
    parser.add_argument('--output_dir', default = '/data/csl/exp/TransferNet/input', type = str)
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
        'word2id': init_word2id(),
        'entity2id': init_entity2id(),
        'relation2id': {},
        'topic_entity': {}
    }

    print('Encode kb')
    encode_kb(args, vocab)

    print('Encode qa')
    encode_qa(args, vocab)

if __name__ == '__main__':
    main()
