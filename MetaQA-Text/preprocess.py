import os
import json
import pickle
import argparse
import numpy as np
from nltk import word_tokenize
import collections
from collections import Counter, defaultdict
from itertools import chain
from tqdm import tqdm
from utils.misc import *
import re
import random


SUB_PH = '__subject__'
OBJ_PH = '__object__'
ENT_PH = '__entity__'
SELF_PH = '__self_rel__'


def encode_kb(args, vocab):
    def tokenPat(s):
        # avoid that s is a substring of another token
        return r'(^|(?<=\W))' + s + r'((?=\W)|$)'

    kb = defaultdict(list)
    for line in open(os.path.join(args.input_dir, 'kb/kb.txt')):
        s, r, o = line.strip().split('|')
        kb[s].append((r, o))

    # read from wiki
    triples = []
    cache = []
    for line in tqdm(chain(open(os.path.join(args.input_dir, 'kb/wiki.txt')), ['\n'])):
        line = line.strip()
        if line == '':
            if len(cache) == 0:
                continue
            subject = re.sub(r'\(.*\)', '', cache[0]).strip()
            for line in cache[1:]:
                # Note: force the match is in lower case, but keep subject and object in original case
                line = line.lower()
                # first replace subject with placeholder
                line = re.sub(tokenPat(subject.lower()), SUB_PH, line)
                used_objs = set([o for _, o in kb[subject] if re.search(tokenPat(o.lower()), line)])
                for obj in used_objs:
                    desc = line
                    # Note: some objects share the same name, so we must guarantee the OBJ_PH is placed before ENT_PH
                    desc = re.sub(tokenPat(obj.lower()), OBJ_PH, desc)
                    for other in used_objs-{obj}:
                        desc = re.sub(tokenPat(other.lower()), ENT_PH, desc)
                    # Note: operations to desc must be after placeholder
                    desc = desc.replace('/', ' / ').replace('–', ' - ').replace('-', ' - ')
                    tokens = word_tokenize(desc)
                    if OBJ_PH not in tokens:
                        # print()
                        # print(line)
                        # print(desc)
                        # print(obj)
                        # print(tokens)
                        # from IPython import embed; embed()
                        if len(obj) > 3:
                            tokens = word_tokenize(' '.join(tokens).replace(OBJ_PH, ' '+OBJ_PH+' '))
                        else:
                            continue

                    # filter out useless tokens
                    tokens = list(filter(lambda x: x not in {',', '.', 'and', ENT_PH}, tokens))
                    # truncate to max_desc
                    c = tokens.index(OBJ_PH)
                    if len(tokens) > args.max_desc:
                        tokens = tokens[max(c-args.max_desc//2, 0): c+args.max_desc//2]
                    triples.append((subject, obj, tokens))

                    backward_tokens = []
                    for t in tokens:
                        if t == SUB_PH:
                            backward_tokens.append(OBJ_PH)
                        elif t == OBJ_PH:
                            backward_tokens.append(SUB_PH)
                        else:
                            backward_tokens.append(t)
                    triples.append((obj, subject, backward_tokens))
            cache = []
        else:
            line = ' '.join(line.split()[1:])
            cache.append(line)

    # add structured knowledge based on required ratio
    if args.kb_ratio > 0:
        assert args.kb_ratio <= 1
        cnt = 0
        for s in kb:
            for (r, o) in kb[s]:
                if random.random() < args.kb_ratio:
                    triples.append((s, o, [r]))
                    cnt += 2
                    triples.append((o, s, [r+'_inv']))
        print('add {} ({}%) structured triples'.format(cnt, args.kb_ratio*100))

    # add self relation
    if args.add_self == 1:
        print('add self relations')
        entities = set()
        for sub, obj, desc in triples:
            entities.add(sub)
        for e in entities:
            triples.append((e, e, [SELF_PH]))
    else:
        print('NOT self relations')

    for tri in triples[:100]:
        print(tri)
    print('===')
    print('number of triples: {}'.format(len(triples)))


    triples = sorted(triples)
    # for tri in triples[:50]:
    #     print(tri)
    
    # build vocabulary
    word_counter = Counter()
    for sub, obj, desc in triples:
        add_item_to_x2id(sub, vocab['entity2id'])
        add_item_to_x2id(obj, vocab['entity2id'])
        word_counter.update(desc)
    cnt = 0
    for w, c in word_counter.items():
        if w and c >= args.min_cnt:
            add_item_to_x2id(w, vocab['word2id'])
        else:
            cnt += 1
    print('remove {} words whose frequency < {}'.format(cnt, args.min_cnt))
    print('vocabulary size: {} entities, {} words'.format(len(vocab['entity2id']), len(vocab['word2id'])))

    # [start, end) of each entity
    knowledge_range = np.full((len(vocab['entity2id']), 2), -1)
    start = 0
    for i in range(len(triples)):
        if i > 0 and triples[i][0] != triples[i-1][0]:
            idx = vocab['entity2id'][triples[i-1][0]]
            knowledge_range[idx] = (start, i)
            start = i
    idx = vocab['entity2id'][triples[-1][0]]
    knowledge_range[idx] = (start, len(triples))
    
    # Encode
    so_pair = [[vocab['entity2id'][s], vocab['entity2id'][o]] for s,o,_ in triples]
    descs = [[vocab['word2id'].get(w, vocab['word2id']['<UNK>']) for w in d] for _,_,d in triples]
    for d in descs:
        while len(d) < args.max_desc:
            d.append(vocab['word2id']['<PAD>'])

    so_pair = np.asarray(so_pair, dtype=np.int64)
    knowledge_range = np.asarray(knowledge_range, dtype=np.int64)
    descs = np.asarray(descs, dtype=np.int64)
    print(so_pair.shape, knowledge_range.shape, descs.shape)

    with open(os.path.join(args.output_dir, 'wiki.pt'), 'wb') as f:
        pickle.dump(so_pair, f)
        pickle.dump(knowledge_range, f)
        pickle.dump(descs, f)

    print('finish wiki process\n=====')


def encode_qa(args, vocab):
    pattern = re.compile(r'\[(.*)\]')
    hops = ['%d-hop'%((int)(num)) for num in args.num_hop.split(',')]
    drop_cnt = 0
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

                    # Note: some entities are not included in wiki
                    # assert topic_entity in vocab['entity2id']
                    # for answer in answers:
                    #     assert answer in vocab['entity2id']
                    answers = [a for a in answers if a in vocab['entity2id']]
                    if topic_entity not in vocab['entity2id'] or len(answers) == 0:
                        drop_cnt += 1
                        continue

                    data.append({'question':question, 'topic_entity':topic_entity, 'answers':answers, 'hop':int(hop[0])})
        datasets.append(data)
        json.dump(data, open(os.path.join(args.output_dir, '%s.json'%(dataset)), 'w'))

    train_set, test_set, val_set = datasets[0], datasets[1], datasets[2]
    print('size of training data: {}'.format(len(train_set)))
    print('size of test data: {}'.format(len(test_set)))
    print('size of valid data: {}'.format(len(val_set)))
    print('drop number: {}'.format(drop_cnt))
    print('=====')
    print('Build question vocabulary')
    word_counter = Counter()
    for qa in tqdm(train_set):
        tokens = word_tokenize(qa['question'].lower())
        word_counter.update(tokens)
    for w, c in word_counter.items():
        if w and c >= args.min_cnt:
            add_item_to_x2id(w, vocab['word2id'])
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
    parser.add_argument('--input_dir', required=True, type = str)
    parser.add_argument('--output_dir', required=True, type = str)
    parser.add_argument('--kb_ratio', type=float, default=0, 
        help='How many structured knowledge will be incorporated into textual knowledge. Note they are randomly selected.')
    parser.add_argument('--add_self', type = int, default = 0, help='whether add self relation, 0 means not')

    parser.add_argument('--min_cnt', type=int, default=5)
    parser.add_argument('--max_desc', type=int, default=16)
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
    }

    print('Encode kb')
    encode_kb(args, vocab)

    print('Encode qa')
    encode_qa(args, vocab)

if __name__ == '__main__':
    main()
