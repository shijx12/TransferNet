import os
import ujson as json
import argparse
import pickle
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from utils.misc import *
import re
from multiprocessing import Pool

from transformers import BertTokenizer
from .hotpot_evaluate_v1 import f1_score

import spacy
nlp = spacy.load('en')

from IPython import embed


SUB_PH = '__sub__'
OBJ_PH = '__obj__'
ENT_PH = '__ent__'
SELF_PH = '__self__'


tokenizer = None

def _process_article(inputs):
    article, args = inputs
    if 'supporting_facts' in article:
        sp_set = set(list(map(tuple, article['supporting_facts'])))
    else:
        sp_set = set()

    def tokenPat(s):
        # avoid that s is a substring of another token
        return r'(^|(?<=\W))' + re.escape(s) + r'((?=\W)|$)'

    paragraphs = article['context']
    # some articles in the fullwiki dev/test sets have zero paragraphs
    if len(paragraphs) == 0:
        paragraphs = [['some random title', 'some random stuff']]

    triples = []
    for para in paragraphs:
        cur_title, cur_para = para[0], para[1]
        subject = re.sub(r'\(.*\)', '', cur_title).strip()
        sub_pat = re.compile(tokenPat(subject), re.IGNORECASE) # to replace subject with placeholder
        for sent_id, sent in enumerate(cur_para):
            is_sup_fact = (cur_title, sent_id) in sp_set
            sent = sent.strip()

            for e in nlp(sent).ents:
                # print(sent)
                # print(e.text, e.start_char, e.end_char)
                # TODO filter ents by e.label_
                obj = e.text
                prefix = sent[max(0, e.start_char-args.max_desc//2):e.start_char]
                prefix = prefix[prefix.find(' ')+1:]
                suffix = sent[e.end_char: min(len(sent), e.end_char+args.max_desc//2)]
                suffix = suffix[:suffix.rfind(' ')]
                # forward
                triples.append((
                    subject,
                    obj,
                    sub_pat.sub(SUB_PH, prefix, 1) + OBJ_PH + sub_pat.sub(SUB_PH, suffix, 1)
                    ))
                # backward
                triples.append((
                    obj,
                    subject,
                    sub_pat.sub(OBJ_PH, prefix, 1) + SUB_PH + sub_pat.sub(OBJ_PH, suffix, 1)
                    ))
    
    entity2id = {}
    for sub, obj, desc in triples:
        add_item_to_x2id(sub, entity2id)

    # add self relation
    for e in entity2id:
        triples.append((e, e, SELF_PH))

    triples = sorted(triples)
    # print(triples)
    # print(len(entity2id))
    # print(len(triples))

    # [start, end) of each entity
    knowledge_range = np.full((len(entity2id), 2), -1)
    start = 0
    for i in range(len(triples)):
        if i > 0 and triples[i][0] != triples[i-1][0]:
            idx = entity2id[triples[i-1][0]]
            knowledge_range[idx] = (start, i)
            start = i
    idx = entity2id[triples[-1][0]]
    knowledge_range[idx] = (start, len(triples))

    so_pair = [(entity2id[s], entity2id[o]) for s,o,_ in triples]
    descs = [d for _,_,d in triples]
    
    def _align_to_ent(entities, answer):
        max_f1 = -1
        max_ent = None
        for e in entities:
            f1, p, r = f1_score(e, answer)
            if f1 > max_f1:
                max_f1 = f1
                max_ent = e
        return max_ent, max_f1

    question = article['question']
    if 'answer' in article:
        answer = article['answer']
        align_answer, align_f1 = _align_to_ent(entity2id.keys(), answer)
        align_idx = entity2id[align_answer]
    else:
        align_idx, answer = None, None
    
    # TODO yes/no questions
    # print('{}; {}; {}'.format(align_answer, answer, align_f1))
    # if answer in ['yes', 'no', 'noanswer']:
    #     print(question)

    id2entity = {v:k for k,v in entity2id.items()}
    entities = [id2entity[i] for i in range(len(id2entity))]

    # tokenize sequences
    # token_entities = tokenizer(entities, padding=True, return_tensors="pt")
    # token_descs = tokenizer(descs, padding=True, return_tensors="pt")
    # token_question = tokenizer(question, padding=True, return_tensors="pt")

    # TODO supporting fact
    """
    entities (list of str) : ordered entities
    so_pair (list of (int, int)) : subject index and object index of i-th triple
    descs (list of str) : description of i-th triple
    knowledge_range (list of (int, int)) : triple range of j-th entity
    question (str)
    align_idx (int) : aligned entity index
    answer (str) : real answer
    """
    processed_document = {
        'entity': entities, 
        'kb_pair': so_pair, 
        'kb_desc': descs, 
        'kb_range': knowledge_range, 
        'question': question, 
        'answer_idx': align_idx, 
        'gold_answer': answer
    }
    return processed_document


def main():
    global tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = '/data/sjx/dataset/HotpotQA', type = str)
    parser.add_argument('--output_dir', default = '/data/sjx/exp/TransferNet/HotpotQA/input', type = str)
    
    parser.add_argument('--max_desc', type=int, default=140)
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer_type = 'bert-base-cased'
    print('Init tokenizer: {}'.format(tokenizer_type))
    tokenizer = BertTokenizer.from_pretrained(tokenizer_type)

    # data = json.load(open(os.path.join(args.input_dir, 'hotpot_train_v1.1.json')))
    # embed()

    for split, fn in (('train', 'hotpot_train_v1.1.json'), ('dev', 'hotpot_dev_distractor_v1.json')):
        data = json.load(open(os.path.join(args.input_dir, fn)))
        print('number of {}: {}'.format(split, len(data)))
        # docs = []
        # for i in tqdm(range(100)):
        #     doc = _process_article(data[i], args)
        #     docs.append(doc)

        # data = data[:1000]
        with Pool(10) as p:
            docs = list(tqdm(
                p.imap(_process_article, zip(data, [args]*len(data)), chunksize=4), 
            total=len(data)))

        with open(os.path.join(args.output_dir, '{}.pt'.format(split)), 'wb') as f:
            pickle.dump(docs, f)

    embed()


def build_vocab():
    from nltk import word_tokenize
    vocab = init_word2id()
    fn = '/data/sjx/exp/TransferNet/HotpotQA/input/train.pt'
    with open(fn, 'rb') as f:
        data = pickle.load(f)

    print('Build vocabulary')
    word_counter = Counter()
    for doc in tqdm(data):
        for e in doc['entity']:
            word_counter.update(word_tokenize(e))
        for d in doc['kb_desc']:
            word_counter.update(word_tokenize(d))
        word_counter.update(word_tokenize(doc['question']))

    min_cnt = 3
    cnt = 0
    for w, c in word_counter.items():
        if w and c >= min_cnt:
            add_item_to_x2id(w, vocab)
        else:
            cnt += 1
    print('remove {} words whose frequency < {}'.format(cnt, min_cnt))
    print('vocabulary size: {}'.format(len(vocab)))

    with open('/data/sjx/exp/TransferNet/HotpotQA/input/vocab.json', 'w') as f:
        json.dump(vocab, f)

    def tokenize_encode_pad(sents, vocab):
        if isinstance(sents, list):
            sents = [word_tokenize(s) for s in sents]
            max_l = max(len(s) for s in sents)
            res = []
            for s in sents:
                s = [vocab.get(w, 0) for w in s] + [0]*(max_l-len(s))
                res.append(s)
            res = np.asarray(res, dtype=np.int32)
            return res
        elif isinstance(sents, str):
            s = word_tokenize(sents)
            s = [[vocab.get(w, 0) for w in s]]
            s = np.asarray(s, dtype=np.int32)
            return s

    print('Encode train set')
    for doc in tqdm(data):
        doc['entity'] = tokenize_encode_pad(doc['entity'], vocab)
        doc['kb_desc'] = tokenize_encode_pad(doc['kb_desc'], vocab)
        doc['question'] = tokenize_encode_pad(doc['question'], vocab)
    with open('/data/sjx/exp/TransferNet/HotpotQA/input/train_encode.pt', 'wb') as f:
        pickle.dump(data, f)

    print('Encode val set')
    with open('/data/sjx/exp/TransferNet/HotpotQA/input/dev.pt', 'rb') as f:
        data = pickle.load(f)
    for doc in tqdm(data):
        doc['entity'] = tokenize_encode_pad(doc['entity'], vocab)
        doc['kb_desc'] = tokenize_encode_pad(doc['kb_desc'], vocab)
        doc['question'] = tokenize_encode_pad(doc['question'], vocab)
    with open('/data/sjx/exp/TransferNet/HotpotQA/input/dev_encode.pt', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    # main()
    build_vocab()
