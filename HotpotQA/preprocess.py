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
from .hotpot_evaluate_v1 import f1_score, normalize_answer

import spacy
from spacy.pipeline import EntityRuler
nlp = spacy.load('en_core_web_lg')
entity_ruler = EntityRuler(nlp)
nlp.add_pipe(entity_ruler, before='ner') # must before ner component

from IPython import embed


SUB_PH = '_sub_'
OBJ_PH = '_obj_'
ENT_PH = '_ent_'


tokenizer = None

def _process_article(inputs):
    article, args, selected_idx = inputs
    if 'supporting_facts' in article:
        sp_set = set(list(map(tuple, article['supporting_facts'])))
    else:
        sp_set = set()

    def tokenPat(s):
        # avoid that s is a substring of another token
        return r'(^|(?<=\W))' + re.escape(s) + r'((?=\W)|$)'

    def truncDesc(desc, max_l, center_word):
        desc = desc.split()
        if len(desc) > max_l:
            rm_l = len(desc) - max_l
            b = len(desc) - 2*desc.index(center_word)
            if b >= 0:
                if rm_l <= b:
                    start = 0
                else:
                    start = (rm_l-b)//2
            else:
                b = -b
                if rm_l <= b:
                    start = rm_l
                else:
                    start = b+ (rm_l-b)//2
            # print(' '.join(desc), ' ---> ', ' '.join(desc[start: start+max_l]))
            desc = desc[start: start+max_l]
        return ' '.join(desc)

    paragraphs = article['context']
    # some articles in the fullwiki dev/test sets have zero paragraphs
    if len(paragraphs) == 0:
        paragraphs = [['some random title', 'some random stuff']]

    patterns = []
    for para in paragraphs:
        # add all title into entity ruler
        cur_title, cur_para = para[0], para[1]
        subject = re.sub(r'\(.*\)$', '', cur_title.strip()).strip()
        patterns.append({'label': 'custom', 'pattern': subject})
    entity_ruler.add_patterns(patterns)

    triples = []
    entity_description = {}
    for idx, para in enumerate(paragraphs):
        if selected_idx and idx not in selected_idx:
            continue

        cur_title, cur_para = para[0], para[1]
        subject = re.sub(r'\(.*\)$', '', cur_title.strip()).strip()
        entity_description[normalize_answer(subject)] = ' '.join(cur_para)
        sub_pat = re.compile(tokenPat(subject), re.IGNORECASE) # to replace subject with placeholder
        for sent_id, sent in enumerate(cur_para):
            is_sup_fact = (cur_title, sent_id) in sp_set
            sent = sent.strip()

            for e in nlp(sent).ents:
                # print(sent)
                # print(e.text, e.start_char, e.end_char)
                if e.label_ in {'ORDINAL'}: # filter ents by e.label_
                    continue
                if normalize_answer(e.text) == normalize_answer(subject):
                    continue

                obj = e.text
                prefix = sent[:e.start_char]
                suffix = sent[e.end_char: ]
                # forward
                desc = sub_pat.sub(SUB_PH, prefix, 1) + ' '+OBJ_PH+' ' + sub_pat.sub(SUB_PH, suffix, 1)
                triples.append((
                    normalize_answer(subject), # normalize entities for better alignment
                    normalize_answer(obj),
                    truncDesc(desc, args.max_desc, OBJ_PH)
                    ))
                # backward
                desc = sub_pat.sub(OBJ_PH, prefix, 1) + ' '+SUB_PH+' ' + sub_pat.sub(OBJ_PH, suffix, 1)
                triples.append((
                    normalize_answer(obj),
                    normalize_answer(subject),
                    truncDesc(desc, args.max_desc, SUB_PH)
                    ))
    
    entity2id = {}
    for sub, obj, desc in triples:
        add_item_to_x2id(sub, entity2id)

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
    
    def _align_to_ent(entities, target):
        max_f1 = -1
        max_ent = None
        for e in entities:
            f1, p, r = f1_score(e, target)
            if f1 > max_f1:
                max_f1 = f1
                max_ent = e
        return max_ent, max_f1

    question = article['question']
    # align question topic entities with document entities
    # note that one question may have multiple topic entities !
    topic_ents = []
    # print(question, nlp(question).ents)
    for topic in nlp(question).ents:
        if topic.label_ in {'ORDINAL'}: # filter ents by e.label_
            continue
        e, f1 = _align_to_ent(entity2id.keys(), topic.text)
        if f1 > 0.5:
            topic_ents.append(e)
    if len(topic_ents) == 0:
        e, f1 = _align_to_ent(entity2id.keys(), question)
        topic_ents.append(e)
    topic_ent_idxs = [entity2id[e] for e in topic_ents]

    # collect description of topic entity
    topic_ent_desc = [entity_description.get(e, 'padding padding') for e in topic_ents]

    # align answer with document entities
    question_type = -1
    align_idx, answer = None, None
    if 'answer' in article:
        answer = article['answer']
        if answer in {'yes', 'no'}:
            question_type = 0 # predict yes/no according to each topic_ent_desc
            print(0, question)
        elif normalize_answer(answer) in topic_ents and len(topic_ents) == 2:
            question_type = 1 # given two topic entities, predict one of them, indicated by align_idx
            align_idx = 0 if normalize_answer(answer)==topic_ents[0] else 1
            print(1, question, topic_ents[align_idx], answer)
        else:
            question_type = 2 # predict from all entities
            align_answer, align_f1 = _align_to_ent(entity2id.keys(), answer)
            align_idx = entity2id[align_answer] 


    id2entity = {v:k for k,v in entity2id.items()}
    entities = [id2entity[i] for i in range(len(id2entity))]

    # TODO supporting fact
    """
    entities (list of str) : ordered entities
    so_pair (list of (int, int)) : subject index and object index of i-th triple
    descs (list of str) : description of i-th triple
    knowledge_range (list of (int, int)) : triple range of j-th entity
    question (str)
    question_type (int) : 0, 1, 2
    topic_ent_idxs (list of int) : aligned index of topic entities
    topic_ent_desc (list of str) : description of topic entities
    align_idx (int) : aligned entity index
    answer (str) : real answer
    """
    processed_document = {
        'entity': entities, 
        'kb_pair': so_pair, 
        'kb_desc': descs, 
        'kb_range': knowledge_range, 
        'question': question,
        'question_type': question_type,
        'topic_ent_idxs': topic_ent_idxs,
        'topic_ent_desc': topic_ent_desc,
        'answer_idx': align_idx, 
        'gold_answer': answer
    }
    return processed_document


def main():
    global tokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default = '/data/sjx/dataset/HotpotQA', type = str)
    parser.add_argument('--output_dir', default = '/data/sjx/exp/TransferNet/HotpotQA/input', type = str)
    parser.add_argument('--preselect_file', type=str, help='if provided, should be the file of selected idx')
    
    parser.add_argument('--max_desc', type=int, default=32, help='max number of words in description')
    parser.add_argument('--n_proc', type=int, default=16)
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # tokenizer_type = 'bert-base-cased'
    # print('Init tokenizer: {}'.format(tokenizer_type))
    # tokenizer = BertTokenizer.from_pretrained(tokenizer_type)

    # data = json.load(open(os.path.join(args.input_dir, 'hotpot_train_v1.1.json')))
    # embed()

    selected_idx = {}
    if args.preselect_file:
        select = json.load(open(args.preselect_file))
        selected_idx = { 'train':select[0], 'dev':select[1] }

    for split, fn in (('train', 'hotpot_train_v1.1.json'), ('dev', 'hotpot_dev_distractor_v1.json')):
        data = json.load(open(os.path.join(args.input_dir, fn)))
        print('number of {}: {}'.format(split, len(data)))
        select = selected_idx.get(split, [None]*len(data))
        # docs = []
        # for i in tqdm(range(100)):
        #     doc = _process_article(data[i], args)
        #     docs.append(doc)

        # data = data[:1000]
        with Pool(args.n_proc) as p:
            docs = list(tqdm(
                p.imap(_process_article, zip(data, [args]*len(data), select), chunksize=4), 
            total=len(data)))

        with open(os.path.join(args.output_dir, '{}.pt'.format(split)), 'wb') as f:
            pickle.dump(docs, f)

    # embed()


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

    min_cnt = 2
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
                s = [vocab.get(w, vocab.get('<UNK>')) for w in s] + [0]*(max_l-len(s))
                res.append(s)
            res = np.asarray(res, dtype=np.int32)
            return res
        elif isinstance(sents, str):
            s = word_tokenize(sents)
            s = [[vocab.get(w, vocab.get('<UNK>')) for w in s]]
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
    main()
    # build_vocab()
