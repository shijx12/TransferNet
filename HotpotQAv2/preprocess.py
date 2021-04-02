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

from transformers import AutoTokenizer
from .hotpot_evaluate_v1 import f1_score, normalize_answer

import spacy
from spacy.pipeline import EntityRuler
nlp = spacy.load('en_core_web_lg')
entity_ruler = EntityRuler(nlp)
nlp.add_pipe(entity_ruler, before='ner') # must before ner component

from IPython import embed


tokenizer = None


def find_ent_position(para_token_ids, e, tokenizer):
    def find_sub_list(long_list, short_list):
        for i in range(len(long_list)-len(short_list)):
            if all(long_list[i+j]==short_list[j] for j in range(len(short_list))):
                return i, i+len(short_list)-1
        return -1, -1

    e_token_ids = tokenizer.encode('[left] ' + e.strip() + ' [right]', add_special_tokens=False)
    return find_sub_list(para_token_ids, e_token_ids)


def best_match(entities, target):
    max_f1 = -1
    max_ent = None
    for e in entities:
        f1, p, r = f1_score(e, target)
        if f1 > max_f1:
            max_f1 = f1
            max_ent = e
    return max_ent, max_f1


def process_article(inputs):
    article, args, selected_idx = inputs
    if 'supporting_facts' in article:
        sp_set = set(list(map(tuple, article['supporting_facts'])))
    else:
        sp_set = set()

    paragraphs = article['context']
    # some articles in the fullwiki dev/test sets have zero paragraphs
    if len(paragraphs) == 0:
        paragraphs = [['some random title', 'some random stuff']]

    patterns = []
    subjects = []
    for para in paragraphs:
        # add all title into entity ruler
        cur_title, cur_para = para[0], para[1]
        subject = re.sub(r'\(.*\)$', '', cur_title.strip()).strip()
        if len(subject) <= 1: # 特殊情况处理
            subject = cur_title.strip()
        subjects.append(subject)
        patterns.append({'label': 'custom', 'pattern': subject})
    entity_ruler.add_patterns(patterns)

    # select paragraphs
    if selected_idx is not None:
        paragraphs = [para for idx, para in enumerate(paragraphs) if idx in selected_idx]
        subjects = [s for idx, s in enumerate(subjects) if idx in selected_idx]

    # merge to title + sentences
    for i in range(len(paragraphs)):
        title, sents = subjects[i], paragraphs[i][1]
        paragraphs[i] = '{} : {}'.format(title, ' '.join(sents))

    # find all entities, and normalize them
    # insert [left]+[right] around entity, indicating left and right of an entity
    entity2id = {}
    ent_name_before_norm = []
    for i, para in enumerate(paragraphs):
        new_para = ''
        last_pos = 0
        for e in nlp(para).ents:
            # print(para)
            # print(e.text, e.start_char, e.end_char)
            if e.label_ in {'ORDINAL'}: # filter ents by e.label_
                continue
            add_item_to_x2id(normalize_answer(e.text), entity2id)
            new_para += para[last_pos:e.start_char] + ' [left] ' + e.text + ' [right] '
            last_pos = e.end_char
            ent_name_before_norm.append(e.text)
        new_para += para[last_pos:]
        new_para = ' '.join(new_para.strip().split())
        paragraphs[i] = new_para
    # Some entity may change after split+join, such as 'at \xa015'. We make ent_name and new_para consistent.
    ent_name_before_norm = [' '.join(e.strip().split()) for e in ent_name_before_norm]

    # tokenize paragraphs, collect the index of entity
    ent_pos = [[] for _ in entity2id]
    pair_so = []
    pair_pos = []
    for i, para in enumerate(paragraphs):
        para_token_ids = tokenizer(para)['input_ids']
        # subject id and position
        sub_id = entity2id[normalize_answer(subjects[i])]
        sub_start, sub_end = find_ent_position(para_token_ids, subjects[i], tokenizer)
        
        for e in ent_name_before_norm:
            e_id = entity2id[normalize_answer(e)]
            start, end = find_ent_position(para_token_ids, e, tokenizer)
            if start == -1:
                continue
            if start >= len(para_token_ids):
                print('why the start index exceeds paragraph length')
                continue
            # assert tokenizer.decode(para_token_ids[start])=='[left]' and tokenizer.decode(para_token_ids[end])=='[end]'
            if (i, start, end) not in ent_pos[e_id]:
                ent_pos[e_id].append((i, start, end)) # 文档序号，[left]位置，[right]位置

                if e_id != sub_id:
                    pair_so.append((sub_id, e_id))
                    pair_pos.append((i, sub_start, sub_end, start, end)) # 文档序号，主语[left]，主语[right]，宾语[left]，宾语[right]
                    pair_so.append((e_id, sub_id))
                    pair_pos.append((i, start, end, sub_start, sub_end)) # 反方向边
    assert all(len(p)>0 for p in ent_pos)


    question = article['question']
    # align question topic entities with document entities
    # note that one question may have multiple topic entities !
    topic_ents = []
    # print(question, nlp(question).ents)
    for topic in nlp(question).ents:
        if topic.label_ in {'ORDINAL'}: # filter ents by e.label_
            continue
        e, f1 = best_match(entity2id.keys(), topic.text)
        if f1 > 0.5:
            topic_ents.append(e)
    if len(topic_ents) == 0:
        e, f1 = best_match(entity2id.keys(), question)
        topic_ents.append(e)
    topic_ent_idxs = [entity2id[e] for e in topic_ents]

    # collect description of topic entity
    topic_ent_desc = []
    for e in topic_ents:
        d = 'padding padding'
        for i in range(len(paragraphs)):
            if normalize_answer(subjects[i]) == e:
                d = paragraphs[i]
        topic_ent_desc.append(d)

    # align answer with document entities
    question_type = -1
    align_idx, answer = None, None
    if 'answer' in article:
        answer = article['answer']
        if answer in {'yes', 'no'}:
            question_type = 0 # predict yes/no according to each topic_ent_desc
            align_idx = 0 if answer == 'yes' else 1
            # print(0, question)
        elif normalize_answer(answer) in topic_ents and len(topic_ents) == 2:
            question_type = 1 # given two topic entities, predict one of them, indicated by align_idx
            align_idx = 0 if normalize_answer(answer)==topic_ents[0] else 1
            # print(1, question, topic_ents[align_idx], answer)
        else:
            question_type = 2 # predict from all entities
            align_answer, align_f1 = best_match(entity2id.keys(), answer)
            align_idx = entity2id[align_answer] 


    id2entity = {v:k for k,v in entity2id.items()}
    entities = [id2entity[i] for i in range(len(id2entity))]

    # TODO supporting fact
    """
    entity (list of str) : ordered entities
    ent_pos (list of (list of (i, start, end)))
    pair_so (list of (int, int)) : subject index and object index of i-th triple
    pair_pos (list of (i, sub_start, sub_end, obj_start, obj_end))
    paragraphs (list of str) : several paragraphs, need to be tokenized using the same tokenizer with this script
    question (str)
    question_type (int) : 0, 1, 2
    topic_ent_idxs (list of int) : aligned index of topic entities
    topic_ent_desc (list of str) : description of topic entities
    align_idx (int) : aligned entity index
    answer (str) : real answer
    """
    processed_document = {
        'entity': entities, 
        'ent_pos': ent_pos,
        'pair_so': pair_so, 
        'pair_pos': pair_pos,
        'paragraphs': paragraphs,
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
    
    parser.add_argument('--n_proc', type=int, default=16)
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # tokenizer_type = 'bert-base-cased'
    tokenizer_type = 'albert-base-v2'
    print('Init tokenizer: {}'.format(tokenizer_type))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, additional_special_tokens=['[left]', '[right]'])

    selected_idx = {}
    if args.preselect_file:
        select = json.load(open(args.preselect_file))
        selected_idx = { 'train':select[0], 'dev':select[1] }

    for split, fn in (('train', 'hotpot_train_v1.1.json'), ('dev', 'hotpot_dev_distractor_v1.json')):
        data = json.load(open(os.path.join(args.input_dir, fn)))
        print('number of {}: {}'.format(split, len(data)))
        select = selected_idx.get(split, [None]*len(data))
        if args.n_proc == 1:
            docs = []
            for i in tqdm(range(len(data))):
                doc = process_article((data[i], args, select[i]))
                docs.append(doc)
        else:
            with Pool(args.n_proc) as p:
                docs = list(tqdm(
                    p.imap(process_article, zip(data, [args]*len(data), select), chunksize=4), 
                total=len(data)))

        with open(os.path.join(args.output_dir, '{}.pt'.format(split)), 'wb') as f:
            pickle.dump(docs, f)


if __name__ == '__main__':
    main()
