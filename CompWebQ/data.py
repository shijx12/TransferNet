import torch
import os
import json
import pickle
from collections import defaultdict
from transformers import AutoTokenizer
from utils.misc import invert_dict

def collate(batch):
    batch = list(zip(*batch))
    topic_entity, question, answer, triples, entity_range = batch
    topic_entity = torch.stack(topic_entity)
    question = {k:torch.cat([q[k] for q in question], dim=0) for k in question[0]}
    answer = torch.stack(answer)
    entity_range = torch.stack(entity_range)
    return topic_entity, question, answer, triples, entity_range


class Dataset(torch.utils.data.Dataset):
    def __init__(self, questions, ent2id):
        self.questions = questions
        self.ent2id = ent2id

    def __getitem__(self, index):
        topic_entity, question, answer, triples, entity_range = self.questions[index]
        topic_entity = self.toOneHot(topic_entity)
        answer = self.toOneHot(answer)
        triples = torch.LongTensor(triples)
        if triples.dim() == 1:
            triples = triples.unsqueeze(0)
        entity_range = self.toOneHot(entity_range)
        return topic_entity, question, answer, triples, entity_range

    def __len__(self):
        return len(self.questions)

    def toOneHot(self, indices):
        indices = torch.LongTensor(indices)
        vec_len = len(self.ent2id)
        one_hot = torch.FloatTensor(vec_len)
        one_hot.zero_()
        one_hot.scatter_(0, indices, 1)
        return one_hot


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, fn, bert_name, ent2id, rel2id, batch_size, add_rev=False, training=False):
        print('Reading questions from {} {}'.format(fn, '(add reverse)' if add_rev else ''))
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name)
        self.ent2id = ent2id
        self.rel2id = rel2id
        self.id2ent = invert_dict(ent2id)
        self.id2rel = invert_dict(rel2id)

        data = []
        cnt_bad = 0
        for line in open(fn):
            instance = json.loads(line.strip())

            question = self.tokenizer(instance['question'].strip(), max_length=64, padding='max_length', return_tensors="pt")
            head = instance['entities']
            ans = [ent2id[a['kb_id']] for a in instance['answers']]
            triples = instance['subgraph']['tuples']

            if len(triples) == 0:
                continue

            sub_ents = set(t[0] for t in triples)
            obj_ents = set(t[2] for t in triples)
            entity_range = sub_ents | obj_ents

            is_bad = False
            if all(e not in entity_range for e in head):
                is_bad = True
            if all(e not in entity_range for e in ans):
                is_bad = True

            if is_bad:
                cnt_bad += 1

            if training and is_bad: # skip bad examples during training
                continue

            entity_range = list(entity_range)

            if add_rev:
                supply_triples = []
                # add self relation
                # for e in entity_range:
                #     supply_triples.append([e, self.rel2id['<self>'], e])
                # add reverse relation
                for s, r, o in triples:
                    rev_r = self.rel2id[self.id2rel[r]+'_rev']
                    supply_triples.append([o, rev_r, s])
                triples += supply_triples

            data.append([head, question, ans, triples, entity_range])

        print('data number: {}, bad number: {} (exluded in training)'.format(len(data), cnt_bad))
        
        dataset = Dataset(data, ent2id)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )

# need to download the data from https://github.com/RichardHGL/WSDM2021_NSM
def load_data(input_dir, bert_name, batch_size, add_rev=False):
    cache_fn = os.path.join(input_dir, 'cache{}.pt'.format('_rev' if add_rev else ''))
    if os.path.exists(cache_fn):
        print('Read from cache file: {} (NOTE: delete it if you modified data loading process)'.format(cache_fn))
        with open(cache_fn, 'rb') as fp:
            ent2id, rel2id, train_data, dev_data, test_data = pickle.load(fp)
        print('Train number: {}, dev number: {}, test number: {}'.format(
            len(train_data.dataset), len(dev_data.dataset), len(test_data.dataset)))
    else:
        print('Read data...')
        ent2id = {}
        for line in open(os.path.join(input_dir, 'entities.txt')):
            ent2id[line.strip()] = len(ent2id)
        print(len(ent2id))
        rel2id = {}
        for line in open(os.path.join(input_dir, 'relations.txt')):
            rel2id[line.strip()] = len(rel2id)
        # add self relation and reverse relation
        # rel2id['<self>'] = len(rel2id)
        if add_rev:
            for line in open(os.path.join(input_dir, 'relations.txt')):
                rel2id[line.strip()+'_rev'] = len(rel2id)
        print(len(rel2id))

        train_data = DataLoader(os.path.join(input_dir, 'train_simple.json'), bert_name, ent2id, rel2id, batch_size, add_rev=add_rev, training=True)
        dev_data = DataLoader(os.path.join(input_dir, 'dev_simple.json'), bert_name, ent2id, rel2id, batch_size, add_rev=add_rev)
        test_data = DataLoader(os.path.join(input_dir, 'test_simple.json'), bert_name, ent2id, rel2id, batch_size, add_rev=add_rev)

        with open(cache_fn, 'wb') as fp:
            pickle.dump((ent2id, rel2id, train_data, dev_data, test_data), fp)

    return ent2id, rel2id, train_data, dev_data, test_data



def cnt_hops(input_dir):
    def bfs(triples, start, end):
        if len(start)==0 or len(end)==0:
            return 1000, 1000

        hops = {i:0 for i in start}
        cur_set = set(start)
        next_set = set()
        for h in range(5):
            for s,r,o in triples:
                if s in cur_set and o not in hops:
                    hops[o] = h
                    next_set.add(o)
            cur_set = next_set
            next_set = set()
        only_forwad_res = min(hops.get(i,1000) for i in end)

        hops = {i:0 for i in start}
        cur_set = set(start)
        next_set = set()
        for h in range(5):
            for s,r,o in triples:
                if s in cur_set and o not in hops:
                    hops[o] = h
                    next_set.add(o)
                if o in cur_set and s not in hops:
                    hops[s] = h
                    next_set.add(s)
            cur_set = next_set
            next_set = set()
        add_reverse_res = min(hops.get(i,1000) for i in end)

        return only_forwad_res, add_reverse_res


    ent2id = {}
    for line in open(os.path.join(input_dir, 'entities.txt')):
        ent2id[line.strip()] = len(ent2id)


    for fn in ['train_simple.json', 'test_simple.json']:
        only_forward_res = defaultdict(int)
        add_reverse_res = defaultdict(int)
        for line in open(os.path.join(input_dir, fn)):
            instance = json.loads(line.strip())
            triples = instance['subgraph']['tuples']
            head = instance['entities']
            ans = [ent2id[a['kb_id']] for a in instance['answers']]
            i, j = bfs(triples, head, ans)
            only_forward_res[i] += 1
            add_reverse_res[j] += 1

        print(fn)
        print(only_forward_res)
        print(add_reverse_res) # increase the ratio of 1-hop

if __name__ == '__main__':
    cnt_hops('/data/sjx/dataset/WSDM_processed/CWQ')
