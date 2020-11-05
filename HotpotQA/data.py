import json
import pickle
import torch
import numpy as np


def collate(batch):
    return batch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, vocab, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __getitem__(self, index):
        entity = self.data[index]['entity']
        # origin_entity = entity
        # entity = self.tokenizer(entity, padding=True, return_tensors="pt")
        origin_entity = [' '.join([self.vocab['id2word'][i] for i in idxs if i > 0]) for idxs in entity]
        entity = torch.LongTensor(entity)

        kb_pair = self.data[index]['kb_pair']
        kb_pair = torch.LongTensor(kb_pair)

        kb_desc = self.data[index]['kb_desc']
        # kb_desc = self.tokenizer(kb_desc, padding=True, return_tensors="pt")
        kb_desc = torch.LongTensor(kb_desc)

        kb_range = self.data[index]['kb_range']
        kb_range = torch.LongTensor(kb_range)

        question = self.data[index]['question']
        # question = self.tokenizer(question, padding=True, return_tensors="pt")
        question = torch.LongTensor(question)

        answer_idx = self.data[index]['answer_idx']
        gold_answer = self.data[index]['gold_answer']
        return origin_entity, entity, kb_pair, kb_desc, kb_range, question, answer_idx, gold_answer


    def __len__(self):
        return len(self.data)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, question_pt, vocab_json, tokenizer, batch_size, training=False):
        
        with open(question_pt, 'rb') as f:
            data = pickle.load(f)

        with open(vocab_json) as f:
            word2id = json.load(f)
            id2word = {v:k for k,v in word2id.items()}
            vocab = {
                'word2id': word2id,
                'id2word': id2word
            }

        # filter empty entity
        filter_idx = set()
        for i in range(len(data)):
            if any(data[i]['entity'].sum(1)==0):
                filter_idx.add(i)
        data = [data[i] for i in range(len(data)) if i not in filter_idx]

        print('data number: {}, filter empty: {}'.format(len(data), len(filter_idx)))

        dataset = Dataset(data, vocab, tokenizer)

        shuffle = training
        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate, 
            )
        self.vocab = vocab
