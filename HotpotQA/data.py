import json
import pickle
import torch
import numpy as np


def collate(batch):
    return batch[0]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, vocab, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.id2word = {v:k for k,v in vocab.items()}

    def __getitem__(self, index):
        entity = self.data[index]['entity']
        # origin_entity = entity
        # entity = self.tokenizer(entity, padding=True, return_tensors="pt")
        origin_entity = [' '.join([self.id2word[i] for i in idxs if i > 0]) for idxs in entity]
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
    def __init__(self, question_pt, vocab_json, tokenizer, batch_size=1, training=False):
        
        with open(question_pt, 'rb') as f:
            data = pickle.load(f)

        with open(vocab_json) as f:
            vocab = json.load(f)

        print('data number: {}'.format(len(data)))

        dataset = Dataset(data, vocab, tokenizer)

        shuffle = training
        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate, 
            )
