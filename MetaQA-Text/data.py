import json
import pickle
import torch
import numpy as np
from utils.misc import invert_dict


def load_vocab(path):
    vocab = json.load(open(path))
    vocab['id2word'] = invert_dict(vocab['word2id'])
    vocab['id2entity'] = invert_dict(vocab['entity2id'])
    return vocab

def collate(batch):
    batch = list(zip(*batch))
    question, topic_entity, answer = list(map(torch.stack, batch[:3]))
    hop = torch.LongTensor(batch[3])
    return question, topic_entity, answer, hop


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.questions, self.topic_entities, self.answers, self.hops = inputs

    def __getitem__(self, index):
        question = torch.LongTensor(self.questions[index])
        topic_entity = torch.LongTensor(self.topic_entities[index])
        answer = torch.LongTensor(self.answers[index])
        hop = self.hops[index]
        return question, topic_entity, answer, hop


    def __len__(self):
        return len(self.questions)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, limit_hop=-1, training=False, curriculum=False):
        vocab = load_vocab(vocab_json)
        
        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(4):
                inputs.append(pickle.load(f))

        if limit_hop > 0:
            print('only keep questions of hop {}'.format(limit_hop))
            mask = inputs[3] == limit_hop
            inputs = [i[mask] for i in inputs]
            curriculum = False

        if curriculum:
            print('curriculum')
            hops = inputs[3]
            idxs = []
            for h in [1, 2, 3]:
                idx = np.nonzero(hops==h)[0]
                np.random.shuffle(idx)
                idxs.append(idx)
            idxs = np.concatenate(idxs)
            inputs = [i[idxs] for i in inputs]

        print('data number: {}'.format(len(inputs[0])))

        dataset = Dataset(inputs)

        shuffle = training
        if curriculum:
            shuffle = False
        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate, 
            )
        self.vocab = vocab
