import json
import pickle
import torch
from utils.misc import invert_dict


def load_vocab(path):
    vocab = json.load(open(path))
    vocab['id2entity'] = invert_dict(vocab['entity2id'])
    vocab['id2relation'] = invert_dict(vocab['relation2id'])
    return vocab

def collate(batch):
    batch = list(zip(*batch))
    sub = torch.LongTensor(batch[0])
    obj = torch.stack(batch[1])
    rel = torch.LongTensor(batch[2])
    return sub, obj, rel


class Dataset(torch.utils.data.Dataset):
    def __init__(self, queries, answers):
        self.subs, self.rels = queries[:,0], queries[:,1]
        self.objs = answers

    def __getitem__(self, index):
        sub = self.subs[index]
        obj = torch.LongTensor(self.objs[index])
        rel = self.rels[index]
        return sub, obj, rel

    def __len__(self):
        return len(self.subs)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, training=False):
        vocab = load_vocab(vocab_json)
        
        with open(question_pt, 'rb') as f:
            queries = pickle.load(f)
            answers = pickle.load(f)
        print('data number: {}'.format(len(queries)))
        dataset = Dataset(queries, answers)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
        self.vocab = vocab
