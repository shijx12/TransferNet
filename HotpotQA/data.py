import json
import pickle
import torch
import torch.distributed as dist
import numpy as np


def collate(batch):
    return batch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        entity = self.data[index]['entity']
        origin_entity = entity
        entity = self.tokenizer(entity, padding=True, return_tensors="pt")

        kb_pair = self.data[index]['kb_pair']
        kb_pair = torch.LongTensor(kb_pair)

        kb_desc = self.data[index]['kb_desc']
        kb_desc = self.tokenizer(kb_desc, padding=True, return_tensors="pt")

        kb_range = self.data[index]['kb_range']
        kb_range = torch.LongTensor(kb_range)

        question = self.data[index]['question']
        question = self.tokenizer(question, padding=True, return_tensors="pt")
        topic_ent_idxs = self.data[index]['topic_ent_idxs']

        answer_idx = self.data[index]['answer_idx']
        gold_answer = self.data[index]['gold_answer']
        return origin_entity, entity, kb_pair, kb_desc, kb_range, question, topic_ent_idxs, answer_idx, gold_answer


    def __len__(self):
        return len(self.data)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, question_pt, tokenizer, batch_size, training=False, distributed=False):
        
        with open(question_pt, 'rb') as f:
            data = pickle.load(f)

        dataset = Dataset(data, tokenizer)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank()
            )
        else:
            if training:
                sampler = torch.utils.data.sampler.RandomSampler(dataset)
            else:
                sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        super().__init__(
            dataset,
            num_workers=0,
            batch_size=batch_size, # per gpu
            sampler=sampler,
            collate_fn=collate, 
            )
