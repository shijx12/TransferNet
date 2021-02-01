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
        self.max_length = self.tokenizer.model_max_length

    def __getitem__(self, index):
        entity = self.data[index]['entity']
        ent_pos = self.data[index]['ent_pos']
        for i in range(len(ent_pos)):
            ent_pos[i] = torch.LongTensor(ent_pos[i]) # each row means (p_idx, start, end)
            ent_pos[i][ent_pos[i] >= self.max_length] = self.max_length - 1 # avoid exceeding max_length

        pair_so = self.data[index]['pair_so']
        pair_pos = self.data[index]['pair_pos']
        pair_so = torch.LongTensor(pair_so)
        pair_pos = torch.LongTensor(pair_pos) # each row means (p_idx, sub_start, sub_end, obj_start, obj_end)
        pair_pos[pair_pos >= self.max_length] = self.max_length - 1

        paragraphs = self.data[index]['paragraphs']
        paragraphs = self.tokenizer(paragraphs, padding=True, truncation=True, return_tensors="pt")

        question = self.data[index]['question']
        question_type = self.data[index]['question_type']
        question = self.tokenizer(question, padding=True, return_tensors="pt")

        topic_ent_idxs = self.data[index]['topic_ent_idxs'] # idx in entity, not in paragraphs!
        topic_ent_desc = self.data[index]['topic_ent_desc']
        topic_ent_desc = self.tokenizer(topic_ent_desc, padding=True, truncation=True, return_tensors="pt")

        answer_idx = self.data[index]['answer_idx']
        gold_answer = self.data[index]['gold_answer']
        return entity, ent_pos, pair_so, pair_pos, paragraphs, question, question_type, topic_ent_idxs, topic_ent_desc, answer_idx, gold_answer


    def __len__(self):
        return len(self.data)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, question_pt, tokenizer, batch_size, training=False, distributed=False, keep_type=-1):
        
        with open(question_pt, 'rb') as f:
            data = pickle.load(f)

        if keep_type > -1:
            data = [d for d in data if d['question_type']==keep_type]

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
