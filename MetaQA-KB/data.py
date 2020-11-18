import json
import pickle
import torch
import numpy as np
from utils.misc import invert_dict


def load_vocab(path):
    vocab = json.load(open(path))
    vocab['id2word'] = invert_dict(vocab['word2id'])
    vocab['id2entity'] = invert_dict(vocab['entity2id'])
    vocab['id2relation'] = invert_dict(vocab['relation2id'])
    return vocab

def collate(batch):
    batch = list(zip(*batch))
    question, topic_entity, answer = list(map(torch.stack, batch[:3]))
    hop = torch.LongTensor(batch[3])
    return question, topic_entity, answer, hop


class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.questions, self.topic_entities, self.answers, self.hops = inputs
        # print(self.questions.shape)
        # print(self.topic_entities.shape)
        # print(self.answers.shape)

    def __getitem__(self, index):
        question = torch.LongTensor(self.questions[index])
        topic_entity = torch.LongTensor(self.topic_entities[index])
        answer = torch.LongTensor(self.answers[index])
        hop = self.hops[index]
        return question, topic_entity, answer, hop


    def __len__(self):
        return len(self.questions)


class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, vocab_json, question_pt, batch_size, ratio=1, training=False):
        vocab = load_vocab(vocab_json)
        
        inputs = []
        with open(question_pt, 'rb') as f:
            for _ in range(4):
                inputs.append(pickle.load(f))

        if ratio < 1:
            total = len(inputs[0])
            num = int(total * ratio)
            index = np.random.choice(total, num)
            print('random select {} of {} (ratio={})'.format(num, total, ratio))
            inputs = [i[index] for i in inputs]

        dataset = Dataset(inputs)

        super().__init__(
            dataset, 
            batch_size=batch_size,
            shuffle=training,
            collate_fn=collate, 
            )
        self.vocab = vocab

# if __name__ == '__main__':
#     vocab_json = '/data/csl/exp/AI_project/SRN/input/vocab.json'
#     question_pt = '/data/csl/exp/AI_project/SRN/input/train.pt'
#     inputs = []
#     with open(question_pt, 'rb') as f:
#         for _ in range(3):
#             inputs.append(pickle.load(f))
#     dataset = Dataset(inputs)
#     # print(dataset[0])
#     print(len(dataset))
#     question, topic_entity, answer = dataset[0]
#     print(question.size())
#     print(topic_entity.size())
#     print(answer.size())
