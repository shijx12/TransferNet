import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.misc import MetricLogger, load_glove, idx_to_one_hot
from .data import DataLoader
from .model import TransferNet

from IPython import embed


def validate(args, model, data, device, verbose = False):
    vocab = data.vocab
    model.eval()
    count = 0
    correct = defaultdict(int)
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            sub, obj, rel = batch
            sub = idx_to_one_hot(sub.unsqueeze(1), len(vocab['entity2id'])).to(device)
            rel = rel.to(device)
            outputs = model(sub, rel)
            e_score = outputs['e_score'].cpu()
            sort_idx = torch.argsort(e_score, dim=1, descending=True)

            for i in range(len(batch)):
                answer = set([_ for _ in obj[i].tolist() if _ > 0])
                count += len(answer)
                for k in (1, 5, 10):
                    for j in range(k):
                        if sort_idx[i,j].item() in answer:
                            correct['hit@{}'.format(k)] += 1                        

    acc = {k:correct[k]/count for k in correct}
    result = ' | '.join(['%s:%.4f'%(key, value) for key, value in acc.items()])
    print(result)
    return acc


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required = True)
    parser.add_argument('--ckpt', required = True)
    parser.add_argument('--mode', default='val', choices=['val', 'vis', 'test'])
    # model hyperparameters
    parser.add_argument('--num_steps', default=3, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    val_loader = DataLoader(vocab_json, val_pt, 64, True)
    test_loader = DataLoader(vocab_json, test_pt, 64)
    vocab = val_loader.vocab

    model = TransferNet(args, args.dim_word, args.dim_hidden, vocab)
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.kg.Msubj = model.kg.Msubj.to(device)
    model.kg.Mobj = model.kg.Mobj.to(device)
    model.kg.Mrel = model.kg.Mrel.to(device)

    if args.mode == 'vis':
        validate(args, model, val_loader, device, True)
    elif args.mode == 'val':
        validate(args, model, val_loader, device, False)
    elif args.mode == 'test':
        validate(args, model, test_loader, device, False)

if __name__ == '__main__':
    main()
