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
            if verbose:
                for i in range(len(sub)):
                    print('================================================================')
                    print('> head entity: {}'.format(vocab['id2entity'][sub[i].max(0)[1].item()]))
                    print('> query relation: {}'.format(vocab['id2relation'][rel[i].item()]))
                    for t in range(args.num_steps):
                        print('>>>>>>>>>> step {} <<<<<<<<<<'.format(t))
                        # for (si, ri, oi) in outputs['path_infos'][i][t]:
                        #     print('{} ---> {} ---> {}'.format(
                        #         vocab['id2entity'][si], vocab['id2relation'][ri], vocab['id2entity'][oi]
                        #         ))
                        for ri in outputs['path_infos'][i][t]:
                            print(vocab['id2relation'][ri])
                        print('> entity: {}'.format('; '.join([vocab['id2entity'][_] for _ in range(len(sub[i])) if outputs['ent_probs'][t+1][i][_].item() > 0.9])))
                    print('-----------')
                    print('> top 10 are {}'.format('; '.join([vocab['id2entity'][sort_idx[i, k].item()] for k in range(10)])))
                    print('> golden: {}'.format('; '.join([vocab['id2entity'][_] for _ in obj[i].tolist() if _ > 0])))
                    embed()
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
    parser.add_argument('--max_active', default=50, type=int, help='max number of active entities at each step')
    parser.add_argument('--dim_hidden', default=100, type=int)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    val_pt = os.path.join(args.input_dir, 'dev.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    val_loader = DataLoader(vocab_json, val_pt, 64, True)
    test_loader = DataLoader(vocab_json, test_pt, 64)
    vocab = val_loader.vocab

    model = TransferNet(args, vocab)
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.kb_triple = model.kb_triple.to(device)
    model.kb_range = model.kb_range.to(device)
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
