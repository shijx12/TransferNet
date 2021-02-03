import os
import json
import torch
import torch.nn as nn
import torch.distributed as dist
import argparse
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
from utils.misc import batch_device
from .data import DataLoader
from .model import TransferNet
from .hotpot_evaluate_v1 import update_answer
from IPython import embed


def validate(args, model, data, device, verbose=False):
    model.eval()
    count = 0
    metrics = defaultdict(int)
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            assert len(batch) == 1
            batch = batch[0]
            # entity, ent_pos, pair_so, pair_pos, paragraphs, question, question_type, topic_ent_idxs, topic_ent_desc, answer_idx, gold_answer = batch
            # pred = entity[answer_idx] # upper bound using golden entity
            outputs = model(*batch_device(batch, device))
            pred = outputs['pred']
            count += 1
            update_answer(metrics, pred, batch[-1])

            if verbose:
                entity, ent_pos, pair_so, pair_pos, paragraphs, question, question_type, topic_ent_idxs, topic_ent_desc, answer_idx, gold_answer = batch
                d = data.dataset.data[count-1]
                print('================================================================')
                print(d['question'])
                print('topic entities: {}'.format('; '.join([entity[i] for i in topic_ent_idxs])))
                
                for t in range(args.num_steps):
                    print('\n>>>>>>>>>> step {} <<<<<<<<<<'.format(t))
                    for w in outputs['vis']['word_attns'][t]:
                        print(w)
                    print('---')
                    for path in outputs['vis']['path_infos'][t]:
                        print(path)
                    print('---')
                    for _ in range(len(entity)):
                        print('  {}: {}'.format(entity[_], outputs['vis']['ent_probs'][t][_].item()))
                print('-----------')
                print('hop attn: {}'.format(outputs['vis']['hop_attn'].tolist()))
                print('> golden: {}'.format(gold_answer))
                print('> prediction: {}'.format(pred))
                embed()
    for k in metrics.keys():
        metrics[k] /= count

    if args.distributed and dist.get_world_size() > 1:
        names = []
        values = []
        for k in sorted(metrics.keys()):
            names.append(k)
            values.append(metrics[k])
        values = torch.Tensor(values).to(device)
        dist.all_reduce(values)
        values /= dist.get_world_size()
        metrics = {k: v.item() for k, v in zip(names, values)}

    return metrics


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default='/data/sjx/exp/TransferNet/HotpotQA/input')
    parser.add_argument('--ckpt', required = True)
    parser.add_argument('--mode', default='val', choices=['val', 'vis', 'test'])
    # model hyperparameters
    parser.add_argument('--bert_type', default='albert-base-v2', choices=['bert-base-cased', 'albert-base-v2'])
    parser.add_argument('--num_steps', default=3, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=768, type=int)
    parser.add_argument('--keep_type', default=-1, type=int, help='keep only one question type, -1 means all')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, additional_special_tokens=['[left]', '[right]'])
    args.tokenizer = tokenizer
    val_pt = os.path.join(args.input_dir, 'dev.pt')
    val_loader = DataLoader(val_pt, tokenizer, 1, distributed=False, keep_type=args.keep_type)

    model = TransferNet(args)
    model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.ckpt, map_location='cpu').items()})
    model.to(device)
    args.distributed = False
    if args.mode == 'vis':
        validate(args, model, val_loader, device, True)
    elif args.mode == 'val':
        res = validate(args, model, val_loader, device, False)
        print(res)

if __name__ == '__main__':
    main()
