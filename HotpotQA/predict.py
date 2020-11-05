import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from collections import defaultdict
from utils.misc import MetricLogger, load_glove, idx_to_one_hot, batch_device
from .data import DataLoader
from .model import TransferNet
from .hotpot_evaluate_v1 import update_answer
from IPython import embed


def validate(args, model, data, device, verbose = False):
    model.eval()
    count = 0
    metrics = defaultdict(int)
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            for b in batch:
                outputs = model(*batch_device(b, device))
                pred = outputs['pred']
                count += 1
                update_answer(metrics, pred, b[-1])

                if verbose:
                    vocab = data.vocab
                    origin_entity, entity, kb_pair, kb_desc, kb_range, question, answer_idx, gold_answer = b
                    print('================================================================')
                    question_str = ' '.join([vocab['id2word'][_] for _ in question[0].tolist() if _ > 0])
                    print(question_str)
                    
                    for t in range(args.num_steps):
                        print('>>>>>>>>>> step {} <<<<<<<<<<'.format(t))
                        tmp = ' '.join(['{}: {:.3f}'.format(vocab['id2word'][x], y) for x,y in 
                            zip(question[0].tolist(), outputs['word_attns'][t][0].tolist()) 
                            if x > 0])
                        print('> ' + tmp)
                        print('---------')
                        for path in outputs['path_infos'][t]:
                            print(path)
                        print('> active entity:')
                        for _ in range(len(entity)):
                            if outputs['ent_probs'][t][_].item() > 0.5:
                                print('  {}: {}'.format(origin_entity[_], outputs['ent_probs'][t][_].item()))
                    print('-----------')
                    print('> golden: {}'.format(gold_answer))
                    print('> prediction: {}'.format(pred))
                    embed()
    for k in metrics.keys():
        metrics[k] /= count
    return metrics


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default = './input')
    parser.add_argument('--ckpt', required = True)
    parser.add_argument('--mode', default='val', choices=['val', 'vis', 'test'])
    # model hyperparameters
    parser.add_argument('--num_steps', default=3, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=768, type=int)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    val_pt = os.path.join(args.input_dir, 'dev_encode.pt')
    val_loader = DataLoader(val_pt, vocab_json, None, 1)
    vocab = val_loader.vocab

    model = TransferNet(args, vocab)
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)

    if args.mode == 'vis':
        validate(args, model, val_loader, device, True)
    elif args.mode == 'val':
        validate(args, model, val_loader, device, False)

if __name__ == '__main__':
    main()
