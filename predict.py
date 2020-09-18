import os
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
from utils import MetricLogger, load_glove, idx_to_one_hot
from data import DataLoader
from model import TransferNet

from IPython import embed


def validate(args, model, data, device, verbose = False):
    vocab = data.vocab
    model.eval()
    count = 0
    correct = {
        'e_score': 0,
        'pred_e': 0
    }
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            questions, topic_entities, answers = batch
            # print(answers)
            topic_entities = idx_to_one_hot(topic_entities, len(vocab['entity2id']))
            answers = idx_to_one_hot(answers, len(vocab['entity2id']))
            questions = questions.to(device)
            topic_entities = topic_entities.to(device)
            outputs = model(questions, topic_entities) # [bsz, Esize]
            pred_e = outputs['pred_e']
            e_score = outputs['e_score']
            scores, idx = torch.max(pred_e, dim = 1) # [bsz], [bsz]
            correct['pred_e'] += torch.gather(answers, 1, idx.unsqueeze(-1)).float().sum().item()
            scores, idx = torch.max(e_score, dim = 1) # [bsz], [bsz]
            correct['e_score'] += torch.gather(answers, 1, idx.unsqueeze(-1)).float().sum().item()
            count += len(answers)
            if verbose:
                for i in range(len(answers)):
                    if answers[i][idx[i]].item() == 0:
                        print('================================================================')
                        question = ' '.join([vocab['id2word'][_] for _ in questions.tolist()[i] if _ > 0])
                        print(question)
                        print('> topic entity: {}'.format(vocab['id2entity'][topic_entities[i].max(0)[1].item()]))
                        for t in range(args.num_steps):
                            print('> > > step {}'.format(t))
                            tmp = ' '.join(['{}: {:.3f}'.format(vocab['id2word'][x], y) for x,y in 
                                zip(questions.tolist()[i], outputs['word_attns'][t].tolist()[i]) 
                                if x > 0])
                            print('> ' + tmp)
                            tmp = ' '.join(['{}: {:.3f}'.format(vocab['id2relation'][x], y) for x,y in 
                                enumerate(outputs['rel_probs'][t].tolist()[i])])
                            print('> ' + tmp)
                            print('> entity: {}'.format('; '.join([vocab['id2entity'][_] for _ in range(len(answers[i])) if outputs['ent_probs'][t+1][i][_].item() > 0.9])))
                        print('----')
                        print('> max is {}'.format(vocab['id2entity'][idx[i].item()]))
                        print('> golden: {}'.format('; '.join([vocab['id2entity'][_] for _ in range(len(answers[i])) if answers[i][_].item() == 1])))
                        print('> prediction: {}'.format('; '.join([vocab['id2entity'][_] for _ in range(len(answers[i])) if e_score[i][_].item() > 0.9])))
                        embed()
    acc = {
        'pred_e': correct['pred_e'] / count,
        'e_score': correct['e_score'] / count
    }
    result = ' | '.join(['%s:%.4f'%(key, value) for key, value in acc.items()])
    logging.info(result)
    return acc


def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default = './input')
    parser.add_argument('--ckpt', required = True)
    # model hyperparameters
    parser.add_argument('--num_steps', default=3, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    val_loader = DataLoader(vocab_json, val_pt, 64)
    test_loader = DataLoader(vocab_json, test_pt, 64)
    vocab = val_loader.vocab

    model = TransferNet(args, args.dim_word, args.dim_hidden, vocab)
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.kg.Msubj = model.kg.Msubj.to(device)
    model.kg.Mobj = model.kg.Mobj.to(device)
    model.kg.Mrel = model.kg.Mrel.to(device)


    acc = validate(args, model, val_loader, device, True)

if __name__ == '__main__':
    main()
