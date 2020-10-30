import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
import numpy as np
from tqdm import tqdm
import time
from utils.misc import MetricLogger, load_glove, idx_to_one_hot, batch_device, RAdam

from transformers import AutoTokenizer

from .data import DataLoader
from .model import TransferNet
from .predict import validate
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from IPython import embed
torch.set_num_threads(1) # avoid using multiple cpus


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # tokenizer = AutoTokenizer.from_pretrained(args.bert_type)
    tokenizer = None

    logging.info("Create train_loader, val_loader.........")
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'dev.pt')
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_loader = DataLoader(train_pt, vocab_json, tokenizer, args.batch_size, training=True)
    val_loader = DataLoader(val_pt, vocab_json, tokenizer, args.batch_size)

    logging.info("Create model.........")
    model = TransferNet(args)
    if not args.ckpt == None:
        logging.info("Load ckpt from {}".format(args.ckpt))
        model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    logging.info(model)

    no_decay = ["bias", "LayerNorm.weight"]
    bert_param = [(n,p) for n,p in model.named_parameters() if n.startswith('bert_encoder')]
    other_param = [(n,p) for n,p in model.named_parameters() if not n.startswith('bert_encoder')]
    print('number of bert param: {}'.format(len(bert_param)))
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.bert_lr},
        {'params': [p for n, p in bert_param if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0, 'lr': args.bert_lr},
        {'params': [p for n, p in other_param if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay, 'lr': args.lr},
        {'params': [p for n, p in other_param if any(nd in n for nd in no_decay)], 
        'weight_decay': 0.0, 'lr': args.lr},
        ]
    if args.opt == 'adam':
        optimizer = optim.Adam(optimizer_grouped_parameters)
    elif args.opt == 'radam':
        optimizer = RAdam(optimizer_grouped_parameters)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(optimizer_grouped_parameters)
    else:
        raise NotImplementedError

    meters = MetricLogger(delimiter="  ")
    # validate(args, model, val_loader, device)
    logging.info("Start training........")

    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1

            loss = model(*batch_device(batch, device))
            optimizer.zero_grad()
            if isinstance(loss, dict):
                total_loss = sum(loss.values())
                meters.update(**{k:v.item() for k,v in loss.items()})
            else:
                total_loss = loss
                meters.update(loss=loss.item())
            total_loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.5)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # if iteration % (len(train_loader) // 100) == 0:
            if True:
                logging.info(
                    meters.delimiter.join(
                        [
                            "progress: {progress:.3f}",
                            "{meters}",
                            "lr: {lr:.6f}",
                        ]
                    ).format(
                        progress=epoch + iteration / len(train_loader),
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
        
        acc = validate(args, model, val_loader, device)
        logging.info(acc)
        scheduler.step(acc['f1'])
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_epoch-{}_acc-{:.4f}.pt'.format(epoch, acc['all'])))
        

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default = None)
    # training parameters
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--opt', default='radam', type = str)
    # model hyperparameters
    parser.add_argument('--bert_type', default='albert-base-v2')
    parser.add_argument('--num_steps', default=3, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=768, type=int)
    args = parser.parse_args()

    # make logging.info display into both shell and file
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    args.log_name = time_ + '_{}_{}_{}.log'.format(args.opt, args.lr, args.batch_size)
    fileHandler = logging.FileHandler(os.path.join(args.save_dir, args.log_name))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    # args display
    for k, v in vars(args).items():
        logging.info(k+':'+str(v))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
