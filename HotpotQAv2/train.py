import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import shutil
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
from utils.misc import MetricLogger, batch_device, RAdam
from utils.lr_scheduler import get_linear_schedule_with_warmup

from transformers import AutoTokenizer, AdamW

from .data import DataLoader
from .model import TransferNet
from .predict import validate
import logging

from IPython import embed
torch.set_num_threads(1) # avoid using multiple cpus


def train(local_rank, args):
    print(local_rank)
    if args.distributed:
        dist.init_process_group(backend='gloo', init_method='env://',
        world_size=args.num_gpus, rank=local_rank)
        torch.cuda.set_device(local_rank)

    logger = setup_logger('hotpotqa', args.save_dir, local_rank)
    for k, v in vars(args).items():
        logger.info(k+':'+str(v))

    logger.info('load data')
    if torch.cuda.is_available():
        if local_rank != -1:
            device = 'cuda:{}'.format(local_rank)
        else:
            device = 'cuda'
    else:
        device = 'cpu'
    logger.info('====== Note the bert type ({}) must be consistent with preprocess ======'.format(args.bert_type))
    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, additional_special_tokens=['[left]', '[right]'])
    args.tokenizer = tokenizer
    logger.info('max sequence length: {}'.format(tokenizer.model_max_length))

    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'dev.pt')
    train_loader = DataLoader(train_pt, tokenizer, 1, training=True, distributed=args.distributed, keep_type=args.keep_type)
    val_loader = DataLoader(val_pt, tokenizer, 1, distributed=args.distributed, keep_type=args.keep_type)
    logger.info('train loader: {}, val loader: {}'.format(len(train_loader), len(val_loader)))
    
    logger.info('build model')
    model = TransferNet(args)
    if not args.ckpt == None:
        model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    logger.info(model)

    no_decay = ["bias", "LayerNorm.weight"]
    bert_param = [(n,p) for n,p in model.named_parameters() if n.startswith('bert_encoder')]
    other_param = [(n,p) for n,p in model.named_parameters() if not n.startswith('bert_encoder')]
    logger.info('number of bert param: {}'.format(len(bert_param)))
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
    # optimizer_grouped_parameters = [{'params':model.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}]
    if args.opt == 'adam':
        optimizer = optim.Adam(optimizer_grouped_parameters)
    elif args.opt == 'radam':
        optimizer = RAdam(optimizer_grouped_parameters)
    elif args.opt == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters)

    if args.fp16:
        from torch.cuda import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                          # output_device=local_rank,
                                                          find_unused_parameters=True)

    t_total = len(train_loader) * args.num_epoch // args.gradient_accumulation_steps
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    meters = MetricLogger(delimiter="  ")
    # validate(args, model, val_loader, device)
    logger.info('Start training')
    optimizer.zero_grad()
    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1
            # print(iteration)
            assert len(batch) == 1
            batch = batch[0]
            loss = model(*batch_device(batch, device))
            if isinstance(loss, dict):
                total_loss = sum(loss.values())
                meters.update(**{k:v.item() for k,v in loss.items()})
            else:
                total_loss = loss
                meters.update(loss=loss.item())

            if args.gradient_accumulation_steps > 1:
                for k in loss:
                    loss[k] = loss[k] / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                nn.utils.clip_grad_norm_(amp.master_params(optimizer), 2.0)
            else:
                total_loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.5)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)

            if iteration % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if iteration % (len(train_loader) // 1000) == 0:
                logger.info(
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
        logger.info(acc)
        if local_rank in [-1, 0]:
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'model_epoch-{}_f1-{:.4f}.pt'.format(epoch, acc['f1'])))

    dist.destroy_process_group()



def setup_logger(name, save_dir, local_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if local_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--ckpt', default = None)
    # training parameters
    parser.add_argument('--num_gpus', type=int, help='number of gpus')
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--opt', default='radam', type = str)
    parser.add_argument('--warmup_proportion', default=0.1, type = float)
    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    # model hyperparameters
    parser.add_argument('--bert_type', default='albert-base-v2', choices=['bert-base-cased', 'albert-base-v2'])
    parser.add_argument('--num_steps', default=3, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=768, type=int)
    parser.add_argument('--keep_type', default=-1, type=int, help='keep only one question type, -1 means all')
    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    args.num_gpus = len(gpus)
    args.distributed = args.num_gpus > 1
    if args.distributed:
        print('Distributed training!!')
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '9999'
        mp.spawn(train, nprocs=args.num_gpus, args=(args,), join=True)
    else:
        train(-1, args)


if __name__ == '__main__':
    main()
