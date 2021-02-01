import os
import ujson as json
import argparse
import random
import numpy as np
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from multiprocessing import Pool
from transformers import AutoTokenizer, AutoModel, AdamW, BatchEncoding
from .hotpot_evaluate_v1 import f1_score
from utils.misc import MetricLogger
from utils.lr_scheduler import get_linear_schedule_with_warmup
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
from IPython import embed


class RelevanceModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert_encoder = AutoModel.from_pretrained(args.model_type, return_dict=True)
        self.proj = nn.Sequential(
            nn.Linear(self.bert_encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            )

    def forward(self, data, device):
        for k,v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = data[k].to(device)
            elif isinstance(v, (dict, BatchEncoding)):
                data[k] = {x:y.to(device) for x,y in data[k].items()}
        q_emb = self.bert_encoder(**data['question']).pooler_output # (1, dim_h)
        q_article = self.bert_encoder(**data['articles']).pooler_output # (n_art, dim_h)
        logit = self.proj(q_emb * q_article).squeeze(1) # (n_art)
        score = torch.sigmoid(logit)
        # print(logit, score, data['labels'])
        loss = torch.nn.BCEWithLogitsLoss()(logit, data['labels'].float())
        return score, loss


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        doc = self.data[index]
        # tokenize sequences
        item = {}
        item['articles'] = self.tokenizer(doc['articles'], padding=True, truncation=True, return_tensors="pt")
        item['question'] = self.tokenizer(doc['question'], padding=True, truncation=True, return_tensors="pt")
        item['labels'] = torch.LongTensor(doc['labels'])
        return item

    def __len__(self):
        return len(self.data)

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, data, tokenizer, training=False, distributed=False):
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

        def collate(batch):
            return batch

        super().__init__(
            dataset,
            num_workers=0,
            batch_size=1, # per gpu
            sampler=sampler,
            collate_fn=collate, 
            )



def validate(args, model, val_loader, device):
    model.eval()
    recalls = []
    with torch.no_grad():
        for batch in tqdm(val_loader, total=len(val_loader)):
            score, loss = model(batch[0], device)
            select_idx = score.topk(min(args.topk, len(score)))[1].tolist()
            label = batch[0]['labels'].nonzero().squeeze(1).tolist()
            if set(label).issubset(set(select_idx)):
                recalls.append(1)
            else:
                recalls.append(0)
    recall = sum(recalls) / len(recalls)

    if args.distributed:
        recall = torch.Tensor([recall]).to(device)
        dist.all_reduce(recall)
        recall /= dist.get_world_size()
        recall = recall.item()
    return recall


def train(local_rank, args, dataset, model):
    print(local_rank)
    if args.distributed:
        dist.init_process_group(backend='gloo', init_method='env://',
        world_size=args.num_gpus, rank=local_rank)
        torch.cuda.set_device(local_rank)

    if torch.cuda.is_available():
        if local_rank != -1:
            device = 'cuda:{}'.format(local_rank)
        else:
            device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    trainset, devset = dataset
    train_loader = DataLoader(trainset, tokenizer, training=True, distributed=args.distributed)
    val_loader = DataLoader(devset, tokenizer, distributed=args.distributed)

    no_decay = ["bias", "LayerNorm.weight"]
    bert_param = [(n,p) for n,p in model.named_parameters() if n.startswith('bert_encoder')]
    other_param = [(n,p) for n,p in model.named_parameters() if not n.startswith('bert_encoder')]
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
    elif args.opt == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters)

    # Distributed training (should be after apex fp16 initialization)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    t_total = len(train_loader) * args.num_epoch // args.gradient_accumulation_steps
    args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    meters = MetricLogger(delimiter="  ")
    # recall = validate(args, model, val_loader, device)
    if local_rank in [-1, 0]:
        # print(recall)
        logging.info('start training, {} batch per gpu'.format(len(train_loader)))
    optimizer.zero_grad()
    for epoch in range(args.num_epoch):
        model.train()
        with torch.autograd.set_detect_anomaly(True):
            for iteration, batch in enumerate(train_loader):
                iteration = iteration + 1
                score, loss = model(batch[0], device)
                meters.update(loss=loss.item())

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.5)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)

                if iteration % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # if local_rank in [-1, 0] and iteration % (len(train_loader) // 100) == 0:
                if local_rank in [-1, 0] and iteration % 10 == 0:
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
        recall = validate(args, model, val_loader, device)
        if local_rank in [-1, 0]:
            logging.info(recall)
            torch.save(model.state_dict(), os.path.join(args.model_path, 'model_epoch-{}-{:.3f}.pt'.format(epoch, recall)))

    dist.destroy_process_group()




def read_article(inputs):
    article, args = inputs
    if 'supporting_facts' in article:
        sp_title = set([s[0] for s in article['supporting_facts']])
    else:
        sp_title = set()

    paragraphs = article['context']
    # some articles in the fullwiki dev/test sets have zero paragraphs
    if len(paragraphs) == 0:
        paragraphs = [['some random title', ['some random stuff']]]

    question = article['question']
    labels = []
    articles = []
    for para in paragraphs:
        cur_title, cur_para = para[0], para[1]
        label = 1 if cur_title in sp_title else 0
        article = ' '.join(cur_para)
        labels.append(label)
        articles.append(article)

    processed_document = {
        'question': question,
        'articles': articles,
        'labels': labels
    }
    return processed_document

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'val', 'infer'])

    parser.add_argument('--input_dir', default = '/data/sjx/dataset/HotpotQA', type = str)
    parser.add_argument('--model_path', default = '/data/sjx/exp/TransferNet/HotpotQA/retriever', type = str)
    parser.add_argument('--model_type', default='bert-base-cased', choices=['bert-base-cased', 'roberta-large', 'albert-base-v2'])
    parser.add_argument('--n_proc', type=int, default=1)

    # training parameters
    parser.add_argument('--num_gpus', type=int, help='number of gpus')
    parser.add_argument('--bert_lr', default=3e-5, type=float)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--opt', default='adam', type = str)
    parser.add_argument('--warmup_proportion', default=0.1, type = float)
    parser.add_argument('--topk', default=3, type=int)

    # infer parameters
    parser.add_argument('--ckpt_fn', type=str, help='checkpoint file name')
    parser.add_argument('--store_file', type=str)

    args = parser.parse_args()
    print(args)

    dataset = []
    for split, fn in (('train', 'hotpot_train_v1.1.json'), ('dev', 'hotpot_dev_distractor_v1.json')):
        data = json.load(open(os.path.join(args.input_dir, fn)))
        print('number of {}: {}'.format(split, len(data)))
        if args.n_proc == 1:
            docs = []
            for article in tqdm(data):
                docs.append(read_article((article, args)))
        else:
            with Pool(args.n_proc) as p:
                docs = list(tqdm(
                    p.imap(read_article, zip(data, [args]*len(data)), chunksize=4), 
                total=len(data)))
        dataset.append(docs)
    trainset, devset = dataset

    model = RelevanceModel(args)

    if args.mode == 'train':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if not os.path.isdir(args.model_path):
            os.makedirs(args.model_path)

        gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        args.num_gpus = len(gpus)
        args.distributed = args.num_gpus > 1
        if args.distributed:
            print('Distributed training!!')
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '9999'
            mp.spawn(train, nprocs=args.num_gpus, args=(args, dataset, model), join=True)
        else:
            train(-1, args, dataset, model)

    elif args.mode == 'val':
        args.distributed = False
        ckpt_fn = os.path.join(args.model_path, args.ckpt_fn)
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(ckpt_fn, map_location='cpu').items()})
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        trainset, devset = dataset
        val_loader = DataLoader(devset, tokenizer, distributed=False)

        recall = validate(args, model, val_loader, device)
        print(recall)

    elif args.mode == 'infer':
        args.distributed = False
        ckpt_fn = os.path.join(args.model_path, args.ckpt_fn)
        model.load_state_dict({k.replace('module.',''):v for k,v in torch.load(ckpt_fn, map_location='cpu').items()})
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        trainset, devset = dataset
        train_loader = DataLoader(trainset, tokenizer, distributed=False) # note: keep original order
        val_loader = DataLoader(devset, tokenizer, distributed=False)

        results = []
        for phase, dataloader in zip(('train', 'dev'), (train_loader, val_loader)):
            print('phase {}'.format(phase))
            select = []
            for batch in tqdm(dataloader):
                score, loss = model(batch[0], device)
                select_idx = score.topk(min(args.topk, len(score)))[1].tolist()
                select.append(select_idx)
            results.append(select)

        with open(os.path.join(args.model_path, 'selected_idx.json'), 'w') as f:
            json.dump(results, f)



if __name__ == '__main__':
    main()
