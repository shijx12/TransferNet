import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm
import time
from utils import MetricLogger, load_glove, idx_to_one_hot, UseStyle, RAdam
from data import DataLoader
from model import TransferNet
from predict import validate
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from IPython import embed
torch.set_num_threads(1) # avoid using multiple cpus


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logging.info("Create train_loader, val_loader and test_loader.........")
    vocab_json = os.path.join(args.input_dir, 'vocab.json')
    train_pt = os.path.join(args.input_dir, 'train.pt')
    val_pt = os.path.join(args.input_dir, 'val.pt')
    test_pt = os.path.join(args.input_dir, 'test.pt')
    train_loader = DataLoader(vocab_json, train_pt, args.batch_size, training=True)
    val_loader = DataLoader(vocab_json, val_pt, args.eval_batch_size)
    test_loader = DataLoader(vocab_json, test_pt, args.eval_batch_size)
    vocab = train_loader.vocab

    logging.info("Create model.........")
    pretrained = load_glove(args.glove_pt, vocab['id2word'])
    model = TransferNet(args, args.dim_word, args.dim_hidden, vocab)
    model.word_embeddings.weight.data = torch.Tensor(pretrained)
    if not args.ckpt == None:
        model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.kg.Msubj = model.kg.Msubj.to(device)
    model.kg.Mobj = model.kg.Mobj.to(device)
    model.kg.Mrel = model.kg.Mrel.to(device)

    logging.info(model)
    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'radam':
        optimizer = RAdam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4], gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    meters = MetricLogger(delimiter="  ")
    logging.info("Start training........")

    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1

            question, topic_entity, answer = batch
            question = question.to(device)
            topic_entity = idx_to_one_hot(topic_entity, len(vocab['entity2id'])).to(device)
            answer = idx_to_one_hot(answer, len(vocab['entity2id'])).to(device)
            answer[:, 0] = 0
            loss = model(question, topic_entity, answer)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.5)
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            meters.update(loss=loss.item())

            if iteration % (len(train_loader) // 100) == 0:
            # if True:
                
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
        scheduler.step(acc['e_score'])
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pt'))
        

def main():
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--input_dir', default = './input')
    parser.add_argument('--save_dir', required=True, help='path to save checkpoints and logs')
    parser.add_argument('--glove_pt', default='/data/csl/resources/word2vec/glove.840B.300d.py36.pt')
    parser.add_argument('--ckpt', default = None)
    # training parameters
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=60, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--eval_batch_size', default = 64, type = int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    parser.add_argument('--opt', default='radam', type = str)
    # model hyperparameters
    parser.add_argument('--dim_emb', default=300, type=int)
    parser.add_argument('--num_steps', default=3, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=1024, type=int)
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

    # set random seed
    torch.manual_seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
