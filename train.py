import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import shutil
from tqdm import tqdm
import time
from utils import MetricLogger, load_glove, idx_to_one_hot, UseStyle
from data import DataLoader
from model import TransferNet
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logFormatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
rootLogger = logging.getLogger()

from IPython import embed

torch.set_num_threads(1) # avoid using multiple cpus

def check(question, topic_entity, answer, vocab):
    print(question.size())
    print(topic_entity.size())
    print(answer.size())
    question = question.cpu().tolist()[0]
    topic_entity = topic_entity.cpu().tolist()[0]
    answer = answer.cpu().tolist()[0]
    print(len(question))
    print(len(topic_entity))
    print(len(answer))
    question = ' '.join([vocab['id2word'][x] for x in question])
    print(question)
    print('topic_entity: ')
    for i in range(len(topic_entity)):
        if topic_entity[i] == 1:
            print(vocab['id2entity'][i])
    print('answers:')
    for i in range(len(answer)):
        if answer[i] == 1:
            print(vocab['id2entity'][i])
            
def validate(model, data, device, verbose = True):
    vocab = data.vocab
    model.eval()
    count = 0
    correct = {
        'e_score': 0,
        'pred_e': 0
    }
    attn_list = []
    with torch.no_grad():
        for batch in tqdm(data, total=len(data)):
            questions, topic_entities, answers = batch
            # print(answers)
            topic_entities = idx_to_one_hot(topic_entities, len(vocab['entity2id']))
            answers = idx_to_one_hot(answers, len(vocab['entity2id']))
            answers[:, 0] = 0
            questions = questions.to(device)
            answers = answers.to(device)
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
                attn_list.append(outputs['attn'])
    acc = {
        'pred_e': correct['pred_e'] / count,
        'e_score': correct['e_score'] / count
    }
    result = ' | '.join(['%s:%.4f'%(key, value) for key, value in acc.items()])
    logging.info(result)
    if verbose:
        attn_list = list(map(torch.cat, list(zip(*attn_list))))
        scores = {}
        indices = {} 
        for idx, attn in enumerate(attn_list):
            score, indice = torch.sort(attn, descending=True)
            score = score.cpu().tolist()
            indice = indice.cpu().tolist()
            for i in range(len(score)):
                for j in range(len(score[0])):
                    score[i][j] = round(score[i][j], 3)

            scores['attn_%d'%(idx)] = score
            indices['attn_%d'%(idx)] = indice
        num_examples = 5
        i = 0
        while i < num_examples:
            logging.info('================================================================')
            question = ' '.join([vocab['id2word'][x] for x in data.dataset.questions[i]])
            logging.info(question)
            for t in range(3):
                logging.info('step {}'.format(t))
                logging.info(scores['attn_%d'%(t)][i])
                tmp = ' '.join([vocab['id2word'][x] for x in data.dataset.questions[i][indices['attn_%d'%(t)][i]]])
                logging.info(tmp)  
            i+=1
    return acc




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
    # with torch.no_grad():
        # model.word_embeddings.weight.set_(torch.Tensor(pretrained))
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
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4], gamma=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    validate(model, val_loader, device)
    # validate(model, test_loader, device)
    meters = MetricLogger(delimiter="  ")
    logging.info("Start training........")

    for epoch in range(args.num_epoch):
        model.train()
        for iteration, batch in enumerate(train_loader):
            iteration = iteration + 1

            question, topic_entity, answer = batch
            question = question.to(device)
            topic_entity = idx_to_one_hot(topic_entity, len(vocab['entity2id'])).to(device)
            # answer = answer.to(device)
            answer = idx_to_one_hot(answer, len(vocab['entity2id'])).to(device)
            answer[:, 0] = 0
            # check(question, topic_entity, answer, vocab)
            # return
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
                # scheduler.step()
                # validate(model, val_loader, device)
        
        acc = validate(model, val_loader, device)
        scheduler.step(acc['pred_e'])

        # acc = validate(model, test_loader, device)
        torch.save(model.state_dict(), os.path.join(args.save_dir, '%s-%s-%s-%d-%d'%(args.model_name, args.opt, str(args.lr), args.batch_size, epoch)))
        

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
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--eval_batch_size', default = 64, type = int)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    # model hyperparameters
    parser.add_argument('--dim_emb', default=300, type=int)
    parser.add_argument('--num_steps', default=3, type=int)
    parser.add_argument('--dim_word', default=300, type=int)
    parser.add_argument('--dim_hidden', default=300, type=int)
    parser.add_argument('--opt', default = 'adam', type = str)
    parser.add_argument('--log_name', default = 'log.txt', type = str)
    parser.add_argument('--model_name', default = 'model.pt', type = str)
    parser.add_argument('--rel', default = 1, type = int)
    parser.add_argument('--verbose', default = 1, type = int)
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