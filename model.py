import torch
import torch.nn as nn

from BiGRU import GRU, BiGRU
from Knowledge_graph import KnowledgeGraph

class TransferNet(nn.Module):
    def __init__(self, args, dim_word, dim_hidden, vocab):
        super().__init__()
        self.args = args
        self.vocab = vocab
        self.kg = KnowledgeGraph(args, vocab)
        num_words = len(vocab['word2id'])
        num_entities = len(vocab['entity2id'])
        num_relations = len(vocab['relation2id'])
        self.num_steps = args.num_steps

        self.question_encoder = BiGRU(dim_word, dim_hidden, num_layers=1, dropout=0.2)
        
        self.word_embeddings = nn.Embedding(num_words, dim_word)
        self.word_dropout = nn.Dropout(0.3)
        self.step_encoders = []
        for i in range(self.num_steps):
            m = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.Tanh()
            )
            self.step_encoders.append(m)
            self.add_module('step_encoders_{}'.format(i), m)
        self.rel_classifier = nn.Linear(dim_hidden, num_relations)
        self.cq_linear = nn.Linear(2 * dim_hidden, dim_hidden)
        self.ca_linear = nn.Linear(dim_hidden, 1)
        self.ent_classifier = nn.Sequential(
                nn.Linear(dim_hidden, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_entities),
            )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def follow(self, e, r):
        x = torch.sparse.mm(self.kg.Msubj, e.t()) * torch.sparse.mm(self.kg.Mrel, r.t())
        return torch.sparse.mm(self.kg.Mobj.t(), x).t() # [bsz, Esize]
        
    def transfer(self, q_word_h, q_embeddings, e_s):
        """
        Args:
            q_word_h [bsz, max_q, dim_h]: hidden state for each word in the question
            q_embeddings [bsz, dim_h]: question representation
            e_s [bsz, Esize]: one hot vector for topic entity
        """
        device = q_word_h.device
        bsz = q_word_h.size(0)
        dim_h = q_word_h.size(-1)
        last_e = e_s
        last_c = torch.zeros((bsz, dim_h)).to(device) # [bsz, dim_h]
        word_attns = []
        rel_probs = []
        ent_probs = [e_s]
        for t in range(self.num_steps):
            cq_t = self.step_encoders[t](q_embeddings) # [bsz, dim_h]
            # cq_t = torch.cat((last_c, q_t), dim = 1) # [bsz, 2 * dim_h]
            # cq_t = self.cq_linear(cq_t) # [bsz, dim_h]
            q_logits = torch.sum(cq_t.unsqueeze(1) * q_word_h, dim=2) # [bsz, max_q]
            q_dist = torch.softmax(q_logits, 1).unsqueeze(1) # [bsz, 1, max_q]
            word_attns.append(q_dist.squeeze(1))
            last_c = (q_dist @ q_word_h).squeeze(1) # [bsz, dim_h]
            rel_dist = torch.softmax(self.rel_classifier(last_c), 1) # [bsz, num_relations]
            rel_probs.append(rel_dist)

            last_e = self.follow(last_e, rel_dist)

            # reshape >1 scores to 1 in a differentiable way
            m = last_e.gt(1).float()
            z = (m * last_e + (1-m)).detach()
            last_e = last_e / z

            # reshape cycle entities to 0, because A-r->B-r_inv->A is not allowed in MetaQA
            if t > 0:
                prev_rel = torch.argmax(rel_probs[-2], dim=1)
                curr_rel = torch.argmax(rel_probs[-1], dim=1)
                prev_prev_ent_prob = ent_probs[-2]
                # in our vocabulary, indices of inverse relations are adjacent. e.g., director:0, director_inv:1
                m = torch.zeros((bsz,1)).to(device)
                m[(torch.abs(prev_rel-curr_rel)==1) & (torch.remainder(torch.min(prev_rel,curr_rel),2)==0)] = 1
                ent_m = m.float() * prev_prev_ent_prob.gt(0.9).float()
                last_e = (1-ent_m) * last_e
            ent_probs.append(last_e.detach())
        return {
            'last_e': last_e,
            'word_attns': word_attns,
            'rel_probs': rel_probs,
            'ent_probs': ent_probs
        }

    def forward(self, questions, e_s, answers = None):
        question_lens = questions.size(1) - questions.eq(0).long().sum(dim=1) # 0 means <PAD>
        q_word_emb = self.word_dropout(self.word_embeddings(questions)) # [bsz, max_q, dim_hidden]
        q_word_h, q_embeddings, q_hn = self.question_encoder(q_word_emb, question_lens) # [bsz, max_q, dim_h], [bsz, dim_h], [num_layers, bsz, dim_h]

        outputs = self.transfer(q_word_h, q_embeddings, e_s) # [bsz, num_entities]
        
        e_score = outputs['last_e']
        normed_e_score = e_score / (e_score.sum(1, True) + 1e-6) # [bsz, num_entities]
        pred_e = normed_e_score @ self.kg.entity_embeddings.weight # [bsz, dim_hidden]
        pred_e = self.ent_classifier(pred_e) # [bsz, num_entities]

        if answers is None:
            return {
                'e_score': e_score,
                'pred_e': pred_e,
                'word_attns': outputs['word_attns'],
                'rel_probs': outputs['rel_probs'],
                'ent_probs': outputs['ent_probs']
            }
        # CrossEntropy with most possible entity
        # pos_pred_e = pred_e * answers
        # _, idx = pos_pred_e.max(1) # [bsz]
        # loss = nn.CrossEntropyLoss()(pred_e, idx)

        # distance loss
        weight = answers * 9 + 1
        loss = torch.mean(weight * torch.pow(e_score - answers, 2))

        return loss
