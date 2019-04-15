from fastai import *
from fastai.text import *
from train import DartsCell, DartsRnn
from copy import deepcopy as dc
import copy
from darts_callbacks import Genotype

STEPS = 8
CONCAT = 8
edges_cnt = sum(i for i in range(1, STEPS+1))

PRIMITIVES = [
    'none',
    'tanh',
    'relu',
    'sigmoid',
    'identity'
]

class DartsCellSearch(DartsCell):
    
    def __init__(self, ninp, nhid, 
                 dropouth, dropoutx, initrange):
        super(DartsCellSearch, self).__init__(ninp, nhid, dropouth, 
                                              dropoutx, initrange, genotype=None)
        self.arch_p = torch.rand((edges_cnt, len(PRIMITIVES)), device="cuda").mul_(1e-3)
        self.arch_p.requires_grad = True
        self.bn = nn.BatchNorm1d(nhid, affine=False)        

    def cell(self, x, h_prev, x_mask, h_mask):
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
        s0 = self.bn(s0)
        probs = F.softmax(self.arch_p, dim=-1)

        offset = 0
        states = s0.unsqueeze(0)
        for i in range(STEPS):
            if self.training:
                masked_states = states * h_mask.unsqueeze(0)
            else:
                masked_states = states
            ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid) 
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()

            s = torch.zeros_like(s0)
            for k, name in enumerate(PRIMITIVES):
                if name == 'none': continue
                fn = self._get_activation(name)
                unweighted = states + c * (fn(h) - states)
                s += torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)
            s = self.bn(s)
            states = torch.cat([states, s.unsqueeze(0)], 0)
            offset += i+1
        
        cell_out = torch.mean(states[-CONCAT:], dim=0)
        return cell_out  
    
        
class DartsRnnSearch(DartsRnn):
    
    def __init__(self, emb_sz, vocab_sz, 
                 ninp, nhid, 
                 dropout,
                 dropouth, dropoutx, 
                 dropouti, dropoute,
                 bs_train, bs_val, bs_test=1):
        super(DartsRnnSearch, self).__init__(emb_sz, vocab_sz,
                                             ninp, nhid, 
                                             dropout,
                                             dropouth, dropoutx, 
                                             dropouti, dropoute,   
                                             bs_train, bs_val, bs_test,
                                             cell_cls=DartsCellSearch, 
                                             genotype=None)
        
    def genotype_parse(self):
        def _parse(probs):
            gene = []
            start = 0
            for i in range(STEPS):
                end = start + i + 1
                W = probs[start:end].copy()
                j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) \
                                                            if k != PRIMITIVES.index('none')))[0]
                k_best = None
                for k in range(len(W[j])):
                    if k != PRIMITIVES.index('none'):
                        if k_best is None or W[j][k] > W[j][k_best]:
                            k_best = k
                gene.append((PRIMITIVES[k_best], j))
                start = end
            return gene
        
        with torch.no_grad():
            gene = _parse(F.softmax(self.rnn.arch_p, dim=-1).cpu().numpy())
        genotype = Genotype(recurrent=gene, concat=range(STEPS+1)[-CONCAT:])
        return genotype  
    
    
class ArchParamUpdate(LearnerCallback):
    
    def __init__(self, learn:Learner, search_dat, 
                 arch_lr, arch_wdecay, wdecay): 
        super().__init__(learn)
        self.wdecay = wdecay
        self.search_dat = search_dat
        self.len_sd = len(search_dat)
        self.cnt = 0
        self.epsilon = 0.
        self.arch_opt = torch.optim.Adam([self.learn.model.rnn.arch_p], 
                                         lr=arch_lr, weight_decay=arch_wdecay)
        self.par = self.learn.model.parameters
        
    def clip_norm(self, inp, clip=0.25):
        total_norm = 0
        for g in inp:
            param_norm = g.norm(2)
            total_norm += param_norm ** 2
        total_norm = total_norm ** 0.5
        clip_coef =  clip/ (total_norm + 1e-6)
        if clip_coef < 1:
            for g in inp:
                g.mul_(clip_coef) 
        return clip_coef
   
    def on_batch_begin(self, last_input, last_target, **kwargs):
        if self.learn.model.training:
            original_hid = dc(self.learn.model.hid.detach())
            original_model_dict = dc(self.learn.model.state_dict())

            loss = self._loss(last_input, last_target)
            unrolled_grads = torch.autograd.grad(loss, self.par())
              
            clip_coef = self.clip_norm(unrolled_grads)
                    
            with torch.no_grad():
                for p, v in zip(self.par(), unrolled_grads):
                    v.add_(self.wdecay, p) 
                    p.sub_(self.learn.opt.lr, v)
            
            if self.cnt >= self.len_sd: self.cnt = 0
            x_search, y_search = self.search_dat[self.cnt]
            self.cnt += 1
            
            self.learn.model.hid = dc(original_hid)
            loss = self._loss(x_search, y_search, hid_search=True)
            loss.backward()
            dalpha = dc(self.learn.model.rnn.arch_p.grad) # first part of equation 6
            w_prime_grad = dc([v.grad for v in self.par()]) # save for w+, w-
            _ = self.clip_norm(w_prime_grad)

            self.learn.opt.zero_grad()
            self.arch_opt.zero_grad()
            self.epsilon = 1e-2 / torch.cat([x.view(-1) for x in w_prime_grad]).norm()
            self.learn.model.load_state_dict(original_model_dict)

            implicit_grads = self.impl(w_prime_grad, original_hid, last_input, last_target)
            self.learn.model.rnn.arch_p.grad = dalpha - self.learn.opt.lr * clip_coef * implicit_grads
        
            self.arch_opt.step()
            self.arch_opt.zero_grad()
            self.learn.opt.zero_grad()
            self.learn.model.hid = dc(original_hid) 
            
        return 
    
    def impl(self, w_prime_grad, original_hid, last_input, last_target):
        with torch.no_grad():
            for p, v in zip(self.par(), w_prime_grad):
                p.add_(self.epsilon, v) # w+

        self.learn.model.hid = dc(original_hid)
        loss = self._loss(last_input, last_target)
        w_plus_grad = torch.autograd.grad(loss, self.learn.model.rnn.arch_p)[0]

        with torch.no_grad():        
            for p, v in zip(self.par(), w_prime_grad):
                p.sub_(2*self.epsilon, v) # w-

        self.learn.model.hid = dc(original_hid) 
        loss = self._loss(last_input, last_target)
        w_minus_grad = torch.autograd.grad(loss, self.learn.model.rnn.arch_p)[0]

        with torch.no_grad():
            for p, v in zip(self.par(), w_prime_grad):
                p.add_(self.epsilon, v) # revert back to original

        implicit_grads = (w_plus_grad - w_minus_grad)/(2*self.epsilon)
        return implicit_grads
    
    def _loss(self, inp, target, hid_search=False):
        return self.learn.loss_func(self.learn.model(inp, 
                                                     details = False, 
                                                     hid_search=hid_search), target)
    
class PrintGenotype(LearnerCallback):
    def on_epoch_end(self, **kwargs):
        print(self.learn.model.genotype_parse())
        return