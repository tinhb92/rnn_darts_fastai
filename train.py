from fastai import *
from fastai.text import *
from fastai.text.models import EmbeddingDropout, RNNDropout

from copy import deepcopy
# LockedDropout ~ RNNDropout of fastai

STEPS = 8

def mask2d(B, D, keep_prob):
    m = torch.floor(torch.rand(B, D) + keep_prob) / keep_prob
    m.requires_grad=False
    return m.cuda()

class DartsCell(nn.Module):
    def __init__(self, ninp, nhid, dropouth, dropoutx, initrange, genotype):
        super(DartsCell, self).__init__()
        self.nhid = nhid
        self.dropouth = dropouth
        self.dropoutx = dropoutx
        self.genotype = genotype
        self.initrange = initrange
        
        # genotype is None when doing arch search
        steps = len(self.genotype.recurrent) if self.genotype is not None else STEPS
        self._W0 = nn.Parameter(torch.Tensor(ninp+nhid, 2*nhid).uniform_(-self.initrange, self.initrange))
        self._Ws = nn.ParameterList([
            nn.Parameter(torch.Tensor(nhid, 2*nhid).uniform_(-self.initrange, self.initrange)) for i in range(steps)
        ])
        
    def _get_activation(self, name):
        if name == 'tanh': f = torch.tanh
        elif name == 'relu': f = torch.relu
        elif name == 'sigmoid': f = torch.sigmoid
        elif name == 'identity': f = lambda x: x
        else: raise NotImplementedError
        return f
    
    def forward(self, x, hidden):
        b_sz, seq_len = x.size(0), x.size(1)
        if self.training:
            x_mask = mask2d(b_sz, x.size()[2], keep_prob = 1 - self.dropoutx)
            h_mask = mask2d(b_sz, hidden.size()[1], keep_prob = 1 - self.dropouth)
        else:
            x_mask = h_mask = None

        hiddens = []
        for t in range(seq_len):
            hidden = self.cell(x[:, t], hidden, x_mask, h_mask)
            hiddens.append(hidden)
        hiddens = torch.stack(hiddens, dim=1)
        return hiddens, hiddens[:, -1]

    def _compute_init_state(self, x, h_prev, x_mask, h_mask):
        if self.training:
            xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
        else:
            xh_prev = torch.cat([x, h_prev], dim=-1)
        c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
        c0 = c0.sigmoid()
        h0 = h0.tanh()
        s0 = h_prev + c0 * (h0-h_prev)
        return s0

    def cell(self, x, h_prev, x_mask, h_mask):
        s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

        states = [s0]
        for i, (name, pred) in enumerate(self.genotype.recurrent):
            s_prev = states[pred]
            if self.training:
                ch = (s_prev * h_mask).mm(self._Ws[i])
            else:
                ch = s_prev.mm(self._Ws[i])
            c, h = torch.split(ch, self.nhid, dim=-1)
            c = c.sigmoid()
            fn = self._get_activation(name)
            h = fn(h)
            s = s_prev + c * (h-s_prev)
            states += [s]
        output = torch.mean(torch.stack([states[i] for i in self.genotype.concat], -1), -1)
        return output

    
class DartsRnn(nn.Module):
    
    def __init__(self, emb_sz, vocab_sz,
                 ninp, nhid, 
                 dropout, dropouth, dropoutx, 
                 dropouti, dropoute,
                 bs_train, bs_val, bs_test=1,
                 initrange = 0.04,
                 cell_cls=DartsCell,
                 genotype=None):
        super().__init__()
        
        self.nhid = nhid
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.nhid = nhid
        self.encoder = nn.Embedding(vocab_sz, emb_sz)
        self.encoder_dp = EmbeddingDropout(self.encoder, self.dropoute) 
        
        self.decoder = nn.Linear(emb_sz, vocab_sz)
        self.decoder.weight = self.encoder.weight
        
        self.input_dp = RNNDropout(self.dropouti)
        self.hidden_dp = RNNDropout(self.dropout)
        
        self.bs_train = bs_train
        self.bs_val = bs_val
        self.bs_test = bs_test
        
        self.initrange = initrange
        self.init_weights()
        
        self.hid = self.init_hid(bs=self.bs_train)
        self.hid_search = self.init_hid(bs=self.bs_train)
        self.hid_val = self.init_hid(bs=self.bs_val)   
        self.hid_test = self.init_hid(bs=self.bs_test)
        
        self.test = False
        
        if cell_cls == DartsCell:
            assert genotype is not None
            self.rnn = cell_cls(ninp, nhid, dropouth, dropoutx, initrange, genotype)
        else:
            assert genotype is None
            self.rnn = cell_cls(ninp, nhid, dropouth, dropoutx, initrange)  
               
    def init_hid(self, bs):    
        return torch.zeros(bs, self.nhid).cuda()
    
    def init_weights(self):
        self.encoder.weight.data.uniform_(-self.initrange, self.initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-self.initrange, self.initrange) 
        
    def forward(self, x, details = True, hid_search = False):       
        x = self.encoder_dp(x)
        x = self.input_dp(x)
              
        if self.training:
            if not hid_search:
                x, self.hid = self.rnn(x, self.hid)
                self.hid = self.hid.detach()
                
            else:
                x, self.hid_search = self.rnn(x, self.hid_search)
                self.hid_search = self.hid_search.detach()
                
        elif not self.test:
                x, self.hid_val = self.rnn(x, self.hid_val)
                self.hid_val = self.hid_val.detach()
        else:
                x, self.hid_test = self.rnn(x, self.hid_test)
                self.hid_test = self.hid_test.detach()
            
        x_dropped = self.hidden_dp(x)
        x_out = self.decoder(x_dropped)        
         
        if self.training & details : return x, x_dropped, x_out
        else: return x_out    
              
class ASGD_Switch(LearnerCallback):
    _order = 10 # run after gradient clipping
    def __init__(self, learn:Learner, nonmono=5, asgd=False, eta=1e-3):
        super().__init__(learn)
        self.nonmono = nonmono + 1 # because recorder appended to val_losses
        self.asgd = asgd
        self.asgd_opt = torch.optim.ASGD(self.learn.model.parameters(), 
                                         lr=self.learn.opt.lr, 
                                         t0=0, lambd=0., 
                                         weight_decay=self.learn.opt.wd)
        if self.asgd:
            self.learn.opt.opt = self.asgd_opt
               
    def on_epoch_end(self, epoch, **kwargs):
        if not self.asgd and \
        len(self.learn.recorder.val_losses) > self.nonmono and \
        self.learn.recorder.val_losses[-1] > min(self.learn.recorder.val_losses[:-self.nonmono]):
            self.asgd = True
            self.learn.opt.opt = self.asgd_opt
            print('Switching to ASGD at epoch', epoch+1)
                 