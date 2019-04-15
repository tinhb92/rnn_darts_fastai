from fastai import *
from fastai.text import *
from collections import namedtuple

Genotype = namedtuple('Genotype', 'recurrent concat')

class Regu(LearnerCallback):
    def __init__(self, learn:Learner, alpha=0, beta=1e-3):
        super().__init__(learn)
        self.alpha = alpha # L2 regularization on RNN activation
        self.beta = beta # slowness regularization applied on RNN activiation
        
    def on_loss_begin(self, last_output, **kwargs):
        if self.learn.model.training:
            self.raw, self.dropped = last_output[0], last_output[1]
            return {'last_output': last_output[2]}
    
    def on_backward_begin(self, last_loss, **kwargs):   
        if self.learn.model.training:
            last_loss += self.alpha * self.dropped.pow(2).mean() # ar
            last_loss += self.beta * (self.raw[:, 1:] - self.raw[:, :-1]).pow(2).mean() # tar
            return {'last_loss': last_loss}
    
class HidInit(LearnerCallback):
    def on_epoch_begin(self, **kwargs):
        self.learn.model.hid = self.learn.model.init_hid(bs=self.learn.model.bs_train)
        self.learn.model.hid_search = self.learn.model.init_hid(bs=self.learn.model.bs_train)
        self.learn.model.hid_val = self.learn.model.init_hid(bs=self.learn.model.bs_val)      

class GcCol(LearnerCallback):
    def on_backward_end(self, **kwargs):   
        gc.collect()    
        
class SaveModel(LearnerCallback):
    _order = 50 # after everything
    def __init__(self, learn:Learner, gap, name:str='bestmodel'):
        super().__init__(learn)
        self.name = name
        self.gap = gap
    
    def on_epoch_end(self, epoch:int, **kwargs:Any)->None:
        if epoch % self.gap == 0: 
            try:
                # save train_search
                torch.save({'model': self.learn.model.state_dict(), 
                            'opt':self.learn.opt.state_dict(),
                            'arch_p': self.learn.model.rnn.arch_p}, 
                           self.path/self.model_dir/f'{self.name}_{epoch}.pth')
            except AttributeError:
                # save train
                torch.save({'model': self.learn.model.state_dict(), 
                            'opt':self.learn.opt.state_dict()}, 
                           self.path/self.model_dir/f'{self.name}_{epoch}.pth')

            print('Saved model at end of epoch', epoch)

class ResumeModel(LearnerCallback):
    _order = 15 # after asgd_switch
    def __init__(self, learn:Learner, name:str):
        super().__init__(learn)
        self.name = name
        
    def on_train_begin(self, **kwargs:Any):
        checkpoint = torch.load(self.path/self.model_dir/f'{self.name}.pth', 
                                map_location=lambda storage, loc: storage)
        self.learn.model.load_state_dict(checkpoint['model'])
        self.learn.opt.load_state_dict(checkpoint['opt'])
        
        try:
            self.learn.model.rnn.arch_p = checkpoint['arch_p']
        except:
            pass
        
        print(f'Resume from file {self.name}.pth')