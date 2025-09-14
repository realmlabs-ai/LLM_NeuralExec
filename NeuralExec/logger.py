import torch
import numpy as np

class CandidatePool:
    def __init__(self, hparams):
        self.B = []
        self.hparams = hparams
        self.patience = hparams['patience']
        self.disable = False
        
        self.number_of_reconf = 0
        
    def get_best(self):
        if self.disable:
            print("Disabled")
            ne_cpu, loss = self.B[-1]
            # Return a GPU copy while keeping original on CPU
            ne_gpu = ne_cpu.__class__(ne_cpu.prefix.clone(), ne_cpu.postfix.clone(), ne_cpu.sep)
            ne_gpu.to_gpu()
            return ne_gpu, loss
        
        i = np.argmin([loss for (ne, loss) in self.B])
        # is last
        if i != (len(self.B)-1):
            self.patience -= 1
            print(f'patience decreased to {self.patience}')
        else:
            self.patience = self.hparams['patience']
        
        if self.patience <= 0:
            print("Reconfiguration")
            self.patience = self.hparams['patience']
            self.reconfigure()
        
        # Return a GPU copy while keeping original on CPU
        ne_cpu, loss = self.B[i]
        ne_gpu = ne_cpu.__class__(ne_cpu.prefix.clone(), ne_cpu.postfix.clone(), ne_cpu.sep)
        ne_gpu.to_gpu()
        return ne_gpu, loss
    
    def insert_candidate(self, ne, loss):
        # Store CPU version to prevent GPU memory accumulation
        ne_cpu = ne.detach()  # Moves to CPU as per ex_triggers.py:54
        self.B.append((ne_cpu, loss))
        
        # Limit candidate pool size to prevent excessive memory usage
        max_pool_size = 20  # Keep only last 20 candidates
        if len(self.B) > max_pool_size:
            self.B = self.B[-max_pool_size:]
        
    def reconfigure(self):
                
        self.hparams['new_candidate_pool_size'] = self.hparams['new_candidate_pool_size'] + self.hparams['new_candidate_pool_size_increment']
        self.hparams['#prompts_to_sample_for_eval'] += self.hparams['#prompts_to_sample_for_eval_increment']
        
        if self.hparams['m'] > 1:
            self.hparams['m'] -= self.hparams['m_decrement']
            
        if self.hparams['topk_probability_new_candidate'] > self.hparams['min_topk_probability_new_candidate']:
            self.hparams['topk_probability_new_candidate'] -= self.hparams['topk_probability_new_candidate_decrement']
            
        if self.number_of_reconf >= self.hparams['max_number_reconf']:
            self.disable = True
            
        self.number_of_reconf += 1
        
                
########################################################################################

class Logger:
    def __init__(self, hparams):
        self.log_train = []
        self.log_eval = []
        self.confs = hparams
        
        self.candidate_pool = CandidatePool(hparams)
        
    def add_train_log(self, loss, ne, tokenizer):
        
        if type(loss) is torch.Tensor:
            loss = loss.detach().cpu().numpy()
            
        adv_seg = ne.decode(tokenizer)
        ne_cpu = ne.detach()  # This moves to CPU as per ex_triggers.py:54
        
        print(f'Neural Exec:-----> {adv_seg[0]} [PAYLOAD] {adv_seg[1]}')
        
        self.log_train.append({'loss':loss, 'NeuralExec':ne_cpu, 'NeuralExec_str':adv_seg})
        
    def add_eval_log(self, ne, loss, tokenizer):
        if type(loss) is torch.Tensor:
            loss = loss.detach().cpu().numpy()
            
        adv_seg = ne.decode(tokenizer)
        ne_cpu = ne.detach()  # This moves to CPU as per ex_triggers.py:54
                
        print(f'\tEval loss: {loss.mean()}')
        print(f'\tNeural Exec:-----> {adv_seg[0]} [PAYLOAD] {adv_seg[1]}')
        
        self.log_eval.append({'loss':loss, 'NeuralExec':ne_cpu, 'NeuralExec_str':adv_seg})
        
    def get_last_adv_tok(self, best=True):
        assert len(self.log_eval)
        if best:
            # Handle both scalar and array loss formats for backward compatibility
            loss_eval = []
            for l in self.log_eval:
                if hasattr(l['loss'], 'mean'):  # Array format
                    loss_eval.append(l['loss'].mean())
                else:  # Scalar format
                    loss_eval.append(l['loss'])
            loss_eval = np.array(loss_eval)
            best_idx = loss_eval.argmin()
            last_ne = self.log_eval[best_idx]['NeuralExec']
            loss = loss_eval[best_idx]
        else:
            last_ne = self.log_train[-1]['NeuralExec']
            loss_val = self.log_train[-1]['loss']
            loss = loss_val.mean() if hasattr(loss_val, 'mean') else loss_val
           
        # Create a fresh GPU copy instead of modifying the logged version
        last_ne_gpu = last_ne.__class__(last_ne.prefix.clone(), last_ne.postfix.clone(), last_ne.sep)
        last_ne_gpu.to_gpu()
        
        return last_ne_gpu, loss
    
    def get_ne_with_i(self, i):
        _i = i // self.confs[1]['eval_fq']
        loss, ne = self.log_eval[_i]
        ne.to_gpu()
        return ne, loss