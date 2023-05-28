"""Implements noisy SGD optimizer
"""
import torch
import torch.optim as optim
import torch.nn as nn


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class NoisySGD(optim.Optimizer):
    """Implements noisy SGD optimizer for the 2-layer toy CNN
    """

    def __init__(self, model: nn.Module, learning_rate, weight_decay_1, weight_decay_2, temperature):

        defaults = {
            'learning_rate': learning_rate,
            'weight_decay': 0.,
            'temperature': temperature
        }
        
        # a list of dictionaries which gives a simple way of breaking a model’s parameters into separate components   for optimization.
        groups = [{'params' : list(model.modules())[2].parameters(), # Conv1d layer
                   'learning_rate' : learning_rate,
                   'weight_decay' : weight_decay_1,
                   'temperature' : temperature},
                   {'params' : list(model.modules())[4].parameters(), # Linear layer
                    'learning_rate' : learning_rate, 
                    'weight_decay' : weight_decay_2, 
                    'temperature' : temperature}
                 ] 
        super(NoisySGD, self).__init__(groups, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        grad_L2_norm = 0.
        for group in self.param_groups:

            learning_rate = group['learning_rate']
            weight_decay = group['weight_decay']
            temperature = group['temperature']
             
            for parameter in group['params']:

                if parameter.grad is None:
                    continue

                d_p = torch.randn_like(parameter) * (2*learning_rate*temperature)**0.5
                # out = input + alpha*other 
                d_p.add_(parameter, alpha=-learning_rate*weight_decay)
                d_p.add_(parameter.grad, alpha=-learning_rate)
                grad_L2_norm += torch.linalg.matrix_norm(d_p).cpu().numpy()**2
                parameter.add_(d_p)

        print(grad_L2_norm)
        return grad_L2_norm
                
                
                
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------       
                
                
                
class LangevinMyrtle5(optim.Optimizer):
    """Implements Langevin (GD+noise) optimizer for the Myrtle5 CNN
    """

    def __init__(self, model: nn.Module, learning_rate, weight_decay_1, weight_decay_2, weight_decay_3, temperature):

        defaults = {
            'learning_rate': learning_rate,
            'weight_decay': 0.,
            'temperature': temperature
        }
        
        # a list of dictionaries which gives a simple way of breaking a model’s parameters into separate components for optimization.
        groups = [{'params' : list(model.modules())[1].parameters(), # input to hidden Conv2d
                   'learning_rate' : learning_rate,
                   'weight_decay' : weight_decay_1,
                   'temperature' : temperature},
                   {'params' : list(model.modules())[3].parameters(), # hidden to hidden Conv2d 
                    'learning_rate' : learning_rate, 
                    'weight_decay' : weight_decay_2, 
                    'temperature' : temperature},
                  {'params' : list(model.modules())[4].parameters(), # hidden to hidden Conv2d 
                    'learning_rate' : learning_rate, 
                    'weight_decay' : weight_decay_2, 
                    'temperature' : temperature},
                  {'params' : list(model.modules())[5].parameters(), # hidden to hidden Conv2d 
                    'learning_rate' : learning_rate, 
                    'weight_decay' : weight_decay_2, 
                    'temperature' : temperature},
                  {'params' : list(model.modules())[-1].parameters(), # Linear layer
                    'learning_rate' : learning_rate, 
                    'weight_decay' : weight_decay_3, 
                    'temperature' : temperature}
                 ] 
        super(LangevinMyrtle5, self).__init__(groups, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        grad_L2_norm = 0. 
        for group in self.param_groups:

            learning_rate = group['learning_rate']
            weight_decay = group['weight_decay']
            temperature = group['temperature']
            
            for parameter in group['params']:

                if parameter.grad is None:
                    continue

                d_p = torch.randn_like(parameter) * (2*learning_rate*temperature)**0.5
                # out = input + alpha*other 
                d_p.add_(parameter, alpha=-learning_rate*weight_decay)
                d_p.add_(parameter.grad, alpha=-learning_rate)

                grad_L2_norm += torch.linalg.norm(d_p).cpu().numpy()**2
                parameter.add_(d_p)
                
        return grad_L2_norm
                
