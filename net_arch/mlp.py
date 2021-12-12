from typing import List, Union
import torch
import torch.nn as nn
import numpy as np


class MLPNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 net_arch: List[int],
                 output_dim: int,
                 activation_fn: nn.Module,
                 device: str) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.model = self._create_network()
        self.device = device
        self.to(self.device)
        
    def _create_network(self):
        layers = [nn.Linear(self.input_dim, self.net_arch[0]), self.activation_fn]
        
        for i in range(len(self.net_arch) - 1):
            layers.append(nn.Linear(self.net_arch[i], self.net_arch[i + 1]))
            layers.append(self.activation_fn)
            
        layers.append(nn.Linear(self.net_arch[-1], self.output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, 
                obs: Union[np.ndarray, torch.Tensor], 
                state: torch.Tensor=None,
                info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
            
        batch = obs.shape[0]
        obs = obs.view(batch, -1)
        logits = self.model(obs.to(self.device))
        return logits, state


class SB3MLPDQNNetwork(MLPNetwork):
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 activation_fn: nn.Module,
                 device: str) -> None:
        super().__init__(input_dim, 
                         [64, 64], 
                         output_dim, 
                         activation_fn, 
                         device)
        

class CartPoleNetwork(MLPNetwork):
    def __init__(self, 
                 device,
                 input_dim: int=4, 
                 output_dim: int=2, 
                 activation_fn: nn.Module=torch.nn.ReLU()) -> None:
        super().__init__(input_dim, 
                         [128, 128], 
                         output_dim, 
                         activation_fn,
                         device)