from typing import Any, Tuple
import torch.nn as nn
import torch
from net_arch.feature_extractor import PensieveFeatureExtractor

from net_arch.mlp import MLPNetwork


class SinglePathPolicy(nn.Module):
    def __init__(self, 
                 feature_extractor: PensieveFeatureExtractor, 
                 main_net: MLPNetwork):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor = feature_extractor
        self.main_net = main_net
        self.to(self.device)
        
    def forward(self, obs, state=None, info={}) -> Tuple[torch.Tensor, Any]:
        extracted_feature = self.feature_extractor(obs)
        q_values = self.main_net(extracted_feature)
        
        return q_values
