from typing import Dict
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PensieveFeatureExtractor(nn.Module):
    def __init__(self, device, features_dim: int = 256, num_quality: int=7):
        super().__init__()
        self.features_dim = features_dim
        self.num_quality = num_quality
        self.device = device
        
        self.network_speed = nn.Sequential(
            nn.Conv1d(1, 128, 4, 1), 
            nn.ReLU(), 
            nn.Flatten()
        )
        
        self.next_chunk_size = nn.Sequential(
            nn.Conv1d(1, 128, 4, 1), 
            nn.ReLU(), 
            nn.Flatten()
        )
        
        self.buffer_size = nn.Sequential(
            nn.Linear(1, 128), 
            nn.ReLU()
        )
        
        self.percentage_remain_video_chunks = nn.Sequential(
            nn.Linear(1, 128), 
            nn.ReLU()
        )
        
        self.last_play_quality = nn.Sequential(
            nn.Linear(num_quality, 128), 
            nn.ReLU(), 
            nn.Flatten()
        )
        
        self.delay_net = nn.Sequential(
            nn.Conv1d(1, 128, 4, 1), 
            nn.ReLU(), 
            nn.Flatten()
        )
        
        self.last_layer = nn.Sequential(
            nn.Linear(1664, self.features_dim * 2),
            nn.Tanh(),
            nn.Linear(self.features_dim * 2, self.features_dim),
            nn.Tanh(),
        )
        
    def _preprocess_state(self, observations: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        observations_tensor: Dict[str, torch.Tensor] = {}
        for key in observations.keys():
            if key != "last_down_quality":
                observations_tensor[key] = torch.tensor(observations[key], dtype=torch.float)
            else:
                observations_tensor[key] = torch.tensor(observations[key], dtype=torch.long)
            
            if key in ["network_speed", "next_chunk_size", "delay"]:
                observations_tensor[key] = observations_tensor[key].unsqueeze(-2)
                
            elif key == "last_down_quality":
                observations_tensor[key] = F.one_hot(observations_tensor[key], num_classes=self.num_quality)
                observations_tensor[key] = torch.tensor(observations_tensor[key], dtype=torch.float)
        
        return observations_tensor

    def forward(self, observations: Dict[str, np.ndarray]) -> torch.Tensor:
        observations_tensor = self._preprocess_state(observations)
        network_speed = self.network_speed(
            observations_tensor["network_speed"].to(self.device)
        )
        
        next_chunk_size = self.next_chunk_size(
            observations_tensor["next_chunk_size"].to(self.device)
        )
        
        buffer_size = self.buffer_size(observations_tensor["buffer_size"].to(self.device))
        
        percentage_remain_video_chunks = self.percentage_remain_video_chunks(
            observations_tensor["percentage_remain_video_chunks"].to(self.device)
        )
        
        last_down_quality = self.last_play_quality(observations_tensor["last_down_quality"].to(self.device))
        
        delay = self.delay_net(observations_tensor["delay"].to(self.device))
        
        cat = torch.cat(
            (
                network_speed,
                next_chunk_size,
                buffer_size,
                percentage_remain_video_chunks,
                last_down_quality,
                delay,
            ),
            dim=1,
        )
        out = self.last_layer(cat)
        return out
