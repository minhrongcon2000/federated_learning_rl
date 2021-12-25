from copy import deepcopy
import gym
from typing import Any, Callable
import tianshou as ts

from fl.client.base_client import BaseClient


class DQNClient(BaseClient):
    def __init__(self,
                 client_id: Any,
                 env: gym.Env,
                 policy: ts.policy.DQNPolicy,
                 batch_size: int,
                 epochs: int,
                 buffer: ts.data.ReplayBuffer,
                 exploration_update: Callable[[float], float],
                 step_per_collect: int,
                 test_num: int) -> None:
        
        self.client_id = client_id
        self.policy = policy
        
        self.buffer = buffer
        self.train_env = env
        self.test_env = env
        
        self.train_collector = ts.data.Collector(self.policy, 
                                                 self.train_env, 
                                                 self.buffer, 
                                                 exploration_noise=True)
        
        self.test_collector = ts.data.Collector(self.policy,
                                                self.test_env,
                                                exploration_noise=False)
        
        self.batch_size = batch_size
        self.epochs = epochs
        self.step_per_collect = step_per_collect
        self.test_num = test_num
        self.exp_update = exploration_update
        self.eps = 1
        self.mean_reward = None
        
    def update_weights(self):
        for _ in range(self.epochs):
            self.policy.set_eps(self.eps)
            self.train_collector.collect(n_step=self.step_per_collect)
            self.policy.update(self.batch_size, self.train_collector.buffer)
            
            test_result = self.test_collector.collect(n_episode=self.test_num)
            
            self.mean_reward = test_result["rew"].mean()
            
            self.eps = self.exp_update(self.eps)
            
        return test_result["rew"].mean(), deepcopy(self.policy.state_dict())
            