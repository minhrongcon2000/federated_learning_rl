from copy import deepcopy
import gym
from typing import Any, Callable
import tianshou as ts

from fl.client.base_client import BaseClient

class DQNClient(BaseClient):
    def __init__(self,
                 client_id: Any,
                 client_name: str,
                 train_env: gym.Env,
                 test_env: gym.Env,
                 policy: ts.policy.DQNPolicy,
                 batch_size: int,
                 epochs: int,
                 buffer: ts.data.ReplayBuffer,
                 exploration_update: Callable[[float], float],
                 step_per_collect: int,
                 test_num: int) -> None:
        
        self.client_id = client_id
        self.client_name = client_name
        self.policy = policy
        
        self.buffer = buffer
        self.train_env = train_env
        self.test_env = test_env
        
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
        self.mean_reward = 0
        self.reward_std = 0
        
    def test(self):
        test_result = self.test_collector.collect(n_episode=self.test_num)
        return test_result["rews"].mean(), test_result["rews"].std(ddof=1)
        
    def update_weights(self):
        for _ in range(self.epochs):
            self.policy.set_eps(self.eps)
            self.train_collector.collect(n_step=self.step_per_collect)
            self.policy.update(self.batch_size, self.train_collector.buffer)
            mean_reward, reward_std = self.test()
            self.mean_reward = mean_reward
            self.reward_std = reward_std
            self.eps = self.exp_update(self.eps)
    
    def receive_weight(self, state_dict):
        self.policy.load_state_dict(state_dict)