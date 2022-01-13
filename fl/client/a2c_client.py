import gym
from typing import Any
import tianshou as ts

from fl.client.base_client import BaseClient


class A2CClient(BaseClient):
    def __init__(self,
                 client_id: Any,
                 client_name: str,
                 train_env: gym.Env,
                 test_env: gym.Env,
                 policy: ts.policy.A2CPolicy,
                 batch_size: int,
                 epochs: int,
                 buffer: ts.data.ReplayBuffer,
                 step_per_collect: int,
                 test_num: int,
                 repeat: int) -> None:
        super().__init__()
        self.client_id = client_id
        self.client_name = client_name
        self.buffer = buffer
        self.step_per_collect = step_per_collect
        self.test_num = test_num
        self.policy = policy
        self.batch_size = batch_size
        self.epochs = epochs
        self.repeat = repeat
        
        self.train_collector = ts.data.Collector(self.policy, 
                                                 train_env,
                                                 self.buffer,
                                                 exploration_noise=True)
        
        self.test_collector = ts.data.Collector(self.policy, 
                                                test_env,
                                                self.buffer,
                                                exploration_noise=True)
        
        self.mean_reward = 0
        self.reward_std = 0
        
    def test(self):
        test_result = self.test_collector.collect(n_episode=self.test_num)
        return test_result["rews"].mean(), test_result["rews"].std(ddof=1)
        
    def update_weights(self):
        for _ in range(self.epochs):
            self.train_collector.collect(n_step=self.step_per_collect)
            self.policy.update(
                sample_size=self.batch_size, 
                buffer=self.train_collector.buffer,
                batch_size=self.batch_size,
                repeat=self.repeat
            )
        mean_reward, reward_std = self.test()
        self.mean_reward = mean_reward
        self.reward_std = reward_std
    
    def receive_weight(self, state_dict):
        self.policy.load_state_dict(state_dict)
