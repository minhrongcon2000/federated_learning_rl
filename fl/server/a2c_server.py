import gym
from typing import List
import random
import copy
import numpy as np
import tianshou as ts
import torch
import multiprocessing as mp
import os
from fl.client.a2c_client import A2CClient

from fl.server.base_server import BaseServer
from fl.client.dqn_client import DQNClient
from fl.utils.logger import create_logger


class A2CServer(BaseServer):
    def __init__(self,
                 policy: ts.policy.A2CPolicy,
                 chosen_prob: float) -> None:

        # global model info
        self.policy = policy
        
        # create clients
        self.client_pool: List[A2CClient] = []
        self._chosen_client: List[A2CClient] = []
        self.chosen_prob = chosen_prob
        
        self.logger = create_logger(__name__)
        self.num_round = 0
        
    def add_client(self, 
                   client_id: int,
                   client_name: str,
                   train_env: gym.Env, 
                   test_env: gym.Env,
                   local_batch_size: int, 
                   local_epochs: int, 
                   buffer: ts.data.ReplayBuffer,
                   step_per_collect: int,
                   test_num: int,
                   repeat=1 # how many sample to collect to estimate policy gradient theorem, more samples = better estimation + slower time
                   ):
        new_client = A2CClient(
            client_id=client_id,
            client_name=client_name,
            train_env=train_env,
            test_env=test_env,
            policy=copy.deepcopy(self.policy),
            batch_size=local_batch_size,
            epochs=local_epochs,
            buffer=buffer,
            step_per_collect=step_per_collect,
            test_num=test_num,
            repeat=repeat
        )
        self.client_pool.append(new_client)
        
    def broadcast(self, clients: List[A2CClient]) -> None:
        for client in clients:
            client.receive_weight(copy.deepcopy(self.policy.state_dict()))
            
    def update_client(self, client: DQNClient):
        client.update_weights()
        return client
        
    def aggregate(self) -> None:
        self.num_round += 1
        self._chosen_client: List[DQNClient] = random.sample(self.client_pool, 
                                                       int(len(self.client_pool) * self.chosen_prob))
        
        self.broadcast(self._chosen_client)
        
        global_policy_weights = copy.deepcopy(self.policy.state_dict())
        
        for key in global_policy_weights:
            global_policy_weights[key] = torch.zeros_like(global_policy_weights[key])
        
        
        with mp.Pool(os.cpu_count()) as p:
            results = p.map(self.update_client, self._chosen_client)
            
            for client in results:
                # weight average
                weight = copy.deepcopy(client.policy.state_dict())
                for key in weight:
                    global_policy_weights[key] += 1 / len(self._chosen_client) * weight[key]
                
                # update client for mp.Pool() will not modify object's state
                self.client_pool[client.client_id] = client
        
        self.policy.load_state_dict(global_policy_weights)
        self.make_result()
        self.logger.info("Round {} complete. Mean reward: {}, std reward: {}".format(self.num_round, 
                                                                                     self.result["server/mean_reward"], 
                                                                                     self.result["server/reward_std"]))
        
    def save_model(self, model_dir):
        torch.save(self.policy, model_dir)
        
    def load_model(self, model_dir):
        self.policy.load_state_dict(torch.load(model_dir))
        
    def make_result(self):
        self.result = {}
        rewards = []
        for client in self.client_pool:
            rewards.append(client.mean_reward)
            self.result["{}/mean_reward".format(client.client_name)] = client.mean_reward
            self.result["{}/reward_std".format(client.client_name)] = client.reward_std
            
        mean_reward = np.mean(rewards)
        reward_std = np.std(rewards, ddof=1)
        
        self.result["server/mean_reward"] = mean_reward
        self.result["server/reward_std"] = reward_std
