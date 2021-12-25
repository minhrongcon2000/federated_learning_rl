import gym
from typing import Any, Callable, List
import random
import copy
import tianshou as ts
import torch
import numpy as np
import multiprocessing as mp
import os

from fl.server.base_server import BaseServer
from fl.client.dqn_client import DQNClient
from fl.utils.logger import create_logger


class DQNServer(BaseServer):
    def __init__(self,
                 policy: ts.policy.DQNPolicy,
                 chosen_prob: float) -> None:

        # global model info
        self.policy = policy
        
        # create clients
        self.client_pool: List[DQNClient] = []
        self.chosen_prob = chosen_prob
        self.mean_reward = 0
        self.reward_std = 0
        
        self.logger = create_logger(__name__)
        self.num_round = 0
        
    def add_client(self, 
                   client_id: Any, 
                   env: gym.Env, 
                   local_batch_size: int, 
                   local_epochs: int, 
                   buffer: ts.data.ReplayBuffer,
                   exploration_update: Callable[[float], float],
                   step_per_collect: int,
                   test_num: int):
        new_client = DQNClient(
            client_id=client_id,
            env=env,
            policy=copy.deepcopy(self.policy),
            batch_size=local_batch_size,
            epochs=local_epochs,
            buffer=buffer,
            exploration_update=exploration_update,
            step_per_collect=step_per_collect,
            test_num=test_num
        )
        self.client_pool.append(new_client)
        
    def broadcast(self) -> None:
        for client in self.client_pool:
            client.policy.load_state_dict(copy.deepcopy(self.policy.state_dict()))
            
    def update_client(self, client: DQNClient):
        mean_reward, weight = client.update_weights()
        self.logger.debug("Client {} finished update. Reward: {}".format(client.client_id, mean_reward))
        return mean_reward, weight
        
    def aggregate(self) -> None:
        self.num_round += 1
        chosen_client: List[DQNClient] = random.sample(self.client_pool, 
                                                       int(len(self.client_pool) * self.chosen_prob))
        
        global_policy_weights = copy.deepcopy(self.policy.state_dict())
        
        rewards = []
        
        for key in global_policy_weights:
            global_policy_weights[key] = torch.zeros_like(global_policy_weights[key])
            
        with mp.Pool(os.cpu_count()) as p:
            results = p.map(self.update_client, chosen_client)
            for mean_reward, weight in results:
                rewards.append(mean_reward)
                
                for key in weight:
                    global_policy_weights[key] += 1 / len(chosen_client) * weight[key]
            p.close()
            
        # for client in chosen_client:
        #     client.update_weights()
            
        #     rewards.append(client.mean_reward)
            
        #     local_policy_weights = copy.deepcopy(client.policy.state_dict())
            
        #     for key in global_policy_weights:
        #         global_policy_weights[key] += 1 / len(chosen_client) * local_policy_weights[key]
            
        #     self.logger.debug("Client {} finished update. Reward: {}".format(client.client_id, client.mean_reward))
        
        self.policy.load_state_dict(global_policy_weights)
        self.mean_reward = np.mean(rewards)
        self.reward_std = np.std(rewards, ddof=1)
        
        self.logger.info("Round {} complete. Mean reward: {}, std reward: {}".format(self.num_round, self.mean_reward, self.reward_std))
        
    def save_model(self, model_dir):
        torch.save(self.policy, model_dir)
        
    def load_model(self, model_dir):
        self.policy.load_state_dict(torch.load(model_dir))
