# environment dependencies
import numpy as np
import video_streaming_env.singlepath_env.envs.singlepath_gym
import gym

# Deep RL dependencies
import tianshou as ts
from torch import nn

# FL dependencies
from fl.server.dqn_server import DQNServer

# utilities
from typing import Any, Dict
import wandb
import os
import matplotlib.pyplot as plt
from utils import set_global_seed


def plot_training_curve(reward_means, reward_stdds, n_clients, figure_dir):
    reward_means = np.array(reward_means)
    reward_stdds = np.array(reward_stdds)
    plt.plot(np.arange(len(reward_means)), reward_stdds)
    plt.fill_between(np.arange(len(reward_means)), 
                    reward_means + 1.96 * reward_stdds / np.sqrt(n_clients),
                    reward_means - 1.96 * reward_stdds / np.sqrt(n_clients),
                    alpha=0.2)
    plt.savefig(figure_dir, dpi=300)
    plt.show()


def train_dqn_fl(env_config: Dict[Any, str],
                 model_config: Dict[Any, str],
                 server_config: Dict[Any, str],
                 client_config: Dict[Any, str],
                 train_config: Dict[Any, str],
                 wandb_config: Dict[Any, str]=None):
    
    set_global_seed(train_config["seed"])
    
    if wandb_config is not None:
        os.environ["WANDB_API_KEY"] = wandb_config["WANDB_API_KEY"]
        
    if not os.path.exists(train_config["model_dir"]):
        os.mkdir(train_config["model_dir"])
    model_save_dir = os.path.join(train_config["model_dir"], train_config["model_name"])
    
    if not os.path.exists(train_config["result_dir"]):
        os.mkdir(train_config["result_dir"])
    figure_save_dir = os.path.join(train_config["result_dir"], train_config["result_fig"])
    
    net: nn.Module = model_config["model"]
    optimizer_class = model_config["optimizer_class"]
    optimizer = optimizer_class(net.parameters(), **model_config["optimizer_conf"])
    
    policy = ts.policy.DQNPolicy(net, 
                                 optimizer, 
                                 discount_factor=model_config["gamma"], 
                                 estimation_step=model_config["n_step"], 
                                 target_update_freq=model_config["target_update_freq"])
    
    server = DQNServer(policy=policy, chosen_prob=server_config["chosen_prob"])
    
    for i in range(server_config["num_clients"]):
        if "params" not in env_config:
            env=ts.env.DummyVectorEnv(
                [
                    lambda: gym.make(env_config["env_name"]) \
                        for _ in range(env_config["num_env"])
                ]
            )
        else:
            env=ts.env.DummyVectorEnv([
                lambda: gym.make(env_config["env_name"], **env_config["params"]) \
                    for _ in range(env_config["num_env"])
            ])
        
        server.add_client(
            client_id=i, 
            env=env, 
            local_batch_size=client_config["local_batch_size"], 
            local_epochs=client_config["local_epochs"], 
            buffer=ts.data.VectorReplayBuffer(client_config["max_buffer_size"], env_config["num_env"]), 
            exploration_update=model_config["exploration_update"], 
            step_per_collect=model_config["step_per_collect"], 
            test_num=client_config["test_num"]
        )
        
    if wandb_config is not None:
        wandb.init(project=wandb_config["project"], 
                   name=wandb_config["name"])
        
    max_reward = None
    
    reward_means = []
    reward_stds = []
    
    for i in range(train_config["num_rounds"]):
        server.aggregate()
        server.broadcast()
        
        reward_means.append(server.mean_reward)
        reward_stds.append(server.reward_std)
        
        if max_reward is None or max_reward < server.mean_reward:
            server.save_model(model_save_dir)
            max_reward = server.mean_reward
        
        if wandb_config is not None:
            wandb.log({"mean_reward": server.mean_reward, "reward_std": server.reward_std})
            
    plot_training_curve(reward_means, 
                        reward_stds, 
                        server_config["num_clients"], 
                        figure_save_dir)
