# environment dependencies
import numpy as np
from utils.client_pool_builder.base_pool_builder import BaseClientPoolBuilder
import video_streaming_env.singlepath_env.envs.singlepath_gym

# Deep RL dependencies
import tianshou as ts
from torch import nn

# FL dependencies
from fl.server.dqn_server import DQNServer

# utilities
from typing import Any, Dict
import wandb
import os
from utils.random import set_global_seed
from utils.plot import plot_training_curve
from utils.policy_builder import build_dqn_policy


def train_dqn_fl(model_config: Dict[str, Any],
                 server_config: Dict[str, Any],
                 client_pool_conf: Dict[str, Any],
                 train_config: Dict[str, Any],
                 wandb_config: Dict[str, Any]=None):
    
    set_global_seed(train_config["seed"])
    if wandb_config is not None:
        os.environ["WANDB_API_KEY"] = wandb_config["WANDB_API_KEY"]
        
    # logging stuffs
    if not os.path.exists(train_config["model_dir"]):
        os.mkdir(train_config["model_dir"])
    model_save_dir = os.path.join(train_config["model_dir"], train_config["model_name"])
    
    if not os.path.exists(train_config["result_dir"]):
        os.mkdir(train_config["result_dir"])
    figure_save_dir = os.path.join(train_config["result_dir"], train_config["result_fig"])
    ##########################
    
    policy = build_dqn_policy(model_config=model_config)
    
    server = DQNServer(policy=policy, **server_config)
    
    client_pool_builder: BaseClientPoolBuilder = client_pool_conf["builder"]()
    client_conf_list = client_pool_builder.build_pool(**client_pool_conf["builder_params"])
    
    for client_conf in client_conf_list:
        server.add_client(**client_conf)
        
    if wandb_config is not None:
        wandb.init(project=wandb_config["project"], 
                   name=wandb_config["name"])
        
    max_reward = None
    
    reward_means = []
    reward_stds = []
    
    for _ in range(train_config["num_rounds"]):
        server.aggregate()
        
        results = server.result
        
        if max_reward is None or max_reward < results["server/mean_reward"]:
            server.save_model(model_save_dir)
            max_reward = results["server/mean_reward"]
            
        reward_means.append(results["server/mean_reward"])
        reward_stds.append(results["server/reward_std"])
        
        if wandb_config is not None:
            wandb.log(server.result)
            
    plot_training_curve(reward_means, 
                        reward_stds, 
                        client_pool_conf["builder_params"]["n_client"], 
                        figure_save_dir)
