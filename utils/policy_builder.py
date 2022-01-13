from typing import Any, Dict
from tianshou import policy
import torch.nn as nn
import tianshou as ts


def build_dqn_policy(model_config: Dict[str, Any]):
    net: nn.Module = model_config["model"]
    optimizer_class = model_config["optimizer_class"]
    optimizer = optimizer_class(net.parameters(), **model_config["optimizer_conf"])
    
    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optimizer,
        **model_config["policy_conf"]
    )
    return policy

def build_ppo_policy(model_config: Dict[str, Any]):
    actor_net = model_config["actor"]
    critic_net = model_config["critic"]
    
    optimizer_class = model_config["optimizer_class"]
    optimizer = optimizer_class(list(actor_net.parameters()) + list(critic_net.parameters()), 
                                **model_config["optimizer_conf"])
    
    policy = ts.policy.PPOPolicy(actor=model_config["actor"], 
                                 critic=model_config["critic"],
                                 optim=optimizer,
                                 **model_config["policy_conf"])
    return policy
