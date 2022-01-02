from typing import Any, Dict
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
