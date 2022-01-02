from typing import Any, Dict, List, Type
from utils.client_pool_builder.base_pool_builder import BaseClientPoolBuilder
from utils.env_builder import EnvBuilder
import numpy as np
import tianshou as ts
import copy


class MountainCarPoolBuilder(BaseClientPoolBuilder):
    def build_pool(self, 
                   n_client: int, 
                   client_config: Dict[str, Any], 
                   train_env_config: Dict[str, Any],
                   test_env_config: Dict[str, Any],
                   min_goal=-0.07,
                   max_goal=0.07) -> List[Dict[str, Any]]:
        client_pool_conf = []
        gs = np.linspace(min_goal, max_goal, n_client)
        
        for i in range(n_client):
            train_vec_env_class: Type[ts.env.BaseVectorEnv] = train_env_config["vec_env_class"]
            train_vec_env = train_vec_env_class([
                EnvBuilder(env_name=train_env_config["env_name"], goal_velocity=float(gs[i])) \
                    for _ in range(train_env_config["num_env"])
            ])
            
            test_vec_env_class: Type[ts.env.BaseVectorEnv] = test_env_config["vec_env_class"]
            test_vec_env = test_vec_env_class([
                EnvBuilder(env_name=test_env_config["env_name"], goal_velocity=float(gs[i])) \
                    for _ in range(test_env_config["num_env"])
            ])
            
            client_config_copy = copy.deepcopy(client_config)
            client_config_copy["train_env"] = train_vec_env
            client_config_copy["test_env"] = test_vec_env
            client_config_copy["client_id"] = i
            client_config_copy["client_name"] = "client{}_{}".format(i, gs[i])
            client_pool_conf.append(client_config_copy)
            
        return client_pool_conf
        