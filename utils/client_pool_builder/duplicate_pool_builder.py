from utils.client_pool_builder.base_pool_builder import BaseClientPoolBuilder
import copy
from typing import Dict, Any, List, Type
import tianshou as ts

from utils.env_builder import EnvBuilder


class DuplicateClientPoolBuilder(BaseClientPoolBuilder):
    def build_pool(self, 
                   n_client: int,
                   client_config: Dict[str, Any], 
                   train_env_config: Dict[str, Any],
                   test_env_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        client_pool_conf = []
        
        for i in range(n_client):
            train_vec_env_class: Type[ts.env.BaseVectorEnv] = train_env_config["vec_env_class"]
            train_vec_env = train_vec_env_class([
                EnvBuilder(env_name=train_env_config["env_name"], **train_env_config["params"]) \
                    for _ in range(train_env_config["num_env"])
            ])
            
            test_vec_env_class: Type[ts.env.BaseVectorEnv] = test_env_config["vec_env_class"]
            test_vec_env = test_vec_env_class([
                EnvBuilder(test_env_config["env_name"], **test_env_config["params"]) \
                    for _ in range(test_env_config["num_env"])
            ])
            
            client_conf_copy = copy.deepcopy(client_config)
            client_conf_copy["train_env"] = train_vec_env
            client_conf_copy["test_env"] = test_vec_env
            client_conf_copy["client_id"] = i
            client_conf_copy["client_name"] = "client{}_{}".format(train_env_config["env_name"])
            client_pool_conf.append(client_conf_copy)
        
        return client_pool_conf
