import torch

from net_arch.feature_extractor import PensieveFeatureExtractor
from net_arch.mlp import CartPoleNetwork, SB3MLPDQNNetwork
from net_arch.single_path_net import SinglePathPolicy

from utils.data import create_training_data, get_fcc_test_data
from utils.exp_strat import ExponentialDecayStrategy

import tianshou as ts


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


TRAIN_ENV_CONF = {
    "env_name": "CartPole-v0",
    "num_env": 1,
    "vec_env_class": ts.env.DummyVectorEnv,
    # "params": {
    #     "train": True,
    #     "bitrate_list": create_training_data()
    # }
    "params": {}
}

TEST_ENV_CONF = {
    "env_name": "SinglePath-v0",
    "num_env": 10,
    "vec_env_class": ts.env.DummyVectorEnv,
    "params": {
        "train": False,
        "bitrate_list": get_fcc_test_data()
    }
}

MODEL_CONF = {
    "model": CartPoleNetwork(device=DEVICE),
    "optimizer_class": torch.optim.Adam,
    "optimizer_conf": {
        "lr": 1e-3
    },
    "gamma": 0.99,
    "n_step": 3,
    "target_update_freq": 320,
    "exploration_update": ExponentialDecayStrategy(),
    "step_per_collect": 100
}

SERVER_CONF = {
    "num_clients": 100,
    "chosen_prob": 0.1
}

CLIENT_CONF = {
    "local_batch_size": 32,
    "local_epochs": 10,
    "max_buffer_size": 2000,
    "test_num": 10
}

TRAINING_CONF = {
    "num_rounds": 200,
    "seed": 44,
    "model_dir": "tmp",
    "model_name": "dqn_fl_44.pth",
    "result_dir": "results",
    "result_fig": "result.png"
}

# # Uncomment this for Wandb usage 
# # and provide API key
WANDB_CONFIG = {
    "WANDB_API_KEY": "183c1a6a36cbdf0405f5baacb72690845ecc8573",
    "project": "fl_rl",
    "name": "fl_rl_dqn"
}
# WANDB_CONFIG = None
