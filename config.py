import torch
from net_arch.feature_extractor import PensieveFeatureExtractor

from net_arch.mlp import CartPoleNetwork, SB3MLPDQNNetwork
from net_arch.single_path_net import SinglePathPolicy
from utils.client_pool_builder.duplicate_pool_builder import DuplicateClientPoolBuilder
from utils.client_pool_builder.mountain_car_pool_builder import MountainCarPoolBuilder
from utils.client_pool_builder.single_path_pool_builder import SinglePathClientPoolBuilder

from utils.exp_strat import ExponentialDecayStrategy
import numpy as np

import tianshou as ts


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_BUFFER_SIZE = 2000


TRAIN_ENV_CONF = {
    "env_name": "SinglePath-v0",
    "num_env": 1,
    "vec_env_class": ts.env.DummyVectorEnv,
    "params": {
        "train": True
    }
}

TEST_ENV_CONF = {
    "env_name": "SinglePath-v0",
    "num_env": 1,
    "vec_env_class": ts.env.DummyVectorEnv,
    "params": {
        "train": False
    }
}

MODEL_CONF = {
    "model": SinglePathPolicy(PensieveFeatureExtractor(device=DEVICE),
                              SB3MLPDQNNetwork(256, 7, torch.nn.ReLU(), device=DEVICE)),
    "optimizer_class": torch.optim.Adam,
    "optimizer_conf": {
        "lr": 1e-3
    },
    "policy_conf": {
        "discount_factor": 0.99,
        "estimation_step": 3, # n-step return DQN
        "target_update_freq": 320,
        "is_double": False,
        "reward_normalization": False
    }
}

SERVER_CONF = {
    "chosen_prob": 0.1,
    "update_eps": True
}

CLIENT_CONF = {
    "local_batch_size": 32,
    "local_epochs": 10,
    "test_num": 10,
    "exploration_update": ExponentialDecayStrategy(),
    "step_per_collect": 100,
    "buffer": ts.data.VectorReplayBuffer(MAX_BUFFER_SIZE, TRAIN_ENV_CONF["num_env"])
}

CLIENT_POOL_CONF = {
    "builder": SinglePathClientPoolBuilder,
    "builder_params": {
        "n_client": 100, # can be adjusted
        "client_config": CLIENT_CONF, # CANNOT BE ADJUSTED
        "train_env_config": TRAIN_ENV_CONF, # CANNOT BE ADJUSTED
        "test_env_config": TEST_ENV_CONF # CANNOT BE ADJUSTED
    }
}

TRAINING_CONF = {
    "num_rounds": 5,
    "seed": 2,
    "model_dir": "tmp",
    "model_name": "demo.pth",
    "result_dir": "results",
    "result_fig": "result.png",
    "result_data": "result.csv"
}

# # Uncomment this for Wandb usage 
# # and provide API key
# WANDB_CONFIG = {
#     "WANDB_API_KEY": "183c1a6a36cbdf0405f5baacb72690845ecc8573",
#     "project": "fl_rl",
#     "name": "fl_rl_dqn_non_iid_seed_2"
# }
WANDB_CONFIG = None
