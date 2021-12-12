import torch
from net_arch.feature_extractor import PensieveFeatureExtractor
from net_arch.mlp import CartPoleNetwork, SB3MLPDQNNetwork
from net_arch.single_path_net import SinglePathPolicy
from utils import create_training_data, get_fcc_test_data, get_linear_exp_decay


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


TRAIN_ENV_CONF = {
    "env_name": "SinglePath-v0",
    "num_env": 1,
    "params": {
        "train": True,
        "bitrate_list": create_training_data()
    }
}

TEST_ENV_CONF = {
    "env_name": "SinglePath-v0",
    "num_env": 10,
    "params": {
        "train": False,
        "bitrate_list": get_fcc_test_data()
    }
}

MODEL_CONF = {
    "model": SinglePathPolicy(PensieveFeatureExtractor(device=DEVICE), 
                              SB3MLPDQNNetwork(256, 7, torch.nn.ReLU())),
    "optimizer_class": torch.optim.Adam,
    "optimizer_conf": {
        "lr": 0.0005
    },
    "gamma": 0.99,
    "n_step": 3,
    "target_update_freq": 1,
    "exploration_update": get_linear_exp_decay(),
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
    "model_name": "dqn_fl_44",
    "result_dir": "results",
    "result_fig": "result.png"
}

# # Uncomment this for Wandb usage 
# # and provide API key
# WANDB_CONFIG = {
#     "WANDB_API_KEY": "",
#     "project": "fl_rl",
#     "name": "fl_rl_dqn"
# }
WANDB_CONFIG = None
