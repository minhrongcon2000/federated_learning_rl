import gym
import torch
import tianshou as ts
import video_streaming_env.singlepath_env.envs.singlepath_gym
from utils.data import get_lte_test_data, get_fcc_test_data, get_fcc_train_data, get_lte_train_data


model = torch.load(open("tmp/singlepath_dqn_fl_44.pth", "rb"))
test_envs = ts.env.DummyVectorEnv([lambda : gym.make("SinglePath-v0", bitrate_list=get_lte_train_data(), train=False)])
test_collector = ts.data.Collector(model, test_envs, exploration_noise=True)
results = test_collector.collect(n_episode=100)
print(results["rews"])
print(results["rews"].mean())
