import gym
import torch
import tianshou as ts
from net_arch.mlp import CartPoleNetwork


model = torch.load(open("tmp/dqn_fl_44_2021_12_24", "rb"))
test_envs = ts.env.DummyVectorEnv([lambda : gym.make("CartPole-v0") for _ in range(1000)])

test_collector = ts.data.Collector(model, test_envs, exploration_noise=True)
results = test_collector.collect(n_episode=100)
print(results["rews"])
