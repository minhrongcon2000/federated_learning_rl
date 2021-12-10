import numpy as np
from video_streaming_env.singlepath_env.envs.singlepath_gym import SinglepathEnvGym
import gym
import pickle
import tianshou as ts


np.random.seed(1)
fcc_train = np.asarray(pickle.load(open("./data/bw/fcc_train100kb.pickle", "rb")))
fcc_train = np.repeat(fcc_train, 10, axis=1)
lte_train = np.asarray(pickle.load(open("./data/bw/LTE_train100kb.pickle", "rb")))
train = np.concatenate((fcc_train, lte_train), axis=0)
env = gym.make("SinglePath-v0", bitrate_list=train)
obs = env.reset()
print(obs)
# done = False
# reward_mean = 0
# while not done:
#     obs, reward, done, info = env.step(6)
#     reward_mean += reward
# print(reward_mean)