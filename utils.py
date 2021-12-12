import random
import os
import torch
import numpy as np
import pickle


def set_global_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_training_data():
    fcc_train = np.asarray(pickle.load(open("./data/bw/fcc_train100kb.pickle", "rb")))
    fcc_train = np.repeat(fcc_train, 10, axis=1)
    lte_train = np.asarray(pickle.load(open("./data/bw/LTE_train100kb.pickle", "rb")))
    train = np.concatenate((fcc_train, lte_train), axis=0)
    return train

def get_fcc_test_data():
    fcc_test = np.asarray(pickle.load(open("./data/bw/fcc_test100kb.pickle", "rb")))
    fcc_test = np.repeat(fcc_test, 10, axis=1)
    return fcc_test

def get_lte_test_data():
    lte_test = np.asarray(pickle.load(open("./data/bw/LTE_test100kb.pickle", "rb")))
    return lte_test

def get_linear_exp_decay(min_eps=0.01, max_eps=1, exploration_progress=1e6):
    return lambda eps: max(eps - (max_eps - min_eps) / exploration_progress, min_eps)
