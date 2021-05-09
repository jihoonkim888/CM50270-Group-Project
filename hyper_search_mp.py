### For Hex server
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

### Prevent GPU memory lock
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

import matplotlib.pyplot as plt
import gym
from tqdm import tqdm

import multiprocessing
from itertools import product

env = gym.make("LunarLander-v2")
spec = gym.spec("LunarLander-v2")
num_episodes = 5000

import DQN as model


def run(lr_grid, gamma_grid):
    data = []
    graph = False
    earlystopping = False

    for lr in tqdm(lr_grid):
        print("lr =", lr)
        for gamma in tqdm(gamma_grid, leave=False):
            print("\tgamma =", gamma)
            try:
                agent = model.Agent(lr=lr, gamma=gamma, epsilon=1.0, epsilon_decay=0.995, batch_size=64)
                scores, avg_scores = agent.train_model(env, num_episodes, graph, earlystopping=earlystopping)
                data.append({'lr': lr, 'gamma': gamma, 'scores': scores, 'avg_scores': avg_scores})
                print("\t\tDone!")
            except Exception as e:
                print("Error ocurred:")
                print(e)
                data.append({'lr': lr, 'gamma': gamma, 'scores': None, 'avg_scores': None})

    return data


def plot_comparison(data):
    for row in data:
        plt.plot(row['avg_scores'], label="lr={} gamma={}".format(row['lr'], row['gamma']))
    plt.xlabel("Episode")
    plt.ylabel("Average rewards over 100 episodes")
    plt.legend()
    plt.show()

def merge_names(a, b):
    return '{} & {}'.format(a, b)

if __name__ == '__main__':

    ### HYPERPARAMETER GRIDS
    lr_grid = [0.01, 0.001, 0.0001]
    gamma_grid = [0.99, 0.999]

    with multiprocessing.Pool(processes=6) as pool:
        data = pool.starmap(run, product(lr_grid, gamma_grid))
    plot_comparison(data)