### Prevent GPU memory lock
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import pandas as pd
import matplotlib.pyplot as plt
import gym
from tqdm import tqdm

import multiprocessing
from itertools import product

env = gym.make("LunarLander-v2")
spec = gym.spec("LunarLander-v2")
num_episodes = 2500


def plot_comparison(data):
    for row in data:
        plt.plot(row['avg_scores'], label="lr={} gamma={}".format(row['lr'], row['gamma']))
    plt.xlabel("Episode")
    plt.ylabel("Average rewards over 100 episodes")
    plt.legend()
    plt.show()


def run(lr, gamma):
    
    import DQN as model
    
    graph = False
    earlystopping = True
    
    try:
        agent = model.Agent(lr=lr, gamma=gamma, epsilon=1.0, epsilon_decay=0.995, batch_size=64)
        scores, avg_scores = agent.train_model(env, num_episodes, graph, earlystopping=earlystopping)
        data.append({'lr':lr, 'gamma':gamma, 'scores':scores, 'avg_scores':avg_scores})
        df_data = pd.DataFrame(data)
        df_data.to_csv("hyper_search_DQN.csv")
        print("\t\tDone!")
    
    except Exception as e:
        print("Error occurred:")
        print(e)
        data.append({'lr':lr, 'gamma':gamma, 'scores':None, 'avg_scores':None})
             

if __name__ == '__main__':
    
    
    
    data = []
    
    ### HYPERPARAMETER GRIDS
    lr_grid = [0.001, 0.0001]
    gamma_grid = [0.99, 0.999]
    
    hyper_sets = []
    
    for lr in lr_grid:
        for gamma in gamma_grid:
            hyper_sets.append(tuple([lr, gamma]))

    with multiprocessing.Pool(processes=4) as pool:
        data = pool.starmap(run, hyper_sets)
       
    df_data = pd.DataFrame(data)
    df_data.to_csv("df_data.csv")
        
    print("Processes are successfully finished.")
    
    plot_comparison(data)
