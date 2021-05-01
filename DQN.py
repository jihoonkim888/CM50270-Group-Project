import time
import random
from collections import deque
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class ReplayBuffer:
    
    def __init__(self, size, input_shape):
        self.size = size
        self.counter = 0
        self.state_buffer = np.zeros((self.size, input_shape), dtype=float)
        self.action_buffer = np.zeros(self.size, dtype=int)
        self.reward_buffer = np.zeros(self.size, dtype=float)
        self.next_state_buffer = np.zeros((self.size, input_shape), dtype=float)
        self.terminal_buffer = np.zeros(self.size, dtype=bool)

    
    def store_tuples(self, state, action, reward, next_state, done):
        i = self.counter % self.size
        self.state_buffer[i] = state
        self.action_buffer[i] = action
        self.reward_buffer[i] = reward
        self.next_state_buffer[i] = next_state
        self.terminal_buffer[i] = done
        self.counter += 1

    
    def sample_buffer(self, batch_size):
        max_buffer = min(self.counter, self.size)
        batch = np.random.choice(max_buffer, batch_size, replace=False)
        state_batch = self.state_buffer[batch]
        action_batch = self.action_buffer[batch]
        reward_batch = self.reward_buffer[batch]
        next_state_batch = self.next_state_buffer[batch]
        done_batch = self.terminal_buffer[batch]

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


### DQN model
def model(lr, num_actions, input_dims):

    model = Sequential()
    model.add(Dense(512, input_dim=input_dims, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    model.compile(loss='mse',optimizer=Adam(lr=lr))

    print(model.summary())

    return model


def plot_graph(episodes, scores, avg_scores, obj):
    df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': obj})

    plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
    plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
             label='AverageScore')
    plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
             label='Solved Requirement')
    plt.legend()
    plt.show()

class Agent:
    
    def __init__(self, lr, gamma, epsilon, epsilon_decay, batch_size):
        input_dims = 8
        num_actions = 4
        self.action_space = [i for i in range(num_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.update_rate = 120
        self.step_counter = 0

        self.buffer = ReplayBuffer(500000, input_dims)
#         self.buffer = deque(maxlen=500000)
        self.model = model(lr, num_actions, input_dims)
        self.target_model = model(lr, num_actions, input_dims)

    
    def store_tuple(self, state, action, reward, next_state, done):
        self.buffer.store_tuples(state, action, reward, next_state, done)

    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            # print(state.shape)
            actions = self.model.predict(np.array(state))
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    
    def train(self):
        if self.buffer.counter < self.batch_size:
            return

        if self.step_counter % self.update_rate == 0:
            self.target_model.set_weights(self.model.get_weights())

#         random_sample = random.sample(self.buffer, self.batch_size)

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.buffer.sample_buffer(self.batch_size)

        q_predicted = self.model(state_batch)
        q_next = self.target_model(next_state_batch)
        q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()
        q_target = np.copy(q_predicted)

        for i in range(done_batch.shape[0]):
            target_q_val = reward_batch[i]
            if not done_batch[i]:
                target_q_val += self.gamma * q_max_next[i]
            q_target[i, action_batch[i]] = target_q_val
        self.model.train_on_batch(state_batch, q_target)
        self.step_counter += 1


#         target_q_val = reward_batch + self.gamma * \
#         (np.amax(self.model.predict_on_batch(next_state_batch), axis=1)) * (1 - done_batch)
#         q_target = self.model.predict_on_batch(state_batch)
#         indices = np.array([i for i in range(self.batch_size)])
#         # print("target_q_val.shape, q_target.shape, indices.shape, action_batch.shape:")
#         # print(target_q_val.shape, q_target.shape, indices.shape, action_batch.shape)
#         q_target[[indices], [action_batch]] = target_q_val

    def train_model(self, env, num_episodes, graph):
        
        scores, episodes, avg_scores, obj = [], [], [], []
        goal = 150
        avg_score = 0
        txt = open("saved_networks.txt", "w")
        t1 = time.perf_counter()

        for i in range(num_episodes):

            # Early stopping...
            if avg_score > goal:
                print("The average rewards of the last 100 episodes > {}. Early stopping in Episode {}...".format(goal, i))
                self.model.save(("saved_networks/dqn_model{0}".format(i)))
                self.model.save_weights(("saved_networks/dqn_model{0}/net_weights{0}.h5".format(i)))
                txt.write("Save {0} - Episode {1}/{2}, Score: {3} ({4}), AVG Score: {5}\n".format(i, i, num_episodes,
                                                                                                  score, self.epsilon,
                                                                                                  avg_score))
                return

            done = False
            score = 0.0
            state = env.reset()
            while not done:
                # print('state:', state)
                state = state.reshape(1,-1)
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                score += reward
                self.store_tuple(state, action, reward, next_state, done)
#                 self.buffer.append((state, action, reward, next_state, done))
                state = next_state
                self.train()
            scores.append(score)
            obj.append(goal)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)
            
            # avg_score_10 = np.mean(scores[-10:])
            
            print_count = 50
            if (i % print_count == 0) and (i != 0):
#                 plot_graph(episodes, scores, avg_scores, obj)
                print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(i, num_episodes, score, self.epsilon, avg_score))
                t2 = time.perf_counter()
                print("Finished {} episodes in {} seconds".format(print_count, t2-t1))
                t1 = time.perf_counter()
                
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            

            
            if (i==0) or (i==num_episodes-1):
                self.model.save(("saved_networks/dqn_model{0}".format(i)))
                self.model.save_weights(("saved_networks/dqn_model{0}/net_weights{0}.h5".format(i)))
                txt.write("Save {0} - Episode {1}/{2}, Score: {3} ({4}), AVG Score: {5}\n".format(i, i, num_episodes,
                                                                                                  score, self.epsilon,
                                                                                                  avg_score))
#                 f += 1
                print("Network saved")

        txt.close()
        
        if graph:
            
            plot_graph(episodes, scores, avg_scores, obj)
#             df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': obj})

#             plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
#             plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
#                      label='AverageScore')
#             plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
#                      label='Solved Requirement')
#             plt.legend()
#             plt.savefig('LunarLander_Train.png')
            
        return scores

    def test(self, env, num_episodes, file_type, file, graph):
        if file_type == 'tf':
            self.model = tf.keras.models.load_model(file)
        elif file_type == 'h5':
            self.train_model(env, 5, False)
            self.model.load_weights(file)
        self.epsilon = 0.0
        scores, episodes, avg_scores, obj = [], [], [], []
        goal = 200
        score = 0.0
        for i in range(num_episodes):
            state = env.reset()
            done = False
            episode_score = 0.0
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_score += reward
                state = next_state
            score += episode_score
            scores.append(episode_score)
            print(f"{i}th round - {episode_score}")
            obj.append(goal)
            episodes.append(i)
            avg_score = np.mean(scores[-100:])
            avg_scores.append(avg_score)

        if graph:
            df = pd.DataFrame({'x': episodes, 'Score': scores, 'Average Score': avg_scores, 'Solved Requirement': obj})

            plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
            plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
                     label='AverageScore')
            plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
                     label='Solved Requirement')
            plt.legend()
            plt.savefig('LunarLander_Test.png')

        env.close()




