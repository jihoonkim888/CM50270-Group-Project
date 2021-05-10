import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class ReplayBuffer:
    
    def __init__(self, size, input_shape):
        self.size = size
        self.counter = 0
        self.state_buffer = np.zeros((self.size, *input_shape), dtype=np.float32)
        self.action_buffer = np.zeros(self.size, dtype=np.int32)
        self.reward_buffer = np.zeros(self.size, dtype=np.float32)
        self.new_state_buffer = np.zeros((self.size, *input_shape), dtype=np.float32)
        self.terminal_buffer = np.zeros(self.size, dtype=np.bool_)

    def store_tuples(self, state, action, reward, new_state, done):
        idx = self.counter % self.size
        self.state_buffer[idx] = state
        self.action_buffer[idx] = action
        self.reward_buffer[idx] = reward
        self.new_state_buffer[idx] = new_state
        self.terminal_buffer[idx] = done
        self.counter += 1

    def sample_buffer(self, batch_size):
        max_buffer = min(self.counter, self.size)
        batch = np.random.choice(max_buffer, batch_size, replace=False)
        state_batch = self.state_buffer[batch]
        action_batch = self.action_buffer[batch]
        reward_batch = self.reward_buffer[batch]
        new_state_batch = self.new_state_buffer[batch]
        done_batch = self.terminal_buffer[batch]

        return state_batch, action_batch, reward_batch, new_state_batch, done_batch


class DuelingDQN(keras.Model):
    
    def __init__(self, num_actions, fc1, fc2):
        super(DuelingDQN, self).__init__()
        self.dense1 = Dense(fc1, activation='relu')
        self.dense2 = Dense(fc2, activation='relu')
        self.V = Dense(1, activation=None)
        self.A = Dense(num_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        avg_A = tf.math.reduce_mean(A, axis=1, keepdims=True)
        Q = (V + (A - avg_A))

        return Q, A
		
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
    def __init__(self, lr, gamma, num_actions, epsilon, batch_size, input_dim):
        self.action_space = [i for i in range(num_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epsilon_decay = 0.9995
        self.epsilon_final = 0.005
        self.update_rate = 120
        self.step_counter = 0
        self.buffer = ReplayBuffer(100000, input_dim)
        
        self.model = DuelingDQN(num_actions, 128, 128)
        self.target_model = DuelingDQN(num_actions, 128, 128)
        self.model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        self.target_model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    def store_tuple(self, state, action, reward, new_state, done):
        self.buffer.store_tuples(state, action, reward, new_state, done)

    def get_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            _, actions = self.model(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def train(self):
        if self.buffer.counter < self.batch_size:
            return
        if self.step_counter % self.update_rate == 0:
            self.target_model.set_weights(self.model.get_weights())

        state_batch, action_batch, reward_batch, new_state_batch, done_batch = \
            self.buffer.sample_buffer(self.batch_size)

        q_predicted, _ = self.model(state_batch)
        q_next, _ = self.target_model(new_state_batch)
        q_max_next = tf.math.reduce_max(q_next, axis=1, keepdims=True).numpy()
        q_target = np.copy(q_predicted)

        for idx in range(done_batch.shape[0]):
            target_q_val = reward_batch[idx]
            if not done_batch[idx]:
                target_q_val += self.gamma*q_max_next[idx]
            q_target[idx, action_batch[idx]] = target_q_val
        self.model.train_on_batch(state_batch, q_target)
        self.step_counter += 1

    
    def train_model(self, env, num_episodes, graph, earlystopping=True):

        self.scores, episodes, self.avg_scores, obj = [], [], [], []
        goal = 200
        f = 0
        avg_score = 0
        #txt = open("saved_networks.txt", "w")

        for i in range(num_episodes):
            # Early stopping...
            if earlystopping:
                if avg_score > goal:
                    print("The average rewards of the last 100 episodes > {}. Early stopping in Episode {}...".format(goal, i))
                    self.model.save(("saved_networks/dqn_model{0}".format(i)))
                    self.model.save_weights(("saved_networks/dqn_model{0}/net_weights{0}.h5".format(i)))
                    txt.write("Save {0} - Episode {1}/{2}, Score: {3} ({4}), AVG Score: {5}\n".format(i, i, num_episodes,\
                                                                                                      score, self.epsilon,\
                                                                                                      avg_score))
                    return

            done = False
            score = 0.0
            state = env.reset()
            while not done:
                action = self.get_action(state)
                new_state, reward, done, _ = env.step(action)
                score += reward
                self.store_tuple(state, action, reward, new_state, done)
                state = new_state
                self.train()
            self.scores.append(score)
            obj.append(goal)
            episodes.append(i)
            avg_score = np.mean(self.scores[-100:])
            self.avg_scores.append(avg_score)

            print_count = 50
            if (i % print_count == 0) and (i != 0):
#                 plot_graph(episodes, scores, avg_scores, obj)
                print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(i, num_episodes, score, round(self.epsilon, 2), round(avg_score, 2)))
                t2 = time.perf_counter()
                print("Finished {} episodes in {} seconds. Running...".format(print_count, t2-t1))
                t1 = time.perf_counter()
#            print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(i, num_episodes, score, self.epsilon,avg_score))
            self.epsilon *= self.epsilon_decay
            

        if graph:
            plot_graph(episodes, scores, avg_scores, obj)


        return scores, avg_scores

    
    def test(self, env, num_episodes, file_type, file, graph):
        if file_type == 'tf':
            self.model = tf.keras.models.load_model(file)
        elif file_type == 'h5':
            self.train_model(env, 5, False)
            self.model.load_weights(file)
        self.epsilon = 0.0
        self.scores, episodes, self.avg_scores, obj = [], [], [], []
        goal = 200
        score = 0.0
        for i in range(num_episodes):
            state = env.reset()
            done = False
            episode_score = 0.0
            while not done:
                env.render()
                action = self.get_action(state)
                new_state, reward, done, _ = env.step(action)
                episode_score += reward
                state = new_state
            score += episode_score
            self.scores.append(episode_score)
            obj.append(goal)
            episodes.append(i)
            avg_score = np.mean(self.scores[-100:])
            self.avg_scores.append(avg_score)
            print("Episode {0}/{1}, Score: {2} ({3}), AVG Score: {4}".format(i, num_episodes, episode_score, self.epsilon,
                                                                             avg_score))

        if graph:
            df = pd.DataFrame({'x': episodes, 'Score': self.scores, 'Average Score': self.avg_scores, 'Solved Requirement': obj})

            plt.plot('x', 'Score', data=df, marker='', color='blue', linewidth=2, label='Score')
            plt.plot('x', 'Average Score', data=df, marker='', color='orange', linewidth=2, linestyle='dashed',
                     label='AverageScore')
            plt.plot('x', 'Solved Requirement', data=df, marker='', color='red', linewidth=2, linestyle='dashed',
                     label='Solved Requirement')
            plt.legend()
            plt.savefig('LunarLander_Test.png')

        env.close()