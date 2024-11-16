from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential
import random
import matplotlib.pyplot as plt
import numpy as np
import os


class DQLAgent:
    def __init__(self, gamma=0.95, hu=24, opt=Adam, lr=0.001, finish=False, env=None):
        self.env = env
        self.finish = finish
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = gamma
        self.batch_size = 32
        self.max_treward = 0
        self.averages = []
        self.memory = deque(maxlen=200)
        self.osn = env.observation_space.shape[0]
        self.model = self._build_model(hu, opt, lr)

    def _build_model(self, hu, opt, lr):
        model = Sequential()
        model.add(Dense(hu, input_dim=self.osn, activation='relu'))
        model.add(Dense(hu, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer=opt(learning_rate=lr))
        return model

    def save(self):
        # Use .keras or .h5 extension to specify format
        os.makedirs('saved_model', exist_ok=True)
        self.model.save('saved_model/dql_model.keras')

    def load(self):
        self.model = tf.keras.models.load_model('saved_model/dql_model.keras')

    def act(self, state):
        if random.random() <= self.epsilon:
            return self.env.action_space.sample()
        action = self.model.predict(state)[0]
        return np.argmax(action)

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                reward += self.gamma * np.amax(self.model.predict(next_state)[0])
            target = self.model.predict(state)
            target[0, action] = reward
            self.model.fit(state, target, epochs=1, verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def learn(self, episodes): 
        trewards = []
        for e in range(1, episodes + 1):
            state = self.env.reset()
            state = np.reshape(state, [1, self.osn])
            for i in range(len(self.env.data)):
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.osn])
                self.memory.append([state, action, reward, next_state, done])
                state = next_state
                if done:
                    treward = i + 1
                    trewards.append(treward)
                    av = sum(trewards[-25:]) / 25
                    self.averages.append(av)
                    self.max_treward = max(self.max_treward, treward)
                    print(
                        f'episode: {e:4d}/{episodes} | treward: {treward:4d} | '
                        f'av: {av:6.1f} | max: {self.max_treward:4d}', end='\r'
                    )
                    break
            if av > 195 and self.finish:
                break
            if len(self.memory) > self.batch_size:
                self.replay()
        self.save()

    def test(self, episodes):
        self.load()
        self.epsilon = 0
        self.env.data['action'] = np.nan
        trewards = []
        for e in range(1, episodes + 1):
            state = self.env.reset()
            for i in range(len(self.env.data)):
                if self.env.bar >= len(self.env.data) - 1:
                    break
                state = np.reshape(state, [1, self.osn])
                action = np.argmax(self.model.predict(state)[0])
                next_state, reward, done, info = self.env.step(action)
                state = next_state
                if done:
                    treward = i + 1
                    trewards.append(treward)
                    print(f'episode: {e:4d}/{episodes} | treward: {treward:4d}', end='\r')
        return trewards
