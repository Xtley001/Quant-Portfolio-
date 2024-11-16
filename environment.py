import numpy as np
import random 
from dataset import Dataset

class Actions:
    def __init__(self, n) -> None:
        self.n = n

    def sample(self):
        return random.randint(0, self.n - 1)
    
class ObservationSpace:
    def __init__(self, n) -> None:
        self.shape = (n,)

class Environment:
    def __init__(self, symbol, features, days=30) -> None:
        self.days = days
        self.symbol = symbol
        self.features = features
        self.observation_space = ObservationSpace(5)
        self.look_back = self.observation_space.shape[0]
        self.action_space = Actions(2)
        self.min_accuracy = 0.48
        self._get_data()
        self._create_features()

    def _get_data(self):
        data = Dataset().get_data(days=self.days, ticker=self.symbol, ts='1h')
        self.data = data[self.features]

    def _create_features(self):
        self.data['r'] = np.log(self.data['close'] / self.data['close'].shift(1))
        self.data.dropna(inplace=True)
        self.data = (self.data - self.data.mean()) / self.data.std()
        self.data['d'] = np.where(self.data['r'] > 0, 1, 0)

    def _get_state(self):
        return self.data[self.features]\
            .iloc[self.bar - self.look_back:self.bar]\
            .values

    def reset(self):
        self.total_reward = 0
        self.min_accuracy = 0
        self.bar = self.look_back
        return self._get_state()

    def step(self, action):
        correct = action == self.data['d'].iloc[self.bar]
        reward = 1 if correct else 0
        self.total_reward += reward
        self.bar += 1
        self.accuracy = self.total_reward / (self.bar - self.look_back)
        if self.bar >= len(self.data):
            done = True
        elif reward == 1:
            done = False
        elif (self.accuracy < self.min_accuracy and
              self.bar > self.look_back + 10):
            done = True
        else:
            done = False

        state = self._get_state()
        info = {}
        return state, reward, done, info

