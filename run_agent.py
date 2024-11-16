from environment import Environment
from agent import DQLAgent
import matplotlib.pyplot as plt
import numpy as np

# Initialize the environment and agent
env = Environment('BTCUSDT', ['close'], 4)
episodes = 10
agent = DQLAgent(gamma=0.5, env=env)

# Train the agent
agent.learn(episodes=episodes)

# Plot training progress
x = range(len(agent.averages))
y = np.polyval(np.polyfit(x, agent.averages, deg=3), x)

plt.figure(figsize=(10, 6))
plt.plot(x, agent.averages, label='moving average')
plt.plot(x, y, 'r--', label='regression')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Progress of DQL Agent')
plt.legend()
plt.show() 

# Test the agent
agent.test(1)

# Adjust `env.data` to calculate strategy performance
agent.env.data['returns'] = env.data['close'].pct_change()
agent.env.data['strategy'] = env.data['returns']  # Example of a basic strategy
agent.env.data['strategy'].cumsum().plot(title="Cumulative Returns of Strategy")


