from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pylab as plt

writer = SummaryWriter()

env_name = 'CartPole-v1'
task = GymTask(env_name)

agent = DQNAgent(task.state_size, task.action_size)
env_runner = EnvRunner(task, agent, writer=writer)

scores = env_runner.run(reward_goal=100, max_episodes=100, eps_end=0.002, eps_decay=0.999, log_every=5, gif_every_episodes=10)
env_runner.interact_episode(100, render=True)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
