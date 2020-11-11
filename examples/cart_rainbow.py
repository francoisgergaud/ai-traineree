import pylab as plt
import numpy as np

from ai_traineree.agents.rainbow import RainbowAgent as Agent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask
from torch.utils.tensorboard import SummaryWriter


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


writer = SummaryWriter()
env_name = 'CartPole-v1'
task = GymTask(env_name)

agent = Agent(task.state_size, task.action_size)
env_runner = EnvRunner(task, agent, writer=writer)

scores = env_runner.run(reward_goal=100, max_episodes=500, force_new=True)
env_runner.interact_episode(render=True)


avg_length = 100
ma = running_mean(scores, avg_length)
# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(range(len(scores)), scores)
plt.plot(range(avg_length, avg_length+len(ma)), ma)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig(f'{env_name}.png', dpi=120)
plt.show()
