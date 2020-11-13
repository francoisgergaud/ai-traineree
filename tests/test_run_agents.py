"""
This 'Test suite' simply makes sure that all/most agents run.
Runs might not be perfect, might not be meaningful but at least
the code isn't broken.

**DO NOT TREAT THIS AS A UNIT TEST**
"""

from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.agents.ddpg import DDPGAgent
from ai_traineree.agents.td3 import TD3Agent
from ai_traineree.agents.ppo import PPOAgent
from ai_traineree.agents.sac import SACAgent
from ai_traineree.agents.rainbow import RainbowAgent
from ai_traineree.env_runner import EnvRunner
from ai_traineree.tasks import GymTask


DEVICE = "cpu"


# Discrete agents
def test_runs_dqn():
    # Assign
    task = GymTask('CartPole-v1')
    agent = DQNAgent(task.state_size, task.action_size, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)


def test_runs_rainbow():
    # Assign
    task = GymTask('CartPole-v1')
    agent = RainbowAgent(task.state_size, task.action_size, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)


def test_runs_ppo():
    # Assign
    task = GymTask('Pendulum-v0')
    agent = PPOAgent(task.state_size, task.action_size, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)


def test_runs_ddpg():
    # Assign
    task = GymTask('Pendulum-v0')
    agent = DDPGAgent(task.state_size, task.action_size, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)


def test_runs_td3():
    # Assign
    task = GymTask('Pendulum-v0')
    agent = TD3Agent(task.state_size, task.action_size, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)


def test_runs_sac():
    # Assign
    task = GymTask('BipedalWalker-v3')
    agent = SACAgent(task.state_size, task.action_size, device=DEVICE)
    env_runner = EnvRunner(task, agent, max_iterations=50)

    # Act
    env_runner.run(reward_goal=10, max_episodes=10, force_new=True)
