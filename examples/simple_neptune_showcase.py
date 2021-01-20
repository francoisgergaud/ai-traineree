import datetime
import collections
import os

from ai_traineree.agents.dqn import DQNAgent
from ai_traineree.loggers import NeptuneLogger
from ai_traineree.tasks import GymTask
from typing import Iterable, Tuple

USERNAME = os.getenv("USERNAME")
PROJECT_NAME = os.getenv("PROJECT_NAME", "DQN-CartPole")

metrics = collections.defaultdict(list)
task = GymTask("CartPole-v1")
agent = DQNAgent(task.state_size, task.action_size, device="cpu")
data_logger = NeptuneLogger(f"{USERNAME}/{PROJECT_NAME}", params=agent.hparams, api_token=os.getenv("NEPTUNE_API_TOKEN"))

iteration = 0
MAX_EPISODES = 200


def interact_episode(agent, task, eps: float=0, log_interaction_freq: int=10) -> Tuple[float, int]:
    score = 0
    state = task.reset()
    global iteration
    _it = 0

    while True:
        iteration += 1
        _it += 1

        action = agent.act(state, eps)
        metrics['actions'].append((iteration, action))
        metrics['states'].append((iteration, state))

        next_state, reward, done, _ = task.step(action)
        metrics["rewards"].append((iteration, reward))
        metrics["dones"].append((iteration, done))

        score += float(reward)

        agent.step(state, action, reward, next_state, done)

        # Log every every {log_interaction_freq} iterations
        if iteration % log_interaction_freq == 0:
            log_data_interaction()
        state = next_state
        if done:
            break
    return score, _it


def log_data_interaction():
    """The only method that logs data in the logger"""
    global iteration
    agent.log_metrics(data_logger, iteration)

    while(metrics['states']):
        step, states = metrics['states'].pop(0)
        states = states if isinstance(states, Iterable) else [states]
        data_logger.log_values_dict("env/states", {str(i): a for i, a in enumerate(states)}, step)

    while(metrics['actions']):
        step, actions = metrics['actions'].pop(0)
        actions = actions if isinstance(actions, Iterable) else [actions]
        data_logger.log_values_dict("env/action", {str(i): a for i, a in enumerate(actions)}, step)

    while(metrics['rewards']):
        step, rewards = metrics['rewards'].pop(0)
        data_logger.log_value("env/reward", float(rewards), step)

    while(metrics['dones']):
        step, dones = metrics['dones'].pop(0)
        data_logger.log_value("env/done", int(dones), step)


last_n_scores = collections.deque(maxlen=100)
mean_score = -float('inf')
target_score = 100
epsilon = 0.99

for episode in range(MAX_EPISODES):
    epsilon = max(epsilon*0.99, 0.05)
    score, it = interact_episode(agent, task, epsilon)
    last_n_scores.append(score)
    mean_score = sum(last_n_scores) / len(last_n_scores)

    print(datetime.datetime.now().isoformat()[:-7], end="\t")
    print(f"{episode:3}.\ttotal iter: {iteration:7};\titer {it:4};\tscore: {score};\tmean score: {mean_score:.2f};\teps: {epsilon:.2f}")

    if mean_score > target_score and episode > last_n_scores.maxlen:
        break

print(f"Mean score {mean_score} (last {len(last_n_scores)}) after {episode+1} episodes")

data_logger.close()
