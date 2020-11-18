import json
import logging
import numpy as np
import time
import os
import sys
from ai_traineree.types import ActionType, AgentType, DoneType, MultiAgentType, RewardType, StateType, TaskType
from ai_traineree.types import MultiAgentTaskType


from collections import deque
from functools import reduce
from pathlib import Path
from typing import List, Optional, Tuple

FRAMES_PER_SEC = 25
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def save_gif(path, images: List[np.ndarray]) -> None:
    print(f"Saving as a gif to {path}")
    from PIL import Image
    imgs = [Image.fromarray(img[::2, ::2]) for img in images]  # Reduce /4 size; pick w/2 h/2 pix

    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], optimize=True, quality=85)


class EnvRunner:
    """
    EnvRunner, shorter for Environment Runner, is meant to be used as module that runs your experiments.
    It's expected that the environments are wrapped in a Task which has typical step and act methods.
    The agent can be any agent which *makes sense* as there aren't any checks like whether the output is discrete.

    Typicall run is
    >>> env_runner = EnvRunner(task, agent)
    >>> env_runner.run()
    """

    def __init__(self, task: TaskType, agent: AgentType, max_iterations: int=int(1e5), **kwargs):
        """
        Expects the environment to come as the TaskType and the agent as the AgentType.
        Additional args:

        writer: Tensorboard writer.
        """
        self.logger = logging.getLogger("EnvRunner")
        self.task = task
        self.agent = agent
        self.max_iterations = max_iterations
        self.model_path = f"{task.name}_{agent.name}"
        self.state_dir = 'run_states'

        self.episode = 0
        self.iteration = 0
        self.all_scores = []
        self.all_iterations = []
        self.window_len = kwargs.get('window_len', 100)
        self.__images = []

        self.writer = kwargs.get("writer")
        self.logger.info("writer: %s", str(self.writer))

    def __str__(self) -> str:
        return f"EnvRunner<{self.task.name}, {self.agent.name}>"

    def reset(self):
        """Resets the EnvRunner. The task env and the agent are preserved."""
        self.episode = 0
        self.all_scores = []
        self.all_iterations = []
        self.scores_window = deque(maxlen=self.window_len)

    def interact_episode(
        self, eps: float=0, max_iterations: Optional[int]=None, render: bool=False, render_gif: bool=False, log_loss_every: Optional[int]=None
    ) -> Tuple[RewardType, int]:
        score = 0
        state = self.task.reset()
        iterations = 0
        max_iterations = max_iterations if max_iterations is not None else self.max_iterations

        # Only gifs require keeping (S, A, R) list
        if render_gif:
            self.__images = []

        while(iterations < max_iterations):
            iterations += 1
            self.iteration += 1
            if render:
                self.task.render("human")
                time.sleep(1./FRAMES_PER_SEC)

            action = self.agent.act(state, eps)
            next_state, reward, done, _ = self.task.step(action)
            score += float(reward)
            if render_gif:
                # OpenAI gym still renders the image to the screen even though it shouldn't. Eh.
                img = self.task.render(mode='rgb_array')
                self.__images.append(img)
            self.agent.step(state, action, reward, next_state, done)
            if log_loss_every is not None and (iterations % log_loss_every) == 0:
                self.log_loss()
            state = next_state
            if done:
                break
        return score, iterations

    def run(
        self,
        reward_goal: float=100.0, max_episodes: int=2000,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        log_every=10, gif_every_episodes: Optional[int]=None,
        checkpoint_every=200, force_new=False,
    ):
        """
        Evaluates the agent in the environment.
        The evaluation will stop when the agent reaches the `reward_goal` in the averaged last `self.window_len`, or
        when the number of episodes reaches the `max_episodes`.

        To help debugging one can set the `gif_every_episodes` to a positive integer which relates to how often a gif
        of the episode evaluation is written to the disk.

        Every `checkpoint_every` (default: 200) iterations the Runner will store current state of the runner and the agent.
        These states can be used to resume previous run. By default the runner checks whether there is ongoing run for
        the combination of the environment and the agent.
        """
        self.epsilon = eps_start
        self.reset()
        if not force_new:
            self.load_state(self.model_path)

        while (self.episode < max_episodes):
            self.episode += 1
            render_gif = gif_every_episodes is not None and (self.episode % gif_every_episodes) == 0
            score, iterations = self.interact_episode(self.epsilon, render_gif=render_gif)

            self.scores_window.append(score)
            self.all_iterations.append(iterations)
            self.all_scores.append(score)

            mean_score: float = sum(self.scores_window) / len(self.scores_window)

            self.epsilon = max(eps_end, eps_decay * self.epsilon)

            if self.episode % log_every == 0:
                if 'critic_loss' in self.agent.__dict__ and self.agent.critic_loss is not None:
                    loss = {'actor_loss': self.agent.actor_loss, 'critic_loss': self.agent.critic_loss}
                else:
                    loss = {'loss': self.agent.loss}
                self.info(episode=self.episode, iterations=iterations, score=score, mean_score=mean_score, epsilon=self.epsilon, **loss)

            if render_gif and len(self.__images):
                gif_path = "gifs/{}_e{}.gif".format(self.model_path, str(self.episode).zfill(len(str(max_episodes))))
                save_gif(gif_path, self.__images)
                self.__images = []

            if mean_score >= reward_goal and len(self.scores_window) == self.window_len:
                print(f'Environment solved after {self.episode} episodes!\tAverage Score: {mean_score:.2f}')
                self.save_state(self.model_path)
                self.agent.save_state(f'{self.model_path}_agent.net')
                break

            if self.episode % checkpoint_every == 0:
                self.save_state(self.model_path)

        return self.all_scores

    def info(self, **kwargs):
        """
        Writes out current state into provided loggers.
        Currently supports stdout logger (default on) and Tensorboard SummaryWriter initiated through EnvRun(writer=...)).
        """
        if self.writer is not None:
            self.log_writer(**kwargs)
        if self.logger is not None:
            self.log_logger(**kwargs)

    def log_logger(self, **kwargs):
        line_chunks = ["Episode {episode};", "Iter: {iterations};"]
        line_chunks += ["Current Score: {score:.2f};"]
        line_chunks += ["Average Score: {mean_score:.2f};"]
        if 'critic_loss' in self.agent.__dict__ and self.agent.critic_loss is not None:
            line_chunks += ["Actor loss: {actor_loss:10.4f};"]
            line_chunks += ["Critic loss: {critic_loss:10.4f};"]
        else:
            line_chunks += ["Loss: {loss:10.4f};"]
        line_chunks += ["Epsilon: {epsilon:5.3f};"]
        line = "\t".join(line_chunks)
        self.logger.info(line.format(**kwargs))

    def log_writer(self, **kwargs):
        self.writer.add_scalar("score/score", kwargs['score'], self.episode)
        self.writer.add_scalar("score/avg_score", kwargs['mean_score'], self.episode)
        self.writer.add_scalar("epsilon", kwargs['epsilon'], self.iteration)
        self.log_loss(**kwargs)

    def log_loss(self, **kwargs):
        if hasattr(self.agent, 'log_writer'):
            self.agent.log_writer(self.writer, self.iteration)
        elif 'critic_loss' in self.agent.__dict__ and self.agent.critic_loss is not None:
            self.writer.add_scalar("Actor loss", kwargs['actor_loss'], self.iteration)
            self.writer.add_scalar("Critic loss", kwargs['critic_loss'], self.iteration)
        else:
            self.writer.add_scalar("loss", kwargs['loss'], self.iteration)

    def save_state(self, state_name: str):
        """Saves the current state of the runner and the agent.

        Files are stored with appended episode number.
        Agents are saved with their internal saving mechanism.
        """
        state = {
            'tot_iterations': sum(self.all_iterations),
            'episode': self.episode,
            'epsilon': self.epsilon,
            'score': self.all_scores[-1],
            'average_score': sum(self.scores_window) / len(self.scores_window),
        }
        if hasattr(self.agent, "critic_loss") and self.agent.critic_loss is not None:
            state['actor_loss'] = self.agent.actor_loss,
            state['critic_loss'] = self.agent.critic_loss,
        elif hasattr(self.agent, "loss"):
            state['loss'] = self.agent.loss
        else:
            pass

        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        self.agent.save_state(f'{self.state_dir}/{state_name}_e{self.episode}.agent')
        with open(f'{self.state_dir}/{state_name}_e{self.episode}.json', 'w') as f:
            json.dump(state, f)

    def load_state(self, state_prefix: str):
        """
        Loads state with the highest episode value for given agent and environment.
        """
        try:
            state_files = list(filter(lambda f: f.startswith(state_prefix) and f.endswith('json'), os.listdir(self.state_dir)))
            e = max([int(f[f.index('_e')+2:f.index('.')]) for f in state_files])
        except Exception:
            self.logger.warning("Couldn't load state. Forcing restart.")
            return

        state_name = [n for n in state_files if n.endswith(f"_e{e}.json")][0][:-5]
        self.logger.info("Loading saved state under: %s/%s.json", self.state_dir, state_name)
        with open(f'{self.state_dir}/{state_name}.json', 'r') as f:
            state = json.load(f)
        self.episode = state.get('episode')
        self.epsilon = state.get('epsilon')

        self.all_scores.append(state.get('score'))
        self.all_iterations = []

        avg_score = state.get('average_score')
        for _ in range(min(self.window_len, self.episode)):
            self.scores_window.append(avg_score)

        self.logger.info("Loading saved agent state: %s/%s.agent", self.state_dir, state_name)
        self.agent.load_state(f'{self.state_dir}/{state_name}.agent')

        if hasattr(self.agent, "actor_loss") and self.agent.actor_loss is not None:
            self.agent.actor_loss = state.get('actor_loss', 0)
            self.agent.critic_loss = state.get('critic_loss', 0)
        else:
            self.agent.loss = state.get('loss', 0)


class MultiAgentEnvRunner:
    """
    MultiAgentEnvRunner has the same purpose as the EnvRunner but specifically for environments that support multiple agents.
    It's expected that the environments are wrapped in a Task which has typical step and act methods.
    The agent can be any agent which *makes sense* as there aren't any checks like whether the output is discrete.

    Typicall run is
    >>> ma_env_runner = MultiAgentEnvRunner(task, agent)
    >>> ma_env_runner.run()

    Example
    -------
    Check [examples/multi_agents](/examples/multi_agents) directory.

    """

    def __init__(self, task: MultiAgentTaskType, multi_agent: MultiAgentType, mode: str ='coop', max_iterations: int=int(1e5), **kwargs):
        """
        Expects the environment to come as the TaskType and the agent as the AgentType.
        Additional args:

        writer: Tensorboard writer.
        """
        self.logger = logging.getLogger("MAEnvRunner")
        self.task = task
        self.multi_agent: MultiAgentType = multi_agent
        self.max_iterations = max_iterations
        self.model_path = f"{task.name}_{multi_agent.name}"
        self.state_dir = 'run_states'

        self.mode = mode
        self.episode = 0
        self.iteration = 0
        self.all_scores: List[List[RewardType]] = []
        self.all_iterations: List[int] = []
        self.window_len = kwargs.get('window_len', 100)
        self.__images = []

        self.writer = kwargs.get("writer")
        self.logger.info("writer: %s", str(self.writer))

    def __str__(self) -> str:
        return f"EnvRunner<{self.task.name}, {self.multi_agent.name}>"

    def reset(self):
        """Resets the EnvRunner. The task env and the agent are preserved."""
        self.episode = 0
        self.all_scores: List[List[RewardType]] = []
        self.all_iterations = []
        self.scores_window = deque(maxlen=self.window_len)

    def interact_episode(
        self, eps: float=0, max_iterations: Optional[int]=None,
        render: bool=False, render_gif: bool=False, log_loss_every: Optional[int]=None
    ) -> Tuple[List[RewardType], int]:
        score: List[RewardType] = [0.]*self.multi_agent.agents_number
        states: List[StateType] = self.task.reset()

        iterations = 0
        max_iterations = max_iterations if max_iterations is not None else self.max_iterations

        # Only gifs require keeping (S, A, R) list
        if render_gif:
            self.__images = []

        while(iterations < max_iterations):
            iterations += 1
            self.iteration += 1
            if render:
                self.task.render("human")
                time.sleep(1./FRAMES_PER_SEC)

            all_actions: List[ActionType] = self.multi_agent.act(states, eps)
            next_states: List[StateType] = []
            rewards: List[RewardType] = []
            dones: List[DoneType] = []

            for agent_id in range(self.multi_agent.agents_number):
                next_state, reward, done, _ = self.task.step(all_actions[agent_id], agent_id=agent_id)
                next_states.append(next_state)
                rewards.append(reward)
                dones.append(done)
                score[agent_id] += float(reward)  # Score is
                # if done:
                #     break
            if render_gif:
                # OpenAI gym still renders the image to the screen even though it shouldn't. Eh.
                img = self.task.render(mode='rgb_array')
                self.__images.append(img)

            self.multi_agent.step(states, all_actions, rewards, next_states, dones)
            if log_loss_every is not None and (iterations % log_loss_every) == 0:
                self.log_loss()
            states = next_states
            if any(dones):
                break
        return score, iterations

    def run(
        self,
        reward_goal: float=100.0, max_episodes: int=2000,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        log_every=10, gif_every_episodes: Optional[int]=None,
        checkpoint_every=200, force_new=False,
    ) -> List[List[RewardType]]:
        """
        Evaluates the multi_agent in the environment.
        The evaluation will stop when the agent reaches the `reward_goal` in the averaged last `self.window_len`, or
        when the number of episodes reaches the `max_episodes`.

        To help debugging one can set the `gif_every_episodes` to a positive integer which relates to how often a gif
        of the episode evaluation is written to the disk.

        Every `checkpoint_every` (default: 200) iterations the Runner will store current state of the runner and the agent.
        These states can be used to resume previous run. By default the runner checks whether there is ongoing run for
        the combination of the environment and the agent.
        """
        self.epsilon = eps_start
        self.reset()
        if not force_new:
            self.load_state(self.model_path)

        while (self.episode < max_episodes):
            self.episode += 1
            render_gif = gif_every_episodes is not None and (self.episode % gif_every_episodes) == 0
            scores, iterations = self.interact_episode(self.epsilon, render_gif=render_gif)

            # TODO: Assumes that score for the episode is a sum of all. That might be Ok with coop but not in general.
            score = sum(scores)

            self.scores_window.append(score)
            self.all_iterations.append(iterations)
            self.all_scores.append(scores)

            mean_score: float = sum(self.scores_window) / len(self.scores_window)

            self.epsilon = max(eps_end, eps_decay * self.epsilon)

            if self.episode % log_every == 0:
                if hasattr(self.multi_agent, 'critic_loss') and self.multi_agent.critic_loss is not None:
                    loss = {'actor_loss': self.multi_agent.actor_loss, 'critic_loss': self.multi_agent.critic_loss}
                else:
                    loss = {'loss': self.multi_agent.loss}
                self.info(episode=self.episode, iterations=iterations, score=score, mean_score=mean_score, epsilon=self.epsilon, **loss)

            if render_gif and len(self.__images):
                gif_path = "gifs/{}_e{}.gif".format(self.model_path, str(self.episode).zfill(len(str(max_episodes))))
                save_gif(gif_path, self.__images)
                self.__images = []

            if mean_score >= reward_goal and len(self.scores_window) == self.window_len:
                print(f'Environment solved after {self.episode} episodes!\tAverage Score: {mean_score:.2f}')
                self.save_state(self.model_path)
                self.multi_agent.save_state(f'{self.model_path}_agent.net')
                break

            if self.episode % checkpoint_every == 0:
                self.save_state(self.model_path)

        return self.all_scores

    def info(self, **kwargs):
        """
        Writes out current state into provided loggers.
        Currently supports stdout logger (default on) and Tensorboard SummaryWriter initiated through EnvRun(writer=...)).
        """
        if self.writer is not None:
            self.log_writer(**kwargs)
        if self.logger is not None:
            self.log_logger(**kwargs)

    def log_logger(self, **kwargs):
        line_chunks = ["Episode {episode};", "Iter: {iterations};"]
        line_chunks += ["Current Score: {score:.2f};"]
        line_chunks += ["Average Score: {mean_score:.2f};"]
        if 'critic_loss' in self.multi_agent.__dict__ and self.multi_agent.critic_loss is not None:
            line_chunks += ["Actor loss: {actor_loss:10.4f};"]
            line_chunks += ["Critic loss: {critic_loss:10.4f};"]
        else:
            line_chunks += ["Loss: {loss:10.4f};"]
        line_chunks += ["Epsilon: {epsilon:5.3f};"]
        line = "\t".join(line_chunks)
        try:
            self.logger.info(line.format(**kwargs))
        except:
            print("Line: ", line)
            print("kwargs: ", kwargs)

    def log_writer(self, **kwargs):
        self.writer.add_scalar("score/score", kwargs['score'], self.episode)
        self.writer.add_scalar("score/avg_score", kwargs['mean_score'], self.episode)
        self.writer.add_scalar("epsilon", kwargs['epsilon'], self.iteration)
        self.log_loss(**kwargs)

    def log_loss(self, **kwargs):
        if hasattr(self.multi_agent, 'log_writer'):
            self.multi_agent.log_writer(self.writer, self.iteration)
        elif 'critic_loss' in self.multi_agent.__dict__ and self.multi_agent.critic_loss is not None:
            self.writer.add_scalar("Actor loss", kwargs['actor_loss'], self.iteration)
            self.writer.add_scalar("Critic loss", kwargs['critic_loss'], self.iteration)
        else:
            self.writer.add_scalar("loss", kwargs['loss'], self.iteration)

    def save_state(self, state_name: str):
        """Saves the current state of the runner and the multi_agent.

        Files are stored with appended episode number.
        Agents are saved with their internal saving mechanism.
        """
        state = {
            'tot_iterations': sum(self.all_iterations),
            'episode': self.episode,
            'epsilon': self.epsilon,
            'score': self.all_scores[-1],
            'average_score': sum(self.scores_window) / len(self.scores_window),
        }
        if hasattr(self.multi_agent, "critic_loss") and self.multi_agent.critic_loss is not None:
            state['actor_loss'] = self.multi_agent.actor_loss,
            state['critic_loss'] = self.multi_agent.critic_loss,
        elif hasattr(self.multi_agent, "loss"):
            state['loss'] = self.multi_agent.loss
        else:
            pass

        Path(self.state_dir).mkdir(parents=True, exist_ok=True)
        self.multi_agent.save_state(f'{self.state_dir}/{state_name}_e{self.episode}.agent')
        with open(f'{self.state_dir}/{state_name}_e{self.episode}.json', 'w') as f:
            json.dump(state, f)

    def load_state(self, state_prefix: str):
        """
        Loads state with the highest episode value for given agent and environment.
        """
        try:
            state_files = list(filter(lambda f: f.startswith(state_prefix) and f.endswith('json'), os.listdir(self.state_dir)))
            e = max([int(f[f.index('_e')+2:f.index('.')]) for f in state_files])
        except Exception:
            self.logger.warning("Couldn't load state. Forcing restart.")
            return

        state_name = [n for n in state_files if n.endswith(f"_e{e}.json")][0][:-5]
        self.logger.info("Loading saved state under: %s/%s.json", self.state_dir, state_name)
        with open(f'{self.state_dir}/{state_name}.json', 'r') as f:
            state = json.load(f)
        self.episode = state.get('episode')
        self.epsilon = state.get('epsilon')

        self.all_scores.append(state.get('score'))
        self.all_iterations = []

        avg_score = state.get('average_score')
        for _ in range(min(self.window_len, self.episode)):
            self.scores_window.append(avg_score)

        self.logger.info("Loading saved agent state: %s/%s.agent", self.state_dir, state_name)
        self.multi_agent.load_state(f'{self.state_dir}/{state_name}.agent')

        if hasattr(self.multi_agent, "actor_loss") and self.multi_agent.actor_loss is not None:
            self.multi_agent.actor_loss = state.get('actor_loss', 0)
            self.multi_agent.critic_loss = state.get('critic_loss', 0)
        else:
            self.multi_agent.loss = state.get('loss', 0)
