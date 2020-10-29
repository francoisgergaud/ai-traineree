from ai_traineree import DEVICE
from ai_traineree.agents.utils import soft_update
from ai_traineree.buffers import NStepBuffer, PERBuffer, ReplayBuffer
from ai_traineree.networks import NetworkType, NetworkTypeClass
from ai_traineree.networks.heads import DuelingNet
from ai_traineree.types import AgentType
from ai_traineree.utils import to_tensor

import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from typing import Callable, Dict, Optional, Type, Sequence, Union


class DQNAgent(AgentType):
    """Deep Q-Learning Network (DQN).

    The agent is not a vanilla DQN, although can be configured as such.
    The default config includes dual dueling nets and the priority experience buffer.
    Learning is also delayed by slowly copying to target nets (via tau parameter).
    Although NStep is implemented the default value is 1-step reward.

    There is also a specific implemntation of the DQN called the Rainbow which differs
    to this implementation by working on the discrete space projection of the Q(s,a) function.
    """

    name = "DQN"

    def __init__(
        self, state_size: Union[Sequence[int], int], action_size: int,
        lr: float = 3e-4, gamma: float = 0.99, tau: float = 0.002,
        network_fn: Callable[[], NetworkType]=None,
        network_class: Type[NetworkTypeClass]=None,
        hidden_layers: Sequence[int]=(64, 64),
        state_transform: Optional[Callable]=None,
        reward_transform: Optional[Callable]=None,
        device=None, **kwargs
    ):
        """
        Accepted parameters:
        :param float lr: learning rate (default: 1e-3)
        :param float gamma: discount factor (default: 0.99)
        :param float tau: soft-copy factor (default: 0.002)
        """

        self.device = device if device is not None else DEVICE
        self.state_size = state_size if not isinstance(state_size, int) else (state_size,)
        self.action_size = action_size

        self.lr = float(kwargs.get('lr', lr))
        self.gamma = float(kwargs.get('gamma', gamma))
        self.tau = float(kwargs.get('tau', tau))

        self.update_freq = int(kwargs.get('update_freq', 1))
        self.batch_size = int(kwargs.pop('batch_size', 32))
        self.buffer_size = int(kwargs.pop('buffer_size', 1e5))
        self.warm_up = int(kwargs.get('warm_up', 0))
        self.number_updates = int(kwargs.get('number_updates', 1))
        self.max_grad_norm = float(kwargs.get('max_grad_norm', 10))

        self.iteration: int = 0
        self.buffer = PERBuffer(batch_size=self.batch_size, buffer_size=self.buffer_size, **kwargs)
        self.using_double_q = bool(kwargs.get("using_double_q", True))

        self.n_steps = kwargs.get("n_steps", 1)
        self.n_buffer = NStepBuffer(n_steps=self.n_steps, gamma=self.gamma)

        self.state_transform = state_transform if state_transform is not None else lambda x: x
        self.reward_transform = reward_transform if reward_transform is not None else lambda x: x
        if network_fn is not None:
            self.net = network_fn()
            self.target_net = network_fn()
        elif network_class is not None:
            self.net = network_class(self.state_size[0], self.action_size, hidden_layers=hidden_layers, device=self.device)
            self.target_net = network_class(self.state_size[0], self.action_size, hidden_layers=hidden_layers, device=self.device)
        else:
            hidden_layers = kwargs.get('hidden_layers', hidden_layers)
            self.net = DuelingNet(self.state_size[0], self.action_size, hidden_layers=hidden_layers, device=self.device)
            self.target_net = DuelingNet(self.state_size[0], self.action_size, hidden_layers=hidden_layers, device=self.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr)
        self.loss = 0

    def step(self, state, action, reward, next_state, done) -> None:
        self.iteration += 1
        state = to_tensor(self.state_transform(state)).float().to("cpu")
        next_state = to_tensor(self.state_transform(next_state)).float().to("cpu")
        reward = self.reward_transform(reward)

        # Delay adding to buffer to account for n_steps (particularly the reward)
        self.n_buffer.add(state=state.numpy(), action=[int(action)], reward=[reward], done=[done], next_state=next_state.numpy())
        if not self.n_buffer.available:
            return

        self.buffer.add(**self.n_buffer.get().get_dict())

        if self.iteration < self.warm_up:
            return

        if len(self.buffer) >= self.batch_size and (self.iteration % self.update_freq) == 0:
            for _ in range(self.number_updates):
                self.learn(self.buffer.sample())

    def act(self, state, eps: float = 0.) -> int:
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() < eps:
            return random.randint(0, self.action_size-1)

        state = to_tensor(self.state_transform(state)).float()
        state = state.unsqueeze(0).to(self.device)
        action_values = self.net.act(state)
        return int(torch.argmax(action_values.cpu()))

    def learn(self, experiences) -> None:
        rewards = to_tensor(experiences['reward']).type(torch.float32).to(self.device)
        dones = to_tensor(experiences['done']).type(torch.int).to(self.device)
        states = to_tensor(experiences['state']).type(torch.float32).to(self.device)
        next_states = to_tensor(experiences['next_state']).type(torch.float32).to(self.device)
        actions = to_tensor(experiences['action']).type(torch.long).to(self.device)

        with torch.no_grad():
            Q_targets_next = self.target_net(next_states).detach()
            if self.using_double_q:
                _a = torch.argmax(self.net(next_states), dim=-1).unsqueeze(-1)
                max_Q_targets_next = Q_targets_next.gather(1, _a)
            else:
                max_Q_targets_next = Q_targets_next.max(1)[0].unsqueeze(1)
        Q_targets = rewards + self.n_buffer.n_gammas[-1] * max_Q_targets_next * (1 - dones)
        Q_expected = self.net(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.loss = loss.item()

        if hasattr(self.buffer, 'priority_update'):
            error = Q_expected - Q_targets
            assert any(~torch.isnan(error))
            self.buffer.priority_update(experiences['index'], error.abs())

        # Update networks - sync local & target
        soft_update(self.target_net, self.net, self.tau)

    def describe_agent(self) -> Dict:
        """Returns agent's state dictionary."""
        return self.net.state_dict()

    def log_writer(self, writer, episode):
        writer.add_scalar("loss/agent", self.loss, episode)

    def save_state(self, path: str):
        agent_state = dict(net=self.net.state_dict(), target_net=self.target_net.state_dict())
        torch.save(agent_state, path)

    def load_state(self, path: str):
        agent_state = torch.load(path)
        self.net.load_state_dict(agent_state['net'])
        self.target_net.load_state_dict(agent_state['target_net'])

    def save_buffer(self, path: str):
        import json
        dump = self.buffer.dump_buffer(serialize=True)
        with open(path, 'w') as f:
            json.dump(dump, f)

    def load_buffer(self, path: str):
        import json
        with open(path, 'r') as f:
            buffer_dump = json.load(f)
        self.buffer.load_buffer(buffer_dump)
