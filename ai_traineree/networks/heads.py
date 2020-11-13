"""
Heads are build on Brains.
Like in real life, heads do all the difficult part of receiving stimuli,
being above everything else and not falling apart.
You take brains out and they just do nothng. Lazy.
The most common use case is when one head contains one brain.
But who are we to say what you can and cannot do.
You want two brains and a head within your head? Sure, go crazy.

What we're trying to do here is to keep thing relatively simple.
Unfortunately, not everything can be achieved [citation needed] with a serial
topography and at some point you'll need branching.
Heads are "special" in that each is built on networks/brains and will likely need
some special pipeping when attaching to your agent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from typing import Callable, Optional, List, Sequence
from ai_traineree.networks import NetworkType, NetworkTypeClass
from ai_traineree.networks.bodies import FcNet, NoisyNet


class NetChainer(NetworkType):
    """Chains nets into a one happy family.

    As it stands it is a wrapper around pytroch.nn.ModuleList.
    The need for wrapper comes from unified API to reset properties.
    """
    def __init__(self, net_classes: List[NetworkTypeClass], **kwargs):
        super(NetChainer, self).__init__()
        self.nets = nn.ModuleList(net_classes)

        self.in_features = self._determin_feature_size(self.nets[0].layers[0], is_in=True)
        self.out_features = self._determin_feature_size(self.nets[-1].layers[-1], is_in=False)

    @staticmethod
    def _determin_feature_size(layer, is_in=True):
        if 'Conv' in str(layer):
            return layer.in_channels if is_in else layer.out_channels
        else:
            return layer.in_features if is_in else layer.out_features

    def reset_parameters(self):
        for net in self.nets:
            if hasattr(net, "reset_parameters"):
                net.reset_parameters()

    def reset_noise(self):
        for net in self.nets:
            if hasattr(net, "reset_noise"):
                net.reset_noise()

    def forward(self, x):
        return reduce(lambda x, net: net(x), self.nets, x)


class DoubleCritic(NetworkType):
    def __init__(self, in_features: Sequence[int], action_size: Sequence[int], body_cls: NetworkTypeClass, **kwargs):
        super(DoubleCritic, self).__init__()
        hidden_layers = kwargs.get("hidden_layers", (200, 200))
        self.critic_1 = body_cls(in_features=in_features, action_size=action_size, hidden_layers=hidden_layers)
        self.critic_2 = body_cls(in_features=in_features, action_size=action_size, hidden_layers=hidden_layers)

    def reset_parameters(self):
        self.critic_1.reset_parameters()
        self.critic_2.reset_parameters()

    def act(self, states, actions):
        return (self.critic_1.act(states, actions), self.critic_2.act(states, actions))

    def forward(self, state, actions):
        return (self.critic_1(state, actions), self.critic_2(state, actions))


class DuelingNet(NetworkType):
    def __init__(self, input_shape: Sequence[int], output_shape: Sequence[int], hidden_layers: Sequence[int],
                 net_fn: Optional[Callable[..., NetworkType]]=None,
                 net_class: Optional[NetworkTypeClass]=None,
                 **kwargs
    ):
        """
        Parameters
        ----------
            input_shape : Tuple of ints
                Shape of the input. Even in case when input is 1D, a single item tuple is expected, e.g. (4,).
            output_shape : Tuple of ints
                Shape of the output. Same as with the `input_shape`.
        """
        super(DuelingNet, self).__init__()
        device = kwargs.get("device")
        # We only care about the leading size, e.g. (4,) -> 4
        if net_fn is not None:
            self.value_net = net_fn(input_shape, (1,), hidden_layers=hidden_layers)
            self.advantage_net = net_fn(input_shape, output_shape, hidden_layers=hidden_layers)
        elif net_class is not None:
            self.value_net = net_class(input_shape, (1,), hidden_layers=hidden_layers, device=device)
            self.advantage_net = net_class(input_shape, output_shape, hidden_layers=hidden_layers, device=device)
        else:
            self.value_net = FcNet(input_shape, (1,), hidden_layers=hidden_layers, gate_out=None, device=device)
            self.advantage_net = FcNet(input_shape, output_shape, hidden_layers=hidden_layers, gate_out=None, device=device)

    def reset_parameters(self) -> None:
        self.value_net.reset_parameters()
        self.advantage_net.reset_parameters()

    def act(self, x):
        value = self.value_net.act(x).float()
        advantage = self.advantage_net.act(x).float()
        q = value.expand_as(advantage) + (advantage - advantage.mean(1, keepdim=True).expand_as(advantage))
        return q

    def forward(self, x):
        value = self.value_net(x).float()
        advantage = self.advantage_net(x).float()
        q = value.expand_as(advantage) + (advantage - advantage.mean(1, keepdim=True).expand_as(advantage))
        return q


class CategoricalNet(NetworkType):
    """
    Computes discrete probability distribution for the state-action Q function.

    CategoricalNet [1] learns significantly different compared to other nets here.
    For this reason it won't be suitable for simple replacement in most (current) agents.
    Please check the Agent whether it supports. 

    The algorithm is used in the RainbowNet but not this particular net.

    [1] "A Distributional Perspective on Reinforcement Learning" (2017) by M. G. Bellemare, W. Dabney, R. Munos.
        Link: http://arxiv.org/abs/1707.06887
    """
    def __init__(self, state_size: int, action_size: int, 
                 n_atoms: int=21, v_min: float=-10., v_max: float=10.,
                 hidden_layers: Sequence[int]=(200, 200),
                 net: Optional[NetworkType]=None,
                 device: Optional[torch.device]=None,
                 ):
        super(CategoricalNet, self).__init__()
        self.action_size = action_size
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.z_atoms = torch.linspace(v_min, v_max, n_atoms, device=device)
        self.z_delta = self.z_atoms[1] - self.z_atoms[0]
        self.net = net if net is not None else NoisyNet(state_size, action_size*n_atoms, hidden_layers=hidden_layers, device=device)
        self.to(device=device)

    def reset_paramters(self):
        self.net.reset_parameters()

    def forward(self, x, log_prob=False) -> torch.Tensor:
        """
        :param log_prob: bool
            Whether to return log(prob) which uses pytorch's function. According to doc it's quicker and more stable
            than taking prob.log().
        """
        return self.net(x).view((-1, self.action_size, self.num_atoms))


class RainbowNet(NetworkType, nn.Module):
    """Rainbow networks combines dueling and categorical networks.

    """
    def __init__(self, input_shape: Sequence[int], output_shape: Sequence[int], num_atoms: int, hidden_layers=(200, 200), noisy=False, device=None, **kwargs):
        """
        Parameters
        ----------
        pre_network_fn : func
            A shared network that is used before *value* and *advantage* networks.
        """
        super(RainbowNet, self).__init__()

        self.pre_network = None
        in_features = input_shape[0]
        out_features = output_shape[0]
        if 'pre_network_fn' in kwargs:
            self.pre_network = kwargs.get("pre_network_fn")(in_features=in_features)
            self.pre_netowrk_params = self.pre_network.parameters()  # Registers pre_network's parameters to this module
            in_features = self.pre_network.out_features

        if noisy:
            self.value_net = NoisyNet(in_features, num_atoms, hidden_layers=hidden_layers, device=device)
            self.advantage_net = NoisyNet(in_features, (out_features*num_atoms,), hidden_layers=hidden_layers, device=device)
        else:
            self.value_net = FcNet(in_features, num_atoms, hidden_layers=hidden_layers, gate_out=None, device=device)
            self.advantage_net = FcNet((in_features,), (out_features*num_atoms,), hidden_layers=hidden_layers, gate_out=None, device=device)

        self.noisy = noisy
        self.in_features = in_features if self.pre_network is None else self.pre_network.in_features
        self.out_features = out_features
        self.num_atoms = num_atoms
        self.to(device=device)

    def reset_noise(self):
        if self.noisy:
            self.value_net.reset_noise()
            self.advantage_net.reset_noise()

    def act(self, x, log_prob=False):
        """
        :param log_prob: bool
            Whether to return log(prob) which uses pytorch's function. According to doc it's quicker and more stable
            than taking prob.log().
        """
        with torch.no_grad():
            self.eval()
            if self.pre_network is not None:
                x = self.pre_network(x)
            value = self.value_net.act(x).view(-1, 1, self.num_atoms)
            advantage = self.advantage_net.act(x).view(-1, self.out_features, self.num_atoms)
            q = value + advantage - advantage.mean(1, keepdim=True)
            # Doc: It's computationally quicker than log(softmax) and more stable
            out = F.softmax(q, dim=-1) if not log_prob else F.log_softmax(q, dim=-1)
            self.train()
        return out

    def forward(self, x, log_prob=False):
        """
        :param log_prob: bool
            Whether to return log(prob) which uses pytorch's function. According to doc it's quicker and more stable
            than taking prob.log().
        """
        if self.pre_network is not None:
            x = self.pre_network(x)
        value = self.value_net(x).view((-1, 1, self.num_atoms))
        advantage = self.advantage_net(x).view(-1, self.out_features, self.num_atoms)
        q = value + advantage - advantage.mean(1, keepdim=True)
        if log_prob:
            # Doc: It's computationally quicker than log(softmax) and more stable
            return F.log_softmax(q, dim=-1)
        return F.softmax(q, dim=-1)
