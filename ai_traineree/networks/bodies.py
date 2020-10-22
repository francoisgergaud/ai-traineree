from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
from typing import Optional, Sequence, Tuple, Union

from ai_traineree.networks import NetworkType


def hidden_init(layer: nn.Module):
    fan_in = layer.weight.data.size()[0]  # type: ignore
    lim = 1. / sqrt(fan_in)
    return (-lim, lim)


def layer_init(layer: nn.Module, range_value: Optional[Tuple[float, float]]=None):
    if not (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
        return
    if range_value is not None:
        layer.weight.data.uniform_(*range_value)  # type: ignore

    nn.init.xavier_uniform_(layer.weight)


class ScaleNet(NetworkType):
    def __init__(self, scale: Union[float, int]) -> None:
        super(ScaleNet, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


class FlattenNet(NetworkType):
    def __init__(self):
        super(FlattenNet, self).__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvNet(NetworkType):
    def __init__(self, input_dim: Sequence[int], output_dim: int=0, **kwargs):
        """
        Constructs a layered network over torch.nn.Conv2D. Number of layers is set based on `hidden_layers` argument.
        To update other arguments, e.g. kernel_size or bias, pass either a single value or a tuple of the same
        length as `hidden_layers`.

        Quick reminder from the PyTorch doc (https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html):
        in_channels (int) – Number of channels in the input image
        out_channels (int) – Number of channels produced by the convolution
        kernel_size (int or tuple) – Size of the convolving kernel
        stride (int or tuple, optional) – Stride of the convolution. Default: 1
        padding (int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional) – 'zeros', 'reflect', 'replicate' or 'circular'. Default: 'zeros'
        dilation (int or tuple, optional) – Spacing between kernel elements. Default: 1
        groups (int, optional) – Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional) – If True, adds a learnable bias to the output. Default: True

        Example 1:
        >>> config = {"hidden_layers": (300, 200, 100), "kernel_size": 6, "gate": F.relu}
        >>> net = ConvNet(input_dim=(10, 10, 3), output_dim=(3, 1), **config)

        Example 2:
        >>> config = {"hidden_layers": (64, 32, 64), "kernel_size": (3, 4, 3), padding: 2, "gate": F.relu}
        >>> net = ConvNet(input_dim=(20, 10, 1), output_dim=(3, 1), **config)
        """
        super(ConvNet, self).__init__()

        # input_dim = (num_layers, x_img, y_img, channels)
        self.input_dim = input_dim
        hidden_layers = kwargs.get("hidden_layers", (64, 64))
        kernel_size: Union[int, Sequence[int]] = kwargs.get("kernel_size", 3)
        num_layers = [input_dim[0]] + list(hidden_layers)  # + [output_dim]
        layers = []
        for (dim_in, dim_out) in zip(num_layers[:-1], num_layers[1:]):
            layers.append(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size))
            layers.append(nn.MaxPool2d(4, 4))

        # layers = [nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size) for dim_in, dim_out in zip(num_layers[:-1], num_layers[1:])]

        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

        # self.gate = gate if gate is not None else lambda x: x
        self.gate = kwargs.get("gate", lambda x: x)
        self.gate_out = kwargs.get("gate_out", lambda x: x)
        self.reset_parameters()

        self.device = kwargs.get("device")
        self.to(self.device)
    
    @property
    def output_size(self):
        return reduce(lambda a, b: a*b, self._calculate_output_size(self.input_dim, self.layers))

    def _calculate_output_size(self, input_dim: Sequence[int], layers) -> Sequence[int]:
        test_tensor = torch.zeros((1,) + tuple(input_dim)).to(self.device)
        with torch.no_grad():
            out = reduce(lambda x, layer: layer(x), layers, test_tensor)
        return out.shape

    def reset_parameters(self):
        self.layers.apply(layer_init)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        x = self.gate_out(self.layers[-1](x))
        return x


class FcNet(NetworkType):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: Sequence[int]=(200, 100),
                 gate=torch.tanh, gate_out=torch.tanh, last_layer_range=(-3e-3, 3e-3),
                 device: Optional[torch.device]=None,
                 **kwargs
                 ):
        """
        For the activation layer we use tanh by default which was observed to be much better, e.g. compared to ReLU,
        for policy networks [1]. The last gate, however, might be changed depending on the actual task.

        [1] "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study"
            Link: https://arxiv.org/abs/2006.05990
        """
        super(FcNet, self).__init__()

        num_layers = [input_dim] + list(hidden_layers) + [output_dim]
        layers = [nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(num_layers[:-1], num_layers[1:])]

        self.last_layer_range = last_layer_range
        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

        # self.gate = gate if gate is not None else lambda x: x
        self.gate = gate if gate is not None else lambda x: x
        self.gate_out = gate_out if gate_out is not None else lambda x: x
        self.to(device=device)

    def reset_parameters(self):
        for layer in self.layers[:-1]:
            layer_init(layer, hidden_init(layer))
        layer_init(self.layers[-1], self.last_layer_range)

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        if self.gate_out is None:
            return self.layers[-1](x)
        return self.gate_out(self.layers[-1](x))


###
# In most cases, the default ActorBody can be associated with a fully connected network.
# The alias is only for convinience and, hopefully, better understanding of some algorithms.
###
ActorBody = FcNet


class CriticBody(NetworkType):
    def __init__(self, input_dim: int, action_size: int, hidden_layers: Sequence[int]=(200, 100),
                 gate=torch.tanh, gate_out=None, **kwargs
                 ):
        super(CriticBody, self).__init__()

        num_layers = [input_dim] + list(hidden_layers) + [1]
        layers = [nn.Linear(in_dim, out_dim) for in_dim, out_dim in zip(num_layers[:-1], num_layers[1:])]

        # Injects `actions` into the second layer of the Critic
        layers[1] = nn.Linear(num_layers[1]+action_size, num_layers[2])
        self.layers = nn.ModuleList(layers)
        self.reset_parameters()

        self.gate = gate if gate is not None else lambda x: x

    def reset_parameters(self):
        for layer in self.layers:
            layer_init(layer, hidden_init(layer))

    def forward(self, x, actions):
        for idx, layer in enumerate(self.layers[:-1]):
            if idx == 1:
                x = self.gate(layer(torch.cat((x, actions.float()), dim=-1)))
            else:
                x = self.gate(layer(x))
        return self.layers[-1](x)


class NoisyLayer(nn.Module):
    def __init__(self, in_size: int, out_size: int, sigma: float=0.4, factorised: bool=True):
        """
        A Linear layer with values being pertrubed by the noise while training.

        :param sigma: float
            Used to intiated noise distribution.
        :param factorised: bool
            Whether to use independent Gaussian (False) or Factorised Gaussian (True) noise.
            Suggested [1] for DQN and Duelling nets to use factorised as it's quicker.

        Based on:
        [1] "Noisy Networks for Exploration" by Fortunato et al. (ICLR 2018), https://arxiv.org/abs/1706.10295.
        """
        super(NoisyLayer, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.sigma_0 = sigma
        self.factorised = factorised

        self.weight_mu = nn.Parameter(torch.zeros((out_size, in_size)))
        self.weight_sigma = nn.Parameter(torch.zeros((out_size, in_size)))

        self.bias_mu = nn.Parameter(torch.zeros(out_size))
        self.bias_sigma = nn.Parameter(torch.zeros(out_size))

        self.register_buffer('weight_eps', torch.zeros((out_size, in_size)))
        self.register_buffer('bias_eps', torch.zeros(out_size))

        self.bias_noise = torch.zeros(out_size)
        if factorised:
            self.weight_noise = torch.zeros(in_size)
        else:
            self.weight_noise = torch.zeros(out_size, in_size)

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        weight = self.weight_mu
        bias = self.bias_mu
        if self.training:
            weight = weight.add(self.weight_sigma.mul(self.weight_eps))
            bias = bias.add(self.bias_sigma.mul(self.bias_eps))

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        if self.factorised:
            bound = sqrt(1./self.in_size)
            sigma = self.sigma_0 * bound
        else:
            bound = sqrt(3./self.in_size)
            sigma = 0.017  # Yes, that's correct. [1]

        self.weight_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(sigma)

        self.bias_mu.data.uniform_(-bound, bound)
        self.bias_sigma.data.fill_(sigma)

    def reset_noise(self):
        self.bias_noise.normal_(std=self.sigma_0)
        self.weight_noise.normal_(std=self.sigma_0)

        if self.factorised:
            self.weight_eps.copy_(self.noise_function(self.bias_noise).ger(self.noise_function(self.weight_noise)))
            self.bias_eps.copy_(self.noise_function(self.bias_noise))
        else:
            self.weight_eps.copy_(self.weight_noise.data)
            self.bias_eps.copy_(self.bias_noise.data)

    @staticmethod
    def noise_function(x):
        return x.sign().mul_(x.abs().sqrt())


class NoisyNet(NetworkType):
    def __init__(self, in_size: int, out_size: int, hidden_layers=(100, 100), sigma=0.4,
                 gate=None, gate_out=None, factorised=True, device: Optional[torch.device]=None):
        """
            :param factorised: bool
                Whether to use independent Gaussian (False) or Factorised Gaussian (True) noise.
                Suggested [1] for DQN and Duelling nets to use factorised as it's quicker.
        """
        super(NoisyNet, self).__init__()

        num_layers = [in_size] + list(hidden_layers) + [out_size]
        layers = [NoisyLayer(dim_in, dim_out, sigma=sigma, factorised=factorised) for dim_in, dim_out in zip(num_layers[:-1], num_layers[1:])]
        self.layers = nn.ModuleList(layers)

        self.gate = gate if gate is not None else lambda x: x
        self.gate_out = gate_out if gate_out is not None else lambda x: x
        self.to(device=device)

    def reset_noise(self):
        for layer in self.layers:
            layer.reset_noise()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.gate(layer(x))
        return self.gate_out(self.layers[-1](x))
