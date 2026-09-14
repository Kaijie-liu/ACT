"""Versioned output-top2 convolutional family, separate from frozen MLP v1."""
from dataclasses import dataclass
import torch
from torch import nn

from act.back_end.moe.model import OutputLevelMoE
from act.back_end.moe.schema import GateKind, OutputLevelMoESpec


@dataclass(frozen=True)
class ConvOutputMoEConfig:
    input_shape: tuple = (3,32,32)
    num_classes: int = 10
    num_experts: int = 4
    channels: tuple = (16,32)
    hidden: int = 64
    router_pool: int = 4
    seed: int = 17

    def __post_init__(self):
        if len(self.input_shape)!=3 or len(self.channels)!=2:
            raise ValueError('CHW input and two convolution widths required')
        c,h,w=self.input_shape
        if min(c,h,w,*self.channels,self.hidden,self.router_pool)>0 and self.num_classes>=2 and self.num_experts>=2:
            if h%8==0 and w%8==0 and h%self.router_pool==0 and w%self.router_pool==0:
                return
        raise ValueError('invalid dimensions for frozen stride/pooling topology')


def build_conv_output_moe(config):
    c,h,w=config.input_shape; a,b=config.channels; pool=config.router_pool
    # CPU construction never resets an unrelated GPU RNG.
    with torch.device('cpu'), torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        router=nn.Sequential(nn.AvgPool2d(pool),nn.Flatten(),
                             nn.Linear(c*(h//pool)*(w//pool),config.num_experts))
        experts=[]
        for _ in range(config.num_experts):
            experts.append(nn.Sequential(
                nn.Conv2d(c,a,3,stride=2,padding=1),nn.ReLU(),
                nn.Conv2d(a,b,3,stride=2,padding=1),nn.ReLU(),
                nn.AvgPool2d(2),nn.Flatten(),
                nn.Linear(b*(h//8)*(w//8),config.hidden),nn.ReLU(),
                nn.Linear(config.hidden,config.num_classes)))
    spec=OutputLevelMoESpec(num_experts=config.num_experts,top_k=2,
                           gate=GateKind.SELECTED_SOFTMAX,normalized=True)
    return OutputLevelMoE(router,experts,spec)
