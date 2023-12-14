import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

time = 400

class _DenseLayer(nn.Sequential):

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dims):
        super().__init__()
        self.add_module('norm1', nn.LayerNorm([num_input_features, *dims]))
        self.add_module('relu1', set_activation())
        self.add_module(
            'conv1',
            nn.Conv3d(num_input_features,
                      bn_size * growth_rate,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('norm2', nn.LayerNorm([bn_size * growth_rate, *dims]))
        self.add_module('relu2', set_activation())
        self.add_module(
            'conv2',
            nn.Conv3d(bn_size * growth_rate,
                      growth_rate,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super().forward(x)
        # print(new_features.shape)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features,
                                     p=self.drop_rate,
                                     training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate, dims):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate, dims)
            self.add_module('denselayer{}'.format(i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(self, num_input_features, num_output_features, dims):
        super().__init__()
        self.add_module('norm', nn.LayerNorm([num_input_features, *dims]))
        self.add_module('relu', set_activation())
        self.add_module(
            'conv',
            nn.Conv3d(num_input_features,
                      num_output_features,
                      kernel_size=1,
                      stride=1,
                      bias=False))
        self.add_module('pool', nn.MaxPool3d((1,2,2)))


class DenseNet(nn.Module):
    """Densenet-BC model class
    Args:
        growth_rate (int) - how many filters to add each layer (k in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 growth_rate=8,
                 block_config=(2, 2, 2),
                 num_init_features=8,
                 bn_size=4,
                 drop_rate=0.2,
                 num_classes=1000,
                 loss_fnc=nn.CrossEntropyLoss(), time = 90,
                 ):
    
        super().__init__()
        
        self.loss_fnc = loss_fnc
        
        dims = [200 * (time // 30), 8, 8]

        # First convolution
        self.features = [('conv1',
                          nn.Conv3d(n_input_channels,
                                    num_init_features,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=False)),
                         ('norm1', nn.LayerNorm([num_init_features, *dims])),
                         ('relu1', set_activation())]
        if not no_max_pool:
            self.features.append(
                ('pool1', nn.MaxPool3d((1,2,2)))
                )
        dims[1] //= 2
        dims[2] //= 2
        self.features = nn.Sequential(OrderedDict(self.features))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers,
                                num_input_features=num_features,
                                bn_size=bn_size,
                                growth_rate=growth_rate,
                                drop_rate=drop_rate,
                                dims=dims)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2, dims=dims)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2
                dims[1] //= 2
                dims[2] //= 2

        # Final batch norm
        self.features.add_module('norm5', nn.LayerNorm([num_features, *dims]))

        # Linear layer
        # self.classifier = nn.Linear(num_features, num_classes)
        self.classifier = nn.Sequential(
            nn.Linear(num_features * dims[0], 32),
            nn.ReLU(),
            nn.LayerNorm([32]),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        activation_method = getattr(F, 'relu')
        out = activation_method(features, inplace=True)
        # out = F.adaptive_max_pool3d(out,
        #                             output_size=(1, 1,
        #                                          1)).view(features.size(0), -1)
        out = out.view(out.size()[0], -1)
        out = self.classifier(out)
        return out


def set_activation():
    # assert(Config.activation =='leaky_relu' or Config.activation == 'relu')
    # if Config.activation == 'leaky_relu':
    #     return nn.LeakyReLU(Config.negative_slope, inplace=True)
    return nn.ReLU(inplace=True)
