from torch import nn
from torch.nn import functional as F
import torch
from .densenet import DenseNet
from typing import Optional, Callable, Type, Union

def get_model(model_type, *args, **kwargs):
    if model_type == 'cnn':
        return CellCNN(*args, **kwargs)
    elif model_type == 'resnet':
        return CellResNet(BasicBlock, *args, **kwargs)
    elif model_type == 'cnn-lstm':
        return CellCNNLSTM(*args, **kwargs)
    elif model_type == 'densenet':
        return DenseNet(1, loss_fnc=args[0], num_classes=args[1], time=args[2])
    else:
        raise ValueError()

class CellCNN(nn.Module):
    def __init__(self, loss, num_classes = 2, time = 30):
        super(CellCNN, self).__init__()
        self.num_classes = num_classes
        self.loss_fnc = loss
        
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            # nn.BatchNorm3d(8, momentum=0.01, track_running_stats=False),
            nn.LayerNorm([8, 200 * (time // 30), 8, 8]),
            nn.MaxPool3d((2,2,2)),
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.LayerNorm([16, 100 * (time // 30), 4, 4]),
            # nn.BatchNorm3d(16, momentum=0.01, track_running_stats=False),
            nn.MaxPool3d((1,2,2)),
            nn.Flatten(),
            nn.Linear(6400 * (time // 30), 32),
            nn.ReLU(),
            # nn.BatchNorm1d(32, momentum=0.01, track_running_stats=False),
            nn.LayerNorm([32]),
            nn.Linear(32, self.num_classes)
        )
        
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x
    
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv3d:
    """3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        dimensions: list,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.dropout1 = nn.Dropout3d(0.2, inplace=True)
        self.bn1 = nn.LayerNorm([*dimensions])
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, groups, dilation)
        # self.dropout2 = nn.Dropout3d(0.2, inplace=True)
        self.bn2 = nn.LayerNorm([*dimensions])
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        # out = self.dropout1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.dropout2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
    
class Bottleneck(nn.Module):

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        dimension: list,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        # width = int(planes * (base_width / 64.0)) * groups
        width = planes // 4
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.LayerNorm([width, *dimension[1:]])
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.LayerNorm([width, *dimension[1:]])
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.LayerNorm([planes * self.expansion, *dimension[1:]])
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CellResNet(nn.Module):
    def __init__(self, 
                 block: Type[Union[BasicBlock, Bottleneck]],
                 loss, num_classes = 2, time = 30):
        super(CellResNet, self).__init__()
        self.num_classes = num_classes
        self.loss_fnc = loss
        
        self.bl1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            # nn.BatchNorm3d(8, momentum=0.01, track_running_stats=False),
            nn.LayerNorm([8, 200 * (time // 30), 8, 8])
        )
        self.res_block1 = self.__make_residual_block(block, 8, 16, [16, 200 * (time // 30), 8, 8])
        self.res_block2 = self.__make_residual_block(block, 16, 16, [16, 200 * (time // 30), 8, 8])
        self.max_pool1 = nn.MaxPool3d((2,2,2))
        self.res_block3 = self.__make_residual_block(block, 16, 32, [32, 100 * (time // 30), 4, 4])
        self.res_block4 = self.__make_residual_block(block, 32, 32, [32, 100 * (time // 30), 4, 4])
        self.max_pool2 = nn.MaxPool3d((1,2,2))
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12800 * (time // 30), 32),
            nn.ReLU(),
            # nn.BatchNorm1d(32, momentum=0.01, track_running_stats=False),
            nn.LayerNorm([32]),
            nn.Linear(32, self.num_classes)
        )

        self.net = [self.bl1, self.res_block1, self.max_pool1, self.res_block3, self.max_pool2, self.lin]
        
    def __make_residual_block(self, block, inplanes, planes, dimensions):
        return block(inplanes, planes, dimensions, downsample=nn.Sequential(conv1x1(inplanes, planes), nn.LayerNorm([*dimensions])))
    
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x

class CellCNNLSTM(nn.Module):
    def __init__(self, loss, num_classes = 2, time = 30):
        super(CellCNNLSTM, self).__init__()
        self.num_classes = num_classes
        self.loss_fnc = loss
        
        
        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.LayerNorm([8, 200 * (time // 30), 8, 8]),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.LayerNorm([16, 200 * (time // 30), 4, 4]),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout3d(0.2),
            nn.LayerNorm([8, 200 * (time // 30), 2, 2]),
            nn.MaxPool3d((1, 2, 2)),
            # nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Dropout3d(0.2),
            # nn.LayerNorm([1, 200 * (time // 30), 1, 1]),
            
        )
        self.bridge = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8 * 200 * (time // 30), 200 * (time // 30))
        )
        
        self.rnn = nn.Sequential(
            nn.LSTM(input_size=200 * (time // 30), hidden_size=256, num_layers=3)
        )
        
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.LayerNorm([32]),
            nn.Linear(32, self.num_classes)
        )
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bridge(out)
        out, hidden = self.rnn(out)
        out = self.lin(out)
        return out