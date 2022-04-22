import math
import torch.nn as nn
import torch.nn.functional as F
from .modules.activations import SEBlock

__all__ = ['vanilla', 'resnet', 'resnet_se']

def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()

    for m in model.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, ResBasicBlock):
            nn.init.constant_(m.bn2.weight, 0)

class BasicBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2, bias=True):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        self.bach_norm = nn.BatchNorm2d(num_features=out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.bach_norm(self.conv(x)))
        return x

class ResBasicBlock(nn.Module):
    def __init__(self, in_planes, planes,  stride=1, expansion=1, downsample=None, groups=1, residual_block=None):
        super(ResBasicBlock, self).__init__()       
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, expansion * planes, kernel_size=3, stride=1, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(expansion * planes)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    def __init__(self, in_planes, planes,  stride=1, expansion=4, downsample=None, groups=1, residual_block=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out

class Vanilla(nn.Module):
    def __init__(self, in_channels, num_features, max_features, num_blocks, num_classes):
        super(Vanilla, self).__init__()
        # features
        blocks = [BasicBlock(in_channels=in_channels, out_channels=num_features, kernel_size=5, padding=2, stride=1)]
        for i in range(1, num_blocks):
            blocks.append(BasicBlock(in_channels=min(max_features, num_features * min(pow(2, i), 8)), out_channels=min(max_features,num_features * min(pow(2, i + 1), 8)), stride=2))
        self.features = nn.Sequential(*blocks)

        # pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(min(max_features,num_features * min(pow(2, num_blocks), 8)), num_features),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(num_features, num_classes)
        )

        init_model(self)
        
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

class ResNet(nn.Module):
    def __init__(self, in_channels, in_planes, width, layers, block, expansion, residual_block, groups, num_classes):
        super(ResNet, self).__init__()
        # parameters
        self.in_planes = in_planes

        # image to features
        self.image_to_features = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=self.in_planes, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(num_features=self.in_planes),
            nn.ReLU(inplace=True))

        # features
        blocks = []
        for i in range(len(layers)):
            blocks.append(self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion, stride=1 if i == 0 else 2, groups=groups[i], residual_block=residual_block))
        self.features = nn.Sequential(*blocks)

        # pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(width[-1] * expansion, in_planes),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(in_planes, num_classes))

        init_model(self)

    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * expansion),
            )
        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        layers.append(block(self.in_planes, planes, stride, expansion=expansion, downsample=downsample, groups=groups, residual_block=residual_block))
        self.in_planes = planes * expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, expansion=expansion, groups=groups, residual_block=residual_block))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.image_to_features(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

def vanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 32)
    config.setdefault('max_features', 256)
    config.setdefault('num_blocks', 5)
    config.setdefault('num_classes', 4)

    return Vanilla(**config)

def resnet(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('in_planes', 32)
    config.setdefault('width', [32, 32, 64, 64, 128, 128])
    config.setdefault('layers', [2, 2, 2, 2, 2, 2])
    config.setdefault('groups', [1, 1, 1, 1, 1, 1])
    config.setdefault('expansion', 1)
    config.setdefault('num_classes', 4)

    config['block'] = ResBasicBlock
    config['residual_block'] = None

    return ResNet(**config)

def resnet_se(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('in_planes', 32)
    config.setdefault('width', [32, 32, 64, 64, 128, 128])
    config.setdefault('layers', [2, 2, 2, 2, 2, 2])
    config.setdefault('groups', [1, 1, 1, 1, 1, 1])
    config.setdefault('expansion', 1)
    config.setdefault('num_classes', 4)

    config['block'] = ResBasicBlock
    config['residual_block'] = SEBlock

    return ResNet(**config)