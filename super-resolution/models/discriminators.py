import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import spectral_norm as SpectralNorm
from models.modules.misc import random_crop
from models.modules.sparse import SparseCirConv2d, ApproxSparseCirConv2d 
from models.vgg import VGGFeaturesExtractor, MultiVGGFeaturesExtractor

__all__ = ['d_vanilla', 'd_sanvanilla', 'd_advanilla', 'd_sanadvanilla', 'd_vgghead', 'd_multivgghead']

def initialize_model(model, scale=1.):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            m.weight.data *= scale
            if m.bias is not None:
                m.bias.data.zero_()
        else:
            continue

class BasicBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=True, normalization=False):
        super(BasicBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=bias)
        self.bach_norm = nn.BatchNorm2d(num_features=out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if normalization:
            self.conv = SpectralNorm(self.conv)

    def forward(self, x):
        x = self.lrelu(self.bach_norm(self.conv(x)))
        return x

class VanillaBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, bias=False, normalization=False):
        super(VanillaBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)

        if normalization:
            self.conv1 = SpectralNorm(self.conv1)
            self.conv2 = SpectralNorm(self.conv2)

    def forward(self, x):
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        return x

class SANVanillaBlock(nn.Module):
    def __init__(self, conv_block, in_channels=64, out_channels=64, bias=False, gain=1.0):
        super(SANVanillaBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = conv_block(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=bias, gain=gain)
        self.conv2 = conv_block(in_channels=out_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=bias, gain=gain)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=True)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.lrelu(self.bn1(self.conv1(x)))
        x = self.lrelu(self.bn2(self.conv2(x)))
        return x

class Vanilla(nn.Module):
    def __init__(self, in_channels, num_features, max_features, num_blocks, normalization):
        super(Vanilla, self).__init__()
        # parameters
        self.crop_size = 4 * pow(2, num_blocks)

        # features
        blocks = [VanillaBlock(in_channels=in_channels, out_channels=num_features, bias=True, normalization=normalization)]
        for i in range(0, num_blocks - 1):
            blocks.append(VanillaBlock(in_channels=min(max_features, num_features * pow(2, i)), out_channels=min(max_features, num_features * pow(2, i + 1)), normalization=normalization))
        self.features = nn.Sequential(*blocks)
        
        # classifier
        if normalization:
            self.classifier = nn.Sequential(
                SpectralNorm(nn.Linear(min(max_features, num_features * pow(2, num_blocks - 1)) * 4 * 4, num_features)),
                nn.LeakyReLU(negative_slope=0.1),
                SpectralNorm(nn.Linear(num_features, 1)))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(min(max_features, num_features * pow(2, num_blocks - 1)) * 4 * 4, num_features),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(num_features, 1))

        # initialize weights
        initialize_model(self)

    def forward(self, x):
        x = random_crop(x, self.crop_size, self.crop_size)
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

class SANVanilla(nn.Module):
    def __init__(self, in_channels, num_features, max_features, num_blocks, gain, conv_block):
        super(SANVanilla, self).__init__()
        # parameters
        self.crop_size = 4 * pow(2, num_blocks)

        # features
        blocks = [SANVanillaBlock(conv_block=conv_block, in_channels=in_channels, out_channels=num_features, bias=True, gain=gain)]
        for i in range(0, num_blocks - 1):
            blocks.append(SANVanillaBlock(conv_block=conv_block, in_channels=min(max_features, num_features * pow(2, i)), out_channels=min(max_features, num_features * pow(2, i + 1)), gain=gain))
        self.features = nn.Sequential(*blocks)
        
        # classifier
        self.classifier = nn.Sequential(
            SpectralNorm(nn.Linear(min(max_features, num_features * pow(2, num_blocks - 1)) * 4 * 4, num_features)),
            nn.LeakyReLU(negative_slope=0.1),
            SpectralNorm(nn.Linear(num_features, 1)))
        
        # initialize weights
        initialize_model(self)

    def forward(self, x):
        x = random_crop(x, self.crop_size, self.crop_size)
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

class AdaptiveVanilla(nn.Module):
    def __init__(self, in_channels, num_features, max_features, num_blocks, normalization):
        super(AdaptiveVanilla, self).__init__()
        # features
        blocks = [VanillaBlock(in_channels=in_channels, out_channels=num_features, bias=True, normalization=normalization)]
        for i in range(0, num_blocks - 1):
            blocks.append(VanillaBlock(in_channels=min(max_features, num_features * pow(2, i)), out_channels=min(max_features, num_features * pow(2, i + 1)), normalization=normalization))
        self.features = nn.Sequential(*blocks)

        # pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # classifier
        if normalization:
            self.classifier = nn.Sequential(
                SpectralNorm(nn.Linear(min(max_features, num_features * pow(2, num_blocks - 1)), num_features)),
                nn.LeakyReLU(negative_slope=0.1),
                SpectralNorm(nn.Linear(num_features, 1)))
        else:
            self.classifier = nn.Sequential(
                nn.Linear(min(max_features, num_features * pow(2, num_blocks - 1)), num_features),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(num_features, 1))

        # initialize weights
        initialize_model(self)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

class SANAdaptiveVanilla(nn.Module):
    def __init__(self, in_channels, num_features, max_features, num_blocks, gain, conv_block):
        super(SANAdaptiveVanilla, self).__init__()
        # features
        blocks = [SANVanillaBlock(conv_block=conv_block, in_channels=in_channels, out_channels=num_features, bias=True, gain=gain)]
        for i in range(0, num_blocks - 1):
            blocks.append(SANVanillaBlock(conv_block=conv_block, in_channels=min(max_features, num_features * pow(2, i)), out_channels=min(max_features, num_features * pow(2, i + 1)), gain=gain))
        self.features = nn.Sequential(*blocks)

        # pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        # classifier
        self.classifier = nn.Sequential(
            SpectralNorm(nn.Linear(min(max_features, num_features * pow(2, num_blocks - 1)), num_features)),
            nn.LeakyReLU(negative_slope=0.1),
            SpectralNorm(nn.Linear(num_features, 1)))

        # initialize weights
        initialize_model(self)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x

class VGGHead(nn.Module):
    def __init__(self, in_channels, num_features, max_features, num_blocks, feature_layer, normalization):
        super(VGGHead, self).__init__()
        # extractor
        self.extractor = VGGFeaturesExtractor(feature_layer=feature_layer)
        in_channels = self.extractor.features[-2].weight.data.size(0)

        # classifer
        self.classifier = Classifier(in_channels=in_channels, num_features=num_features, max_features=max_features, num_blocks=num_blocks, normalization=normalization)

    def forward(self, x):
        f = self.extractor(x)
        y = self.classifier(f)

        return y

class MultiVGGHead(nn.Module):
    def __init__(self, in_channels, num_features, max_features, num_blocks, target_features, normalization):
        super(MultiVGGHead, self).__init__()
        # parameters
        self.target_features = sorted(target_features)
        
        # extractor
        self.extractor = MultiVGGFeaturesExtractor(target_features=self.target_features)

        # classifier 
        self.classifier = MultiClassifier(in_channels=in_channels, num_features=num_features, max_features=max_features, num_blocks=num_blocks, target_features=self.target_features, normalization=normalization)

    def forward(self, x):
        f = self.extractor(x)
        y = self.classifier(f)
        
        return y

class Classifier(nn.Module):
    def __init__(self, in_channels, num_features, max_features, num_blocks, normalization=False):
        super(Classifier, self).__init__()
        # layers
        blocks = [BasicBlock(in_channels=in_channels, out_channels=min(max_features, num_features * pow(2, num_blocks - 1)), kernel_size=3, padding=0, normalization=normalization)]
        for i in range(num_blocks - 1, 0, -1):
            blocks.append(BasicBlock(in_channels=min(max_features, num_features * pow(2, i)), out_channels=min(max_features, num_features * pow(2, i - 1)), kernel_size=1, padding=0, normalization=normalization))
        blocks.append(nn.Conv2d(in_channels=min(max_features, num_features), out_channels=1, kernel_size=1, padding=0, bias=False))
        self.layers = nn.Sequential(*blocks)
        
        # initialize weights
        initialize_model(self)

    def forward(self, x):
        x = self.layers(x)
        y = x.flatten(start_dim=1).mean(dim=1)
        return y

class MultiClassifier(nn.Module):
    def __init__(self, in_channels, num_features, max_features, num_blocks, target_features, normalization=False):
        super(MultiClassifier, self).__init__()
        # parameters
        self.num_classifiers = len(target_features)

        # classifier 
        for feature in target_features:
            in_channels = min(64 * pow(2, int(feature.split('_')[0][-1]) - 1), 512)
            classifier = Classifier(in_channels=in_channels, num_features=num_features, max_features=max_features, num_blocks=num_blocks, normalization=normalization)
            self.add_module(feature ,classifier)

        # initialize weights
        initialize_model(self)

    def forward(self, f):
        y = 0.
        for key, classifier in self.named_children():
            y += classifier(f[key])
        return y / self.num_classifiers 

def d_vanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('num_blocks', 5)
    config.setdefault('max_features', 512)
    config.setdefault('normalization', False)
    
    return Vanilla(**config)

def d_sanvanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('num_blocks', 5)
    config.setdefault('max_features', 512)
    config.setdefault('gain', 1.0)
    
    config['conv_block'] = SparseCirConv2d
    
    return SANVanilla(**config)

def d_advanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('num_blocks', 5)
    config.setdefault('max_features', 512)
    config.setdefault('normalization', False)
    
    return AdaptiveVanilla(**config)

def d_sanadvanilla(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 64)
    config.setdefault('num_blocks', 5)
    config.setdefault('max_features', 512)
    config.setdefault('gain', 1.0)
    
    config['conv_block'] = SparseCirConv2d
    
    return SANAdaptiveVanilla(**config)

def d_vgghead(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 32)
    config.setdefault('max_features', 256)
    config.setdefault('num_blocks', 6)
    config.setdefault('feature_layer', 'relu3_1')
    config.setdefault('normalization', False)

    return VGGHead(**config)

def d_multivgghead(**config):
    config.setdefault('in_channels', 3)
    config.setdefault('num_features', 32)
    config.setdefault('max_features', 256)
    config.setdefault('num_blocks', 6)
    config.setdefault('target_features', ['relu2_1', 'relu3_1', 'relu4_1'])
    config.setdefault('normalization', False)

    return MultiVGGHead(**config)