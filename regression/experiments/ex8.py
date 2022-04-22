import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('../'))
from models.vgg import VGGFeaturesExtractor
from data.folder import ImageFolder
from utils.core import imresize

class ExtractPatchesStyleGram(nn.Module):
    def __init__(self, kernel_size=(7, 7), reduce_mean=True):
        super(ExtractPatchesStyleGram, self).__init__()
        self.kernel_size = kernel_size
        self.reduce_mean = reduce_mean

    def forward(self, x):
        x_pch = self._image_to_patches(x, kernel_size=self.kernel_size)
        x_gram = self._gram_matrix(x_pch).flatten(start_dim=1)

        return x_gram

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

    def _image_to_patches(self, x, kernel_size):
        x = F.unfold(x, kernel_size=kernel_size, padding=0).unsqueeze(dim=-1)
        if self.reduce_mean:
            x = x - x.mean(dim=1, keepdim=True)
        return x

class ExtractVGGStyleGram(nn.Module):
    def __init__(self, feature_layer='relu3_1'):
        super(ExtractVGGStyleGram, self).__init__()
        self.features_extractor = VGGFeaturesExtractor(feature_layer=feature_layer, use_input_norm=False).eval()

    def forward(self, x):
        x_fea = self.features_extractor(x)
        x_gram = self._gram_matrix(x_fea).flatten(start_dim=1)
        
        return x_gram

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

class ExtractVGGRecStyleGram(nn.Module):
    def __init__(self, feature_layer='relu3_1', scale=0.5):
        super(ExtractVGGRecStyleGram, self).__init__()
        self.scale = scale
        self.features_extractor = VGGFeaturesExtractor(feature_layer=feature_layer, use_input_norm=False).eval()

    def forward(self, x):
        x_resized = imresize(x, scale=self.scale)

        x_fea = self.features_extractor(x)
        x_resized_fea = self.features_extractor(x_resized)

        x_gram = self._gram_matrix(x_fea).flatten(start_dim=1)
        x_resized_gram = self._gram_matrix(x_resized_fea).flatten(start_dim=1)

        d = (x_gram - x_resized_gram)
        
        return d

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

def get_arguments():
    parser = argparse.ArgumentParser(description='features extoraction')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-id', default=0, type=int, help='device id (default: 0)')
    parser.add_argument('--root', required=True, help='root directory')
    parser.add_argument('--resize', default=256, type=int, help='cropping size (default: 256)')
    parser.add_argument('--crop-size', default=224, type=int, help='cropping size (default: 224)')
    parser.add_argument('--scale', default=0.5, type=float, help='super-resolution scale (default: 0.5)')
    parser.add_argument('--kernel-size', default=7, type=int, help='augmentations (default: 7)')
    parser.add_argument('--batch-size', default=100, type=int, help='augmentations (default: 100)')
    parser.add_argument('--num-workers', default=2, type=int, help='number of workers (default: 2)')
    parser.add_argument('--feature-layer', default='relu2_2', type=str, help='features level (default: relu2_2)')
    parser.add_argument('--output', default='./outputs/features-extraction.pkl', help='output file-name (default: ./outputs/features-extraction.pkl)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # arguments
    args = get_arguments()
    print(args)

    # datasets
    data_transforms = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root=args.root, transform=data_transforms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # extractors
    extract_pch = ExtractPatchesStyleGram(kernel_size=(args.kernel_size, args.kernel_size), reduce_mean=False).to(args.device)
    extract_sty = ExtractVGGStyleGram(feature_layer=args.feature_layer).to(args.device)
    extract_rsty = ExtractVGGRecStyleGram(feature_layer=args.feature_layer, scale=args.scale).to(args.device)

    # dataframe
    columns = ['image', 'class', 'pch_gram', 'sty_gram', 'rsty_gram']
    df = pd.DataFrame(columns=columns)

    # iter images
    # for i, data in enumerate(loader):
    loader = iter(loader)
    for i in range(100):
        data = next(loader)
        images = data['sample'].to(args.device)
        targets = data['target']
        paths = data['path']

        # extract features
        with torch.no_grad():
            pch_gram = extract_pch(images)
            sty_gram = extract_sty(images)
            rsty_gram = extract_rsty(images)
        
        # write tensors
        for j in range(args.batch_size):
            d = {'image': paths[j], 'class': float(targets[j]), 'pch_gram': pch_gram[j].cpu().numpy(), 'sty_gram': sty_gram[j].cpu().numpy(), 'rsty_gram': rsty_gram[j].cpu().numpy()}
            df = df.append(d, ignore_index=True)
        print('{}/{}'.format((i + 1) * args.batch_size, len(dataset)))

    df.to_pickle(args.output, compression=None)