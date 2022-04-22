import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
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

class ExtractPatchesRecStyleGram(nn.Module):
    def __init__(self, kernel_size=(7, 7), reduce_mean=True, scale=0.5):
        super(ExtractPatchesRecStyleGram, self).__init__()
        self.kernel_size = kernel_size
        self.reduce_mean = reduce_mean
        self.scale = scale

    def forward(self, x):
        x_resized = imresize(x, scale=self.scale)
        
        x_pch = self._image_to_patches(x, kernel_size=self.kernel_size)
        x_resized_pch = self._image_to_patches(x_resized, kernel_size=self.kernel_size)

        x_gram = self._gram_matrix(x_pch).flatten(start_dim=1)
        x_resized_gram = self._gram_matrix(x_resized_pch).flatten(start_dim=1)

        d = (x_gram - x_resized_gram)

        return d

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

class ExtractVGGMultiStyleGram(nn.Module):
    def __init__(self, feature_layer='relu3_1', scale=0.5):
        super(ExtractVGGMultiStyleGram, self).__init__()
        self.scale = scale
        self.features_extractor = VGGFeaturesExtractor(feature_layer=feature_layer, use_input_norm=False).eval()

    def forward(self, x):
        x_resized = imresize(x, scale=self.scale)

        x_fea = self.features_extractor(x)
        x_resized_fea = self.features_extractor(x_resized)

        x_gram = self._gram_matrix(x_fea).flatten(start_dim=1)
        x_resized_gram = self._gram_matrix(x_resized_fea).flatten(start_dim=1)

        c = torch.cat([x_gram, x_resized_gram], dim=1)
        
        return c

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

def get_arguments():
    parser = argparse.ArgumentParser(description='nn-search')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-id', default=0, type=int, help='device id (default: 0)')
    parser.add_argument('--root', required=True, help='root directory')
    parser.add_argument('--resize', default=256, type=int, help='cropping size (default: 256)')
    parser.add_argument('--crop-size', default=224, type=int, help='cropping size (default: 224)')
    parser.add_argument('--gram-type', default='pch', type=str, choices=['pch', 'rpch', 'sty', 'msty', 'rsty'], help='gram-type pch/rpch/sty/msty/rsty(default: pch)')
    parser.add_argument('--scale', default=0.5, type=float, help='super-resolution scale (default: 0.5)')
    parser.add_argument('--kernel-size', default=7, type=int, help='augmentations (default: 7)')
    parser.add_argument('--batch-size', default=100, type=int, help='augmentations (default: 100)')
    parser.add_argument('--num-batches', default=50, type=int, help='augmentations (default: 50)')
    parser.add_argument('--num-workers', default=2, type=int, help='number of workers (default: 2)')
    parser.add_argument('--feature-layer', default='relu3_1', type=str, help='features level (default: relu3_1)')
    parser.add_argument('--num-samples', default=50, type=int, help='number of samples (default: 50)')
    parser.add_argument('--num-nn', default=5, type=int, help='number of samples (default: 5)')
    parser.add_argument('--output', default='./outputs/nn-search.csv', help='output file-name (default: ./outputs/nn-search.csv)')
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
    if args.gram_type == 'pch':
        extractor = ExtractPatchesStyleGram(kernel_size=(args.kernel_size, args.kernel_size), reduce_mean=False).to(args.device)
    elif args.gram_type == 'rpch':
        extractor = ExtractPatchesRecStyleGram(kernel_size=(args.kernel_size, args.kernel_size), reduce_mean=False, scale=args.scale).to(args.device)
    elif args.gram_type == 'sty':
        extractor = ExtractVGGStyleGram(feature_layer=args.feature_layer).to(args.device)
    elif args.gram_type == 'rsty':
        extractor = ExtractVGGRecStyleGram(feature_layer=args.feature_layer, scale=args.scale).to(args.device)
    else:
        extractor = ExtractVGGMultiStyleGram(feature_layer=args.feature_layer, scale=args.scale).to(args.device)

    # dataframe
    columns = ['image', 'class', args.gram_type]
    database = pd.DataFrame(columns=columns)
    
    columns = ['image'] + ['{}_nn'.format(i) for i in range(args.num_nn)] + ['{}_dist'.format(i) for i in range(args.num_nn)]
    df = pd.DataFrame(columns=columns)

    # build dataset
    loader = iter(loader)
    for i in range(args.num_batches):
        data = next(loader)
        images = data['sample'].to(args.device)
        targets = data['target']
        paths = data['path']

        # extract features
        with torch.no_grad():
            gram = extractor(images)

        # write tensors
        for j in range(args.batch_size):
            d = {'image': paths[j], 'class': float(targets[j]), args.gram_type: gram[j].cpu().numpy()}
            database = database.append(d, ignore_index=True)
        print('{}/{}'.format((i + 1) * args.batch_size, len(dataset)))

    
    # sample
    sampled = database.sample(n=args.num_samples, random_state=1)

    # run-nnseach
    neigh = NearestNeighbors(n_neighbors=args.num_nn, p=1, n_jobs=args.num_workers)
    neigh.fit(np.stack(database[args.gram_type].values, axis=0).reshape(len(database), -1))
    dist, ind = neigh.kneighbors(np.stack(sampled[args.gram_type].values, axis=0), n_neighbors=args.num_nn, return_distance=True)

    # save results
    for i in range(args.num_samples):
        d = {'image': sampled['image'].iloc[i]}
        for j in range(args.num_nn):
            d.update({'{}_nn'.format(j): database['image'].iloc[ind[i, j]], '{}_dist'.format(j): dist[i, j]})
        df = df.append(d, ignore_index=True)
    
    # write csv
    df.to_csv(args.output)

    # save log
    f = open(args.output.replace('csv', 'log'), 'w')
    f.write(str(args))
    f.close()