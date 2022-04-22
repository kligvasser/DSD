import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import glob
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath('../'))
from models.vgg import MultiVGGFeaturesExtractor
from utils.core import imresize

class Dataset(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', transforms=None):
        self.root = root
        self.transforms = transforms
        self.paths = sorted(glob.glob(os.path.join(args.root, '*.png')))

    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert('RGB')

        if self.transforms:
            image = self.transforms(image)

        return {'image': image, 'path': path}

    def __len__(self):
        return len(self.paths)


class DSD(torch.nn.Module):
    def __init__(self, features_to_compute=['relu3_1'], scales=[0.5], criterion=torch.nn.L1Loss()):
        super(DSD, self).__init__()
        self.scales = scales
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs):
        loss = 0
        inputs_fea = self.features_extractor(inputs)
        for scale in self.scales:
            targets = imresize(inputs, scale=scale)

            with torch.no_grad():
                targets_fea = self.features_extractor(targets)

            for key in inputs_fea.keys():
                inputs_gram = self._gram_matrix(inputs_fea[key])
                with torch.no_grad():
                    targets_gram = self._gram_matrix(targets_fea[key]).detach()

                loss += self.criterion(inputs_gram, targets_gram)

        return loss

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

def get_arguments():
    parser = argparse.ArgumentParser(description='imagenet-class-hist')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--root', required=True, help='root directory')
    parser.add_argument('--resize', default=256, type=int, help='cropping size (default: 256)')
    parser.add_argument('--crop-size', default=224, type=int, help='cropping size (default: 224)')
    parser.add_argument('--style-scales', default=[0.5], type=float, nargs='+', help='style scales (e.g 0.5 0.75)')
    parser.add_argument('--features', default=['relu3_1'], nargs='+', type=str, help='features level (default: relu3_1)')
    parser.add_argument('--output', default='./outputs/hist-values.csv', help='output file-name (default: ./outputs/hist-values.csv)')
    args = parser.parse_args()
    return args

def get_images_list(args):
    images_paths = sorted(glob.glob(os.path.join(args.root, '*.JPEG')))
    return images_paths

def save_histogram(args, data):
    hist = np.histogram(data, bins=50, range=(0.0005,0.05))
    dict = {'values' : hist[0], 'values_norm' : hist[0] / len(data), 'bins': hist[1]}
    df = pd.DataFrame.from_dict(dict, orient='index').transpose()

    df.to_csv(args.output.replace('values', 'counts'), index=None, header=True)

if __name__ == "__main__":
    # arguments
    args = get_arguments()

    # datasets
    images_paths = get_images_list(args)
    data_transforms = transforms.Compose([
        transforms.Resize(args.resize),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor()
    ])

    # dsd
    dsd = DSD(features_to_compute=args.features, scales=args.style_scales).to(args.device)
    data = np.zeros(len(images_paths))
    
    # data-frame
    df = pd.DataFrame(columns=['image', 'dsd'])

    for i, path in enumerate(images_paths):
        # path
        print(path)

        # open images
        image = Image.open(path).convert('RGB')
        image = data_transforms(image).to(args.device)

        # compute dsd
        score = dsd(image)

        d = {'image': path, 'dsd': float(score.item())}
        df = df.append(d, ignore_index=True)

        data[i] = score.cpu().numpy()

    save_histogram(args, data)
    df.to_csv(args.output)
    print(df.describe())