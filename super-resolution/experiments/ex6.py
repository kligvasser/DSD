import os
import sys
import argparse
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath('../'))
from models.vgg import VGGFeaturesExtractor
from utils.core import imresize
from torchvision.utils import make_grid

def get_arguments():
    parser = argparse.ArgumentParser(description='vgg-images')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--path', required=True, help='')
    parser.add_argument('--scale', default=0.5, type=float, help='')
    parser.add_argument('--feature', default='relu3_1', help='')
    parser.add_argument('--output', default='./outputs/', help='')
    args = parser.parse_args()
    return args

def save_images_grid(tensor, path, nrow=32, padding=0, normalize=True, scale_each=True, pad_value=0):
    grid = make_grid(tensor.cpu(), nrow=nrow, padding=padding, normalize=normalize, scale_each=scale_each, pad_value=pad_value)
    ndarr = grid.squeeze().mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    image = Image.fromarray(ndarr)
    image.save(path)

if __name__ == "__main__":
    # arguments
    args = get_arguments()

    # read image
    to_tensor = transforms.ToTensor()
    image = Image.open(args.path).convert('RGB')
    image = to_tensor(image).unsqueeze(dim=0).to(args.device)
    image_sc = imresize(image.clone(), scale=args.scale)

    # vgg
    vgg = VGGFeaturesExtractor(feature_layer=args.feature).eval().to(args.device)

    # run within vgg
    with torch.no_grad():
        fea = vgg(image)
        fea_sc = vgg(image_sc)

    fea = fea.view(fea.size(1), 1, fea.size(2), fea.size(3))
    fea_sc = fea_sc.view(fea_sc.size(1), 1, fea_sc.size(2), fea_sc.size(3))

    save_images_grid(fea, os.path.join(args.output, 'output.png'))
    save_images_grid(fea_sc, os.path.join(args.output, 'output_sc.png'))
    





    