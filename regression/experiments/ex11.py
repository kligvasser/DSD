import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('../'))
from data.folder import ImageFolder
from utils.core import imresize

def get_arguments():
    parser = argparse.ArgumentParser(description='nn-search')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-id', default=0, type=int, help='device id (default: 0)')
    parser.add_argument('--root', required=True, help='root directory')
    parser.add_argument('--resize', default=512, type=int, help='cropping size (default: 512)')
    parser.add_argument('--crop-size', default=448, type=int, help='cropping size (default: 224)')
    parser.add_argument('--scale', default=0.5, type=float, help='super-resolution scale (default: 0.5)')
    parser.add_argument('--batch-size', default=64, type=int, help='augmentations (default: 64)')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers (default: 4)')
    parser.add_argument('--output', default='./outputs/imagenet-preds.csv', help='output file-name (default: ./outputs/imagenet-preds.csv)')
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

    # model
    model = torchvision.models.vgg19(pretrained=True)
    model = model.to(args.device)
    model.eval()

    # dataframe
    columns = ['image', 'class', 'pred_1x', 'pred_2x', 'acc_1x', 'acc_2x']
    df = pd.DataFrame(columns=columns)

    # predic data 
    for i, data in enumerate(loader):
        images = data['sample'].to(args.device)
        targets = data['target']
        paths = data['path']

        # extract features
        with torch.no_grad():
            images_resized = imresize(images, scale=args.scale)
            preds_1x = torch.softmax(model(images_resized), dim=1)
            preds_2x = torch.softmax(model(images), dim=1)
            _, predicted_1x = torch.max(preds_1x, dim=1)
            _, predicted_2x = torch.max(preds_2x, dim=1)
            acc_1x = (predicted_1x.cpu() == targets).float()
            acc_2x = (predicted_2x.cpu() == targets).float()
        
        # write tensors
        for j in range(len(preds_1x)):
            d = {'image': paths[j], 'class': float(targets[j]), 'pred_1x': float(predicted_1x[j].cpu()), 'pred_2x': float(predicted_2x[j].cpu()), 'acc_1x': float(acc_1x[j]), 'acc_2x': float(acc_2x[j])}
            df = df.append(d, ignore_index=True)
        print('{}/{}'.format((i + 1) * args.batch_size, len(dataset)))
    
    # describe 
    d = df.describe()

    print(d)

    # write csv
    df.to_csv(args.output)

    # save log
    f = open(args.output.replace('csv', 'log'), 'w')
    f.write(str(args))
    f.close()