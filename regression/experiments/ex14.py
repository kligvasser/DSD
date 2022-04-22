import os
import sys
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('../'))
from models.vgg import MultiVGGFeaturesExtractor
from data.folder import ImageFolder
from utils.core import imresize

class DSD(torch.nn.Module):
    def __init__(self, features_to_compute=['relu2_1', 'relu2_2', 'relu3_1'], scales=[0.5], criterion=torch.nn.L1Loss(reduction='none')):
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

                loss += self.criterion(inputs_gram, targets_gram).mean(dim=(1,2))

        return loss

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

def get_arguments():
    parser = argparse.ArgumentParser(description='imagenet-dsd')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-id', default=0, type=int, help='device id (default: 0)')
    parser.add_argument('--root', required=True, help='root directory')
    parser.add_argument('--resize', default=256, type=int, help='cropping size (default: 256)')
    parser.add_argument('--crop-size', default=224, type=int, help='cropping size (default: 224)')
    parser.add_argument('--scale', default=0.5, type=float, help='super-resolution scale (default: 0.5)')
    parser.add_argument('--batch-size', default=64, type=int, help='augmentations (default: 64)')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers (default: 4)')
    parser.add_argument('--features', default=['relu3_1'], nargs='+', type=str, help='features level (default: relu3_1)')
    parser.add_argument('--output', default='./outputs/imagenet-dsd.csv', help='output file-name (default: ./outputs/imagenet-dsd.csv)')
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
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ImageFolder(root=args.root, transform=data_transforms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # criterion
    criterion = DSD(features_to_compute=args.features, scales=[args.scale]).to(args.device)

    # dataframe
    columns = ['image', 'class'] + [args.features[0]]
    df = pd.DataFrame(columns=columns)

    # predic data 
    for i, data in enumerate(loader):
        images = data['sample'].to(args.device)
        targets = data['target']
        paths = data['path']

        # extract features
        with torch.no_grad():
            # compute dsd
            score = criterion(images)

        # write tensors
        for j in range(len(score)):
            d = {'image': paths[j], 'class': float(targets[j]), args.features[0]: float(score[j].cpu())}
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