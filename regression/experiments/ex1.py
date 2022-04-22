import os
import sys
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('../'))
import models
from models.modules.losses import Style, SlicedWasserstein
from data.datasets import DatasetEval

def get_arguments():
    parser = argparse.ArgumentParser(description='self deep features loss')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-id', default=0, type=int, help='device id (default: 0)')
    parser.add_argument('--root-sr', required=True, help='root directory')
    parser.add_argument('--root-resized', required=True, help='root directory')
    parser.add_argument('--crop-size', default=80, type=int, help='cropping size (default: 80)')
    parser.add_argument('--scale', default=4, type=int, help='super-resolution scale (default: 4)')
    parser.add_argument('--max-size', default=None, type=int, help='super-resolution scale (default: None)')
    parser.add_argument('--batch-size', default=4, type=int, help='augmentations (default: 4)')
    parser.add_argument('--num-workers', default=2, type=int, help='number of workers (default: 2)')
    parser.add_argument('--features', default=['relu1_2', 'relu2_1', 'relu2_2', 'relu3_1'], nargs=4, type=str, help='features level (default: relu1_2 relu2_1 relu2_2 relu3_1)')
    parser.add_argument('--feature-weights', default=[1.0, 1.0, 1.0, 1.0], nargs=4, type=float, help='feature weights (default: 1.0, 1.0, 1.0, 1.0)')
    parser.add_argument('--regressor', default='resnet', help='regressor architecture (default: resnet)')
    parser.add_argument('--reg-to-load', default='', type=str, required=True, help='regressor pkl file (default: None)')
    parser.add_argument('--wasserstein', default=False, action='store_true', help='wasserstein distance (default: false)')
    parser.add_argument('--output', default='./outputs/self-dist.csv', help='output file-name (default: ./outputs/self-dist.csv)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # arguments
    args = get_arguments()
    args.features = sorted(args.features)

    # datasets
    dataset = DatasetEval(root_sr=args.root_sr, root_resized=args.root_resized, scale=args.scale, crop_size=args.crop_size, batch_size=args.batch_size, max_size=args.max_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    if args.wasserstein:
        loss = SlicedWasserstein(features_to_compute=args.features, batch_size=args.batch_size).to(args.device)
    else:
        loss = Style(features_to_compute=args.features, batch_size=args.batch_size).to(args.device)
    
    criterion = torch.nn.L1Loss(reduction='none').to(args.device)

    # config = {'in_channels': 3, 'num_features': 32, 'num_blocks': 5, 'num_classes': len(args.features)}
    config = {}
    model = models.__dict__[args.regressor]
    model = model(**config).to(args.device)
    model.load_state_dict(torch.load(args.reg_to_load, map_location='cuda:{}'.format(args.device_id)))
    model.eval()

    columns = ['image'] + args.features + ['all']
    df = pd.DataFrame(columns=columns)

    # iter images
    for j, data in enumerate(loader):
        resized = data['resized'].to(args.device)
        sr = data['sr'].to(args.device)
        path = data['path'][0]

        with torch.no_grad():
            preds = model(resized)
            outputs = loss(resized, sr)
            dists = criterion(outputs, preds).mean(dim=0)
        
        summed = 0
        for i in range(len(dists)):
            summed += dists[i] * args.feature_weights[i]

        image_name = os.path.basename(path).split('.')[0]
        d = {'image': image_name}
        for i in range(len(dists)):
            d.update({args.features[i]: float(dists[i].item())})
        d.update({'all': float(summed.item())})
        df = df.append(d, ignore_index=True) 

        # print('Image #{}/{} : {}'.format(j + 1, len(loader.dataset) // args.batch_size, image_name))

    df.to_csv(args.output)
    f = open(args.output.replace('csv', 'log'), 'w')
    d = df.describe()
    f.write(str(d))
    f.close()

    print(d)