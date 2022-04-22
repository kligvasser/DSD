import os
import sys
import argparse
import torch
import pandas as pd
import glob
from scipy import stats
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('../'))
import models
from models.modules.losses import *
from data.datasets import DatasetPieAPP
from utils.core import imresize

def get_arguments():
    parser = argparse.ArgumentParser(description='PieAPP-noref')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-id', default=0, type=int, help='device id (default: 0)')
    parser.add_argument('--regressor', default='resnet_se', help='regressor architecture (default: resnet_se)')
    parser.add_argument('--reg-to-load', default='', type=str, required=True, help='regressor pkl file (default: None)')
    parser.add_argument('--root', required=True, help='root directory')
    parser.add_argument('--root-labels', required=True, help='root directory')
    parser.add_argument('--max-size', default=None, type=int, help='number of images (default: None)')
    parser.add_argument('--data-type', default='test', type=str, help='data-type (default: test)')
    parser.add_argument('--scale', default=0.5, type=float, help='super-resolution scale (default: 0.5)')
    parser.add_argument('--bandwidth', default=1., type=float, help='bandwidth (default: 1.)')
    parser.add_argument('--features', default=['relu1_2', 'relu2_1', 'relu2_2', 'relu3_1'], nargs=4, type=str, help='features level (default: relu1_2 relu2_1 relu2_2 relu3_1)')
    parser.add_argument('--weights', default=[1.0, 1.0, 1.0, 1.0], nargs=4, type=float, help='features weights (default: 1.0 1.0 1.0 1.0)')
    parser.add_argument('--wasserstein', default=False, action='store_true', help='wasserstein distance (default: false)')
    parser.add_argument('--margins', default=[0.1, 0.2, 0.25, 0.5], nargs='+', type=float, help='margins (default: 0.1 0.2 0.3)')
    parser.add_argument('--output', default='./outputs/PieAPP-noref.csv', help='output file-name (default: ./outputs/PieAPP-noref.csv)')
    args = parser.parse_args()
    return args

def get_csv_list(args):
    # csv_list = sorted(glob.glob(os.path.join(args.root_labels, '*pairwise_labels*')))[:args.max_size]
    csv_list = glob.glob(os.path.join(args.root_labels, '*pairwise_labels*'))[:args.max_size]
    return csv_list

if __name__ == "__main__":
    # arguments
    args = get_arguments()
    args.features = sorted(args.features)

    # regresor
    config = {}
    model = models.__dict__[args.regressor]
    model = model(**config).to(args.device)
    model.load_state_dict(torch.load(args.reg_to_load, map_location='cuda:{}'.format(args.device_id)))
    model.eval()

    # datasets
    csv_list = get_csv_list(args)
    dataset = DatasetPieAPP(root=args.root, csv=csv_list[0], data_type=args.data_type)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # losses
    if args.wasserstein:
        loss = SlicedWasserstein(features_to_compute=args.features).to(args.device)
    else:
        loss = Style(features_to_compute=args.features, batch_size=1).to(args.device)
    criterion = torch.nn.L1Loss(reduction='none').to(args.device)

    # data-frame
    columns = list(dataset.df.columns) + ['our_A', 'our_B', 'our', 'binary'] + [str(m).replace('.', 'p') for m in args.margins]
    df = pd.DataFrame(columns=columns)

    for csv_file in csv_list:
        # csv
        print(csv_file)

        # dataset
        dataset = DatasetPieAPP(root=args.root, csv=csv_file, data_type=args.data_type)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        # iter images
        for j, data in enumerate(loader):
            image_ref = data['image_ref'].to(args.device)
            image_a = data['image_a'].to(args.device)
            image_b = data['image_b'].to(args.device)
            score = data['score'][0].item()

            image_ref_resized = imresize(image_ref, scale=args.scale)

            score_a, score_b = 0., 0.
            with torch.no_grad():
                # preds_a = model(image_ref_resized)
                # outputs_a = loss(image_ref_resized, image_a)
                # score_all = criterion(outputs_a, preds_a).mean(dim=0)

                image_a_resized = imresize(image_a, scale=args.scale)
                preds_a = model(image_a_resized)
                outputs_a = loss(image_a_resized, image_a)
                score_all = criterion(outputs_a, preds_a).mean(dim=0)

                score_a = 0
                for i, weight in enumerate(args.weights):
                    score_a += score_all[i] * weight

                # preds_b = model(image_ref_resized)
                # outputs_b = loss(image_ref_resized, image_b)
                # score_all = criterion(outputs_b, preds_b).mean(dim=0)

                # image_b_resized = imresize(image_b, scale=args.scale)
                preds_b = model(image_a_resized)
                outputs_b = loss(image_a_resized, image_b)
                score_all = criterion(outputs_b, preds_b).mean(dim=0)

                score_b = 0
                for i, weight in enumerate(args.weights):
                    score_b += score_all[i] * weight

            our = 1. / (1. + torch.exp(args.bandwidth * (score_a - score_b)))
            binary = 1. if (float(score) >= 0.5) == (float(our.item()) >= 0.5) else 0.

            d = {'ref. image': data['ref_name'][0], ' distorted image A': data['distortion_a'][0], ' distorted image B': data['distortion_b'][0], ' preference for A': float(score), 'our_A': float(score_a.item()), 'our_B': float(score_b.item()), 'our': float(our.item()), 'binary': binary}
            
            for margin in args.margins:
                diff = abs(float(score) - float(our.item()))
                d.update({str(margin).replace('.', 'p'): 1. if diff <= margin else 0.})        

            df = df.append(d, ignore_index=True)

    df.to_csv(args.output)

    ax = df.plot.scatter(x=' preference for A', y='our')
    pcc = stats.pearsonr(df[' preference for A'].values, df['our'].values)
    d = df.describe()

    plt.savefig(args.output.replace('csv', 'png'), dpi=600) 
    
    log2save = str(args)
    log2save += '\n\nPearson correlation coefficient: {:.3f}, p-value: {:.3f}\n\n'.format(float(pcc[0]), float(pcc[1]))
    log2save += str(d)
    
    f = open(args.output.replace('csv', 'log'), 'w')
    f.write(log2save)
    f.close()

    print(log2save)