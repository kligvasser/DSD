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
from models.modules.losses import *
from data.datasets import DatasetPIPAL

def get_arguments():
    parser = argparse.ArgumentParser(description='PIPAL-ref')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-id', default=0, type=int, help='device id (default: 0)')
    parser.add_argument('--root', required=True, help='root directory')
    parser.add_argument('--max-size', default=None, type=int, help='number of images (default: None)')
    parser.add_argument('--data-type', default='test', type=str, help='data-type (default: test)')
    parser.add_argument('--scales', default=[0.5], nargs='+', type=float, help='super-resolution scale (default: 0.5)')
    parser.add_argument('--features', default=['relu1_2', 'relu2_1', 'relu3_1'], nargs='+', type=str, help='features level (default: relu1_2 relu2_1 relu3_1)')
    parser.add_argument('--perceptual-weight', default=0, type=float, help='perceptual-weight (default: 0)')
    parser.add_argument('--contextual-weight', default=0, type=float, help='contextual-weight (default: 0)')
    parser.add_argument('--consistency-weight', default=0, type=float, help='consistency-weight (default: 0)')
    parser.add_argument('--reconstruction-weight', default=0, type=float, help='consistency-weight (default: 0)')
    parser.add_argument('--recurrent-style-weight', default=0, type=float, help='recurrent-style-weight (default: 0)')
    parser.add_argument('--wasserstein-weight', default=0, type=float, help='recurrent-wasserstein-weight (default: 0)')
    parser.add_argument('--recurrent-wasserstein-weight', default=0, type=float, help='recurrent-wasserstein-weight (default: 0)')
    parser.add_argument('--recurrent-patch-style-weight', default=0, type=float, help='recurrent-wasserstein-weight (default: 0)')
    parser.add_argument('--margins', default=[0.1, 0.2, 0.25, 0.5], nargs='+', type=float, help='margins (default: 0.1 0.2 0.3)')
    parser.add_argument('--output', default='./outputs/PIPAL-ref.csv', help='output file-name (default: ./outputs/PIPAL-ref.csv)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # arguments
    args = get_arguments()
    args.features = sorted(args.features)

    # datasets
    dataset = DatasetPIPAL(root=args.root, max_size=args.max_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # losses
    if args.perceptual_weight:
        perceptual = PerceptualLoss(features_to_compute=['conv5_4'], criterion=torch.nn.L1Loss()).to(args.device)
    if args.contextual_weight:
        contextual = ContextualLoss().to(args.device)
    if args.consistency_weight:
        consistency = ConsistencyLoss(scale=args.scales[0], criterion=torch.nn.L1Loss()).to(args.device)
    if args.reconstruction_weight:
        reconstruction = torch.nn.L1Loss().to(args.device)
        # reconstruction = torch.nn.MSELoss().to(args.device)
    if args.recurrent_style_weight:
        style = RecurrentStyleLoss(features_to_compute=args.features, scales=args.scales).to(args.device)
    if args.recurrent_patch_style_weight:
        ptc_style = RecurrentPatchesStyleLoss(scales=args.scales).to(args.device)
    if args.wasserstein_weight:
        wasserstein = SlicedWasserstein(features_to_compute=args.features).to(args.device)
    if args.recurrent_wasserstein_weight:
        rwasserstein = RecurrentWassersteinLoss(features_to_compute=args.features, scales=args.scales).to(args.device)

    # data-frame
    columns = ['ref', 'dist', 'label', 'score']
    df = pd.DataFrame(columns=columns)

    # iter images
    for i, data in enumerate(loader):
        image_ref = data['ref'].to(args.device)
        image_dist = data['dist'].to(args.device)
        label = data['label'][0].item()

        score = 0.
        with torch.no_grad():
            if args.perceptual_weight:
                score += perceptual(image_dist, image_ref) * args.perceptual_weight
            if args.contextual_weight:
                score += contextual(image_dist, image_ref) * args.contextual_weight
            if args.consistency_weight:
                score += consistency(image_dist, image_ref) * args.consistency_weight
            if args.reconstruction_weight:
                score += reconstruction(image_dist, image_ref) * args.reconstruction_weight
            if args.recurrent_style_weight:
                score += style(image_dist, image_ref) * args.recurrent_style_weight
            if args.recurrent_patch_style_weight:
                score += ptc_style(image_dist, image_ref) * args.recurrent_patch_style_weight
            if args.wasserstein_weight:
                score += wasserstein(image_dist, image_ref) * args.wasserstein_weight
            if args.recurrent_wasserstein_weight:
                score += rwasserstein(image_dist, image_ref) * args.recurrent_wasserstein_weight

            d = {'ref': data['ref_path'][0], 'dist': data['dist_path'][0], 'label': label, 'score': -float(score)}
            df = df.append(d, ignore_index=True)

    df.to_csv(args.output)

    ax = df.plot.scatter(x='label', y='score')
    pcc = stats.pearsonr(df['label'].values, df['score'].values)
    d = df.describe()

    plt.savefig(args.output.replace('csv', 'png'), dpi=600) 
    
    log2save = str(args)
    log2save += '\n\nPearson correlation coefficient: {:.3f}, p-value: {:.3f}\n\n'.format(float(pcc[0]), float(pcc[1]))
    log2save += str(d)
    
    f = open(args.output.replace('csv', 'log'), 'w')
    f.write(log2save)
    f.close()

    print(log2save)
