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
from data.datasets import DatasetPieAPP

def get_arguments():
    parser = argparse.ArgumentParser(description='PieAPP-ref')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-id', default=0, type=int, help='device id (default: 0)')
    parser.add_argument('--root', required=True, help='root directory')
    parser.add_argument('--root-labels', required=True, help='root directory')
    parser.add_argument('--max-size', default=None, type=int, help='number of images (default: None)')
    parser.add_argument('--data-type', default='test', type=str, help='data-type (default: test)')
    parser.add_argument('--scales', default=[0.5], nargs='+', type=float, help='super-resolution scale (default: 0.5)')
    parser.add_argument('--bandwidth', default=1., type=float, help='bandwidth (default: 1.)')
    parser.add_argument('--features', default=['relu1_2', 'relu2_1', 'relu3_1'], nargs='+', type=str, help='features level (default: relu1_2 relu2_1 relu3_1)')
    parser.add_argument('--perceptual-weight', default=0, type=float, help='perceptual-weight (default: 0)')
    parser.add_argument('--contextual-weight', default=0, type=float, help='contextual-weight (default: 0)')
    parser.add_argument('--consistency-weight', default=0, type=float, help='consistency-weight (default: 0)')
    parser.add_argument('--l1-weight', default=0, type=float, help='consistency-weight (default: 0)')
    parser.add_argument('--l2-weight', default=0, type=float, help='consistency-weight (default: 0)')
    parser.add_argument('--style-weight', default=0, type=float, help='style-weight (default: 0)')
    parser.add_argument('--multi-style-weight', default=0, type=float, help='style-weight (default: 0)')
    parser.add_argument('--recurrent-style-weight', default=0, type=float, help='recurrent-style-weight (default: 0)')
    parser.add_argument('--wasserstein-weight', default=0, type=float, help='recurrent-wasserstein-weight (default: 0)')
    parser.add_argument('--recurrent-wasserstein-weight', default=0, type=float, help='recurrent-wasserstein-weight (default: 0)')
    parser.add_argument('--recurrent-patch-style-weight', default=0, type=float, help='recurrent-wasserstein-weight (default: 0)')
    parser.add_argument('--margins', default=[0.1, 0.2, 0.25, 0.5], nargs='+', type=float, help='margins (default: 0.1 0.2 0.3)')
    parser.add_argument('--output', default='./outputs/PieAPP-ref.csv', help='output file-name (default: ./outputs/PieAPP-ref.csv)')
    args = parser.parse_args()
    return args

def get_csv_list(args):
    csv_list = sorted(glob.glob(os.path.join(args.root_labels, '*pairwise_labels*')))[:args.max_size]
    return csv_list

if __name__ == "__main__":
    # arguments
    args = get_arguments()
    args.features = sorted(args.features)

    # datasets
    csv_list = get_csv_list(args)
    dataset = DatasetPieAPP(root=args.root, csv=csv_list[0], data_type=args.data_type)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # losses
    if args.perceptual_weight:
        perceptual = PerceptualLoss(features_to_compute=args.features, criterion=torch.nn.L1Loss()).to(args.device)
    if args.contextual_weight:
        contextual = ContextualLoss().to(args.device)
    if args.consistency_weight:
        consistency = ConsistencyLoss(scale=args.scales[0], criterion=torch.nn.L1Loss()).to(args.device)
    if args.l1_weight:
        rec_l1 = torch.nn.L1Loss().to(args.device)
    if args.l2_weight:
        rec_l2 = torch.nn.MSELoss().to(args.device)
    if args.style_weight:
        style = Style(features_to_compute=args.features, batch_size=1).to(args.device)
    if args.recurrent_style_weight:
        rec_style = RecurrentStyleLoss(features_to_compute=args.features, scales=args.scales).to(args.device)
    if args.multi_style_weight:
        multi_style = MultiStyleLoss(features_to_compute=args.features, scales=args.scales).to(args.device)
    if args.recurrent_patch_style_weight:
        ptc_style = RecurrentPatchesStyleLoss(scales=args.scales).to(args.device)
    if args.wasserstein_weight:
        wasserstein = SlicedWasserstein(features_to_compute=args.features).to(args.device)
    if args.recurrent_wasserstein_weight:
        rwasserstein = RecurrentWassersteinLoss(features_to_compute=args.features, scales=args.scales).to(args.device)

    # data-frame
    columns = list(dataset.df.columns) + ['our', 'binary'] + [str(m).replace('.', 'p') for m in args.margins]
    df = pd.DataFrame(columns=columns)

    for csv_file in csv_list:
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

            score_a, score_b = 0., 0.
            with torch.no_grad():
                if args.perceptual_weight:
                    score_a += perceptual(image_a, image_ref) * args.perceptual_weight
                if args.contextual_weight:
                    score_a += contextual(image_a, image_ref) * args.contextual_weight
                if args.consistency_weight:
                    score_a += consistency(image_a, image_ref) * args.consistency_weight
                if args.l1_weight:
                    score_a += rec_l1(image_a, image_ref) * args.l1_weight
                if args.l2_weight:
                    score_a += rec_l1(image_a, image_ref) * args.l2_weight
                if args.style_weight:
                    score_a += style(image_a, image_ref) * args.style_weight
                if args.multi_style_weight:
                    score_a += multi_style(image_a, image_ref) * args.multi_style_weight
                if args.recurrent_style_weight:
                    score_a += rec_style(image_a, image_ref) * args.recurrent_style_weight
                if args.recurrent_patch_style_weight:
                    score_a += ptc_style(image_a, image_ref) * args.recurrent_patch_style_weight
                if args.wasserstein_weight:
                    score_a += wasserstein(image_a, image_ref) * args.wasserstein_weight
                if args.recurrent_wasserstein_weight:
                    score_a += rwasserstein(image_a, image_ref) * args.recurrent_wasserstein_weight

                if args.perceptual_weight:
                    score_b += perceptual(image_b, image_ref) * args.perceptual_weight
                if args.contextual_weight:
                    score_b += contextual(image_b, image_ref) * args.contextual_weight
                if args.consistency_weight:
                    score_b += consistency(image_b, image_ref) * args.consistency_weight
                if args.l1_weight:
                    score_b += rec_l1(image_b, image_ref) * args.l1_weight
                if args.l2_weight:
                    score_b += rec_l1(image_b, image_ref) * args.l2_weight
                if args.style_weight:
                    score_b += style(image_b, image_ref) * args.style_weight
                if args.multi_style_weight:
                    score_b += multi_style(image_b, image_ref) * args.multi_style_weight
                if args.recurrent_style_weight:
                    score_b += rec_style(image_b, image_ref) * args.recurrent_style_weight
                if args.recurrent_patch_style_weight:
                    score_b += ptc_style(image_b, image_ref) * args.recurrent_patch_style_weight
                if args.wasserstein_weight:
                    score_b += wasserstein(image_b, image_ref) * args.wasserstein_weight
                if args.recurrent_wasserstein_weight:
                    score_b += rwasserstein(image_b, image_ref) * args.recurrent_wasserstein_weight

            our = 1. / (1. + torch.exp(args.bandwidth * (score_a - score_b)))
            binary = 1. if (float(score) >= 0.5) == (float(our.item()) >= 0.5) else 0.
            d = {'ref. image': data['ref_name'][0], ' distorted image A': data['distortion_a'][0], ' distorted image B': data['distortion_b'][0], ' preference for A': float(score), 'our_A': float(score_a.item()), 'our_B': float(score_b.item()), 'our': float(our.item()), 'binary': binary}
            
            for margin in args.margins:
                diff = abs(float(score) - float(our.item()))
                d.update({str(margin).replace('.', 'p'): 1. if diff <= margin else 0})
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