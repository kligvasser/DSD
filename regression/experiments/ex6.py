import os
import sys
import argparse
import torch
import pandas as pd
import glob
from scipy import stats
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import lpips

sys.path.append(os.path.abspath('../'))
from data.datasets import DatasetPieAPP

def get_arguments():
    parser = argparse.ArgumentParser(description='PieAPP-ref-lpips')
    parser.add_argument('--root', required=True, help='root directory')
    parser.add_argument('--root-labels', required=True, help='root directory')
    parser.add_argument('--max-size', default=None, type=int, help='number of images (default: None)')
    parser.add_argument('--data-type', default='test', type=str, help='data-type (default: test)')
    parser.add_argument('--scales', default=[0.5], nargs='+', type=float, help='super-resolution scale (default: 0.5)')
    parser.add_argument('--bandwidth', default=1., type=float, help='bandwidth (default: 1.)')
    parser.add_argument('--margins', default=[0.1, 0.2, 0.25, 0.5], nargs='+', type=float, help='margins (default: 0.1 0.2 0.3)')
    parser.add_argument('--output', default='./outputs/PieAPP-ref.csv', help='output file-name (default: ./outputs/PieAPP-ref.csv)')
    args = parser.parse_args()
    return args

def get_csv_list(args):
    csv_list = glob.glob(os.path.join(args.root_labels, '*pairwise_labels*'))[:args.max_size]
    return csv_list

if __name__ == "__main__":
    # arguments
    args = get_arguments()

    # datasets
    csv_list = get_csv_list(args)
    dataset = DatasetPieAPP(root=args.root, csv=csv_list[0], data_type=args.data_type)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # loss
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

    # data-frame
    columns = list(dataset.df.columns) + ['our_A', 'our_B', 'our', 'binary'] + [str(m).replace('.', 'p') for m in args.margins]
    df = pd.DataFrame(columns=columns)

    for csv_file in csv_list:
        print(csv_file)

        # dataset
        dataset = DatasetPieAPP(root=args.root, csv=csv_file, data_type=args.data_type)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        # iter images
        for j, data in enumerate(loader):
            image_ref = data['image_ref']
            image_a = data['image_a']
            image_b = data['image_b']
            score = data['score'][0].item()

            image_ref = 2 * image_ref - 1
            image_a = 2 * image_a - 1
            image_b = 2 * image_b - 1

            score_a, score_b = 0., 0.

            score_a = loss_fn_alex(image_a, image_ref)
            score_b = loss_fn_alex(image_b, image_ref)

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