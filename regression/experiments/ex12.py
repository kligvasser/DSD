import os
import sys
import argparse
import math
import pandas as pd
import glob
from scipy import stats
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('../'))
from models.modules.losses import *
from data.datasets import DatasetPieAPP

import matlab.engine

def get_arguments():
    parser = argparse.ArgumentParser(description='PieAPP-noref-brisque')
    parser.add_argument('--root', required=True, help='root directory')
    parser.add_argument('--root-labels', required=True, help='root directory')
    parser.add_argument('--max-size', default=None, type=int, help='number of images (default: None)')
    parser.add_argument('--data-type', default='test', type=str, help='data-type (default: test)')
    parser.add_argument('--bandwidth', default=1., type=float, help='bandwidth (default: 1.)')
    parser.add_argument('--margins', default=[0.1, 0.2, 0.25, 0.5], nargs='+', type=float, help='margins (default: 0.1 0.2 0.3)')
    parser.add_argument('--output', default='./outputs/PieAPP-noref.csv', help='output file-name (default: ./outputs/PieAPP-noref.csv)')
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
    
    # engine 
    eng = matlab.engine.start_matlab()

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
            a_path = data['a_path'][0]
            b_path = data['b_path'][0]
            score = float(data['score'][0].item())

            score_a, score_b = 0., 0.
            
            image_a = eng.imread(a_path)
            score_a = float(eng.brisque(image_a))

            image_b = eng.imread(b_path)
            score_b = float(eng.brisque(image_b))

            our = 1. / (1. + math.exp(args.bandwidth * (score_a - score_b)))
            binary = 1. if (score >= 0.5) == (our >= 0.5) else 0.

            d = {'ref. image': data['ref_name'][0], ' distorted image A': data['distortion_a'][0], ' distorted image B': data['distortion_b'][0], ' preference for A': score, 'our_A': score_a, 'our_B': score_b, 'our': our, 'binary': binary}
            
            for margin in args.margins:
                diff = abs(float(score) - float(our))
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