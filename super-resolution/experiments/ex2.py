import os
import sys
import argparse
import torch
import pandas as pd
import glob
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('../'))
import models
from models.vgg import VGGFeaturesExtractor
from data import get_loaders
from utils.core import imresize

def get_arguments():
    parser = argparse.ArgumentParser(description='norm-check')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--gen-model', default='g_srgan', help='generator architecture (default: srgan)')
    parser.add_argument('--gen-to-load', default='', help='')
    parser.add_argument('--root', required=True, help='root directory')
    parser.add_argument('--scale', default=4, type=float, help='super-resolution scale (default: 4)')
    parser.add_argument('--crop-size', default=64, type=int, help='low resolution cropping size (default: 64)')
    parser.add_argument('--max-size', default=None, type=int, help='validation set max-size (default: None)')
    parser.add_argument('--num-workers', default=1, type=int, help='number of workers (default: 1)')
    parser.add_argument('--batch-size', default=1, type=int, help='batch-size (default: 1)')  
    parser.add_argument('--feature', default='relu3_1', type=str, help='feature level (default: relu2_2)')
    parser.add_argument('--recurrent-scale', default=0.5, type=float, help='recurrent scale (default: 0.5)')
    parser.add_argument('--output', default='./outputs/norm-check.csv', help='output file-name (default: ./outputs/PieAPP-noref.csv)')
    args = parser.parse_args()
    return args

def gram_matrix(x):
    a, b, c, d = x.size()
    features = x.view(a, b, c * d)
    gram = features.bmm(features.transpose(1, 2))
    return gram.div(b * c * d)

if __name__ == "__main__":
    # arguments
    args = get_arguments()

    # model
    model_config = {'scale':4, 'num_blocks':5}
    model = models.__dict__[args.gen_model]
    model = model(**model_config)
    model.load_state_dict(torch.load(args.gen_to_load, map_location='cpu'))
    model = model.to(args.device)

    # loader
    loader = get_loaders(args)
    
    # losses
    extractor = VGGFeaturesExtractor(feature_layer=args.feature).eval().to(args.device)

    # data-frame
    columns = ['image', 'rsty', 'sr_m_gt', 'asr_m_agt']
    df = pd.DataFrame(columns=None)

    # iter images
    for i, data in enumerate(loader['eval']):
        inputs = data['input'].to(args.device)
        targets = data['target'].to(args.device)
        path = data['path'][0]
        
        fakes = model(inputs)

        fakes_fea = extractor(fakes)
        targets_fea = extractor(targets)

        d = {'image': path}
        fakes_scaled = imresize(fakes, scale=args.recurrent_scale)
        targets_scaled = imresize(targets, scale=args.recurrent_scale) 

        fakes_scaled_fea = extractor(fakes_scaled)
        targets_scaled_fea = extractor(targets_scaled)

        fakes_gram = gram_matrix(fakes_fea)
        fakes_scaled_gram = gram_matrix(fakes_scaled_fea)
        targets_gram = gram_matrix(targets_fea)
        targets_scaled_gram = gram_matrix(targets_scaled_fea)

        rsty = ((fakes_gram - fakes_scaled_gram) - (targets_gram - targets_scaled_gram)).norm(p=1)
        sr_m_gt = (fakes_gram - targets_gram).norm(p=1)
        asr_m_agt = (fakes_scaled_gram - targets_scaled_gram).norm(p=1)
                
        d.update({'rsty': float(rsty.item()), 'sr_m_gt': float(sr_m_gt.item()), 'asr_m_agt': float(asr_m_agt.item())})
        df = df.append(d, ignore_index=True)

    df.to_csv(args.output)
    f = open(args.output.replace('csv', 'log'), 'w')
    d = df.describe()
    f.write(str(d))
    f.close()

    print(d)