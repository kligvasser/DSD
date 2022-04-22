import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def get_arguments():
    parser = argparse.ArgumentParser(description='nn-search')
    parser.add_argument('--pkl-file', required=True, help='root directory')
    parser.add_argument('--key', default='pch_gram', type=str, help='key (default: pch_gram)')
    parser.add_argument('--num-samples', default=20, type=int, help='number of samples (default: 20)')
    parser.add_argument('--num-nn', default=5, type=int, help='number of samples (default: 5)')
    parser.add_argument('--output', default='./outputs/nn-search.csv', help='output file-name (default: ./outputs/nn-search.csv)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # arguments
    args = get_arguments()
    print(args)

    # dataframes
    columns = ['image'] + ['{}_nn'.format(i) for i in range(args.num_nn)] + ['{}_dist'.format(i) for i in range(args.num_nn)]
    df = pd.DataFrame(columns=columns)

    dataset = pd.read_pickle(args.pkl_file)
    sampled = dataset.sample(n=args.num_samples, random_state=1)

    # run-nnseach
    neigh = NearestNeighbors(n_neighbors=args.num_nn, radius=0.1)
    neigh.fit(np.stack(dataset[args.key].values, axis=0).reshape(len(dataset), -1))
    dist, ind = neigh.kneighbors(np.stack(sampled[args.key].values, axis=0), n_neighbors=args.num_nn, return_distance=True)

    for i in range(args.num_samples):
        d = {'image': sampled['image'].iloc[i]}
        for j in range(args.num_nn):
            d.update({'{}_nn'.format(j): dataset['image'].iloc[ind[i, j]], '{}_dist'.format(j): dist[i, j]})
        df = df.append(d, ignore_index=True)
    
    # write csv
    df.to_csv(args.output)

        

