import os
import sys
import glob
import random
import argparse

def get_arguments():
    parser = argparse.ArgumentParser(description='imagenet-style')
    parser.add_argument('--root', required=True, help='root directory')
    parser.add_argument('--num-images', default=1, type=int, help='super-resolution scale (default: 2)')
    parser.add_argument('--features', default=['relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], nargs='+', type=str, help='features level (default: relu2_1 relu2_2 relu3_1)')
    args = parser.parse_args()
    return args

def get_libraries(args):
    libraries = glob.glob(os.path.join(args.root, '*/'))
    libraries = [os.path.basename(os.path.dirname(l)) for l in libraries]
    return libraries

if __name__ == "__main__":
    # arguments
    args = get_arguments()

    # libraries
    libraries = get_libraries(args)

    # run expirement
    for i in range(args.num_images):
        print('Image {}/{}'.format(i + 1, args.num_images))
        library = random.choice(libraries)
        paths = glob.glob(os.path.join(args.root, library, '*.*'))
        path_a = random.choice(paths)
        path_b = random.choice(paths)
        feature = random.choice(args.features)
        output_dir = './outputs/' + library

        # cmd = 'python3 ex3.py --image-source {} --image-target {} --features {} --results-dir {} '.format(path_a, path_b, feature, output_dir)
        cmd = 'python3 ex3.py --image-source {} --image-target {} --results-dir {} '.format(path_a, path_b, output_dir)

        os.system(cmd + '--style-type style')
        # os.system(cmd + '--style-type mult-style --style-scales 0.5 1.0')
        os.system(cmd + '--style-type rec-style')

        os.system('cp {} {}'.format(path_a, output_dir))
        os.system('cp {} {}'.format(path_b, output_dir))

