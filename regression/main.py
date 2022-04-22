import argparse
import torch
import logging
import signal
import sys
import torch.backends.cudnn as cudnn
from trainer import Trainer
from datetime import datetime
from os import path
from utils import misc
from random import randint

# torch.autograd.set_detect_anomaly(True)

def get_arguments():
    parser = argparse.ArgumentParser(description='Internal-External super-resolution')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-ids', default=[0], type=int, nargs='+', help='device ids assignment (e.g 0 1 2 3')
    parser.add_argument('--model', default='vanilla', help='model architecture (default: vanilla)')
    parser.add_argument('--model-config', default='', help='additional architecture configuration')
    parser.add_argument('--model-to-load', default='', help='resume training from file (default: None)')
    parser.add_argument('--root', required=True, help='root internal dataset folder')
    parser.add_argument('--max-size', default=None, type=int, help='super-resolution scale (default: None)')
    parser.add_argument('--scale', default=4, type=int, help='super-resolution scale (default: 4)')
    parser.add_argument('--batch-size', default=16, type=int, help='augmentations (default: 16)')
    parser.add_argument('--crop-size', default=64, type=int, help='crop-size for augmentations (default: 64)')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers (default: 4)')
    parser.add_argument('--epochs', default=1000, type=int, help='steps (default: 1000)')
    parser.add_argument('--lr', default=2e-4, type=float, help='lr (default: 2e-4)')
    parser.add_argument('--betas', default=[0.9, 0.999], nargs=2, type=float, help='adams betas (default: 0.9 0.999)')
    parser.add_argument('--step-size', default=500, type=int, help='scheduler step size (default: 500)')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler gamma (default: 0.5)')
    parser.add_argument('--seed', default=-1, type=int, help='random seed (default: random)')
    parser.add_argument('--wasserstein1d', default=False, action='store_true', help='wasserstein1d distance (default: false)')
    parser.add_argument('--weights', default=[1.0, 1.0, 1.0, 1.0], nargs=4, type=float, help='features weights (default: 1.0 1.0 1.0 1.0)')
    parser.add_argument('--features', default=['relu1_2', 'relu2_1', 'relu2_2', 'relu3_1'], nargs=4, type=str, help='features level (default: relu1_2 relu2_1 relu2_2 relu3_1)')
    parser.add_argument('--print-every', default=50, type=int, help='print-every (default: 50)')
    parser.add_argument('--eval-every', default=100, type=int, help='eval-every (default: 100)')
    parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results', help='results dir')
    parser.add_argument('--save', metavar='SAVE', default='', help='saved folder')
    parser.add_argument('--use-tb', default=False, action='store_true', help='use tensorboardx (default: false)')
    parser.add_argument('--evaluation', default=False, action='store_true', help='evaluation (default: false)')
    args = parser.parse_args()

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.save == '':
        args.save = time_stamp
    args.save_path = path.join(args.results_dir, args.save)
    if args.seed == -1:
        args.seed = randint(0, 12345)
    return args

def main():
    args = get_arguments()

    torch.manual_seed(args.seed)

    # cuda
    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_ids[0])
        cudnn.benchmark = True
    else:
        args.device_ids = None

    # set logs
    misc.mkdir(args.save_path)
    misc.setup_logging(path.join(args.save_path, 'log.txt'))

    # print logs
    logging.info(args)

    # trainer
    trainer = Trainer(args)
    if args.evaluation:
        trainer.eval()
    else:
        trainer.train()

if __name__ == '__main__':
    # enables a ctrl-c without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    main()