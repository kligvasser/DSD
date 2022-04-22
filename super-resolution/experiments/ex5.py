import os
import sys
import math
import logging
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath('../'))
from utils import misc
from models.modules.misc import center_crop

from collections import OrderedDict
from utils.core import imresize
from models.vgg import MultiVGGFeaturesExtractor

class RecurrentStyleLoss(nn.Module):
    def __init__(self, features_to_compute=['relu1_2', 'relu2_1', 'relu3_1'], scales=[0.5], base_scale=1.0, criterion=torch.nn.L1Loss()):
        super(RecurrentStyleLoss, self).__init__()
        self.scales = scales
        self.base_scale = base_scale
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute, shave_edge=None).eval()

    def forward(self, inputs, targets):
        if self.base_scale < 1.0:
            inputs = imresize(inputs, scale=self.base_scale)
            targets = imresize(targets, scale=self.base_scale)

        loss = 0.
        for scale in self.scales:
            inputs_scaled = imresize(inputs, scale=scale)
            targets_scaled = imresize(targets, scale=scale)

            inputs_style = self._compute_style(inputs, inputs_scaled)
            with torch.no_grad():
                targets_style = self._compute_style(targets, targets_scaled)
            
            for key in inputs_style.keys():
                loss += self.criterion(inputs_style[key], targets_style[key].detach())
        
        return loss

    def _compute_style(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        targets_fea = self.features_extractor(targets)
        
        style = OrderedDict()
        for key in inputs_fea.keys():
            inputs_gram = self._gram_matrix(inputs_fea[key])
            targets_gram = self._gram_matrix(targets_fea[key])
            diff = inputs_gram - targets_gram
            style.update({key: diff})
        
        return style

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, features_to_compute=['relu1_2', 'relu2_1', 'relu3_1'], base_scale=1.0, criterion=torch.nn.L1Loss()):
        super(StyleLoss, self).__init__()
        self.base_scale = base_scale
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute, shave_edge=None).eval()

    def forward(self, inputs, targets):
        if self.base_scale < 1.0:
            inputs = imresize(inputs, scale=self.base_scale)
            targets = imresize(targets, scale=self.base_scale)

        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        loss = 0
        for key in inputs_fea.keys():
            inputs_gram = self._gram_matrix(inputs_fea[key])
            with torch.no_grad():
                targets_gram = self._gram_matrix(targets_fea[key]).detach()

            loss += self.criterion(inputs_gram, targets_gram)

        return loss

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

class ImageOpt(nn.Module):
    def __init__(self, in_channels=3, optim_size=1200, std=0.001):
        super(ImageOpt, self).__init__()
        # parameters
        self.optim_size = optim_size
        self.optim = nn.Parameter(std * torch.randn(1, in_channels, optim_size, optim_size), requires_grad=True)

    def forward(self, x):
        s = x.size()
        x = F.pad(x, [math.ceil((self.optim_size - s[3]) / 2), math.floor((self.optim_size - s[3]) / 2), math.ceil((self.optim_size - s[2]) / 2), math.floor((self.optim_size - s[2]) / 2)])
        x += self.optim
        x = center_crop(x, s[2], s[3])
        return x

def get_arguments():
    parser = argparse.ArgumentParser(description='image-optim-search ')
    parser.add_argument('--device', default='cuda', help='device assignment ("cpu" or "cuda")')
    parser.add_argument('--device-id', default=0, type=int, help='device id (default: 0)')
    parser.add_argument('--image-source', required=True, help='image-source')
    parser.add_argument('--image-target', required=True, help='image-target')
    parser.add_argument('--lr', default=5e-4, type=float, help='lr (default: 2e-4)')
    parser.add_argument('--iterations', default=3000, type=int, help='iterations (default: 3000)')
    parser.add_argument('--features', default=['relu2_1', 'relu2_2', 'relu3_1'], nargs='+', type=str, help='features level (default: relu2_1 relu2_2 relu3_1)')
    parser.add_argument('--base-scale', default=1.0, type=float, help='base-scale (default: 1.0)')
    parser.add_argument('--style-scales', default=[0.5], type=float, nargs='+', help='style scales (e.g 0.5 0.75)')
    parser.add_argument('--style-type', default='style', type=str, choices=['style', 'rec-style'], help='style-type (default: style)')
    parser.add_argument('--print-every', default=200, type=int, help='print-every (default: 200)')
    parser.add_argument('--results-dir', default='./outputs/', help='results-dir (default: ./outputs/)')
    args = parser.parse_args()

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.save_path = os.path.join(args.results_dir, time_stamp)  

    return args

def read_images(args):
    source = Image.open(args.image_source).convert('RGB')
    target = Image.open(args.image_target).convert('RGB')

    image_transform = transforms.ToTensor()

    source = image_transform(source).unsqueeze(dim=0).to(args.device)
    target = image_transform(target).unsqueeze(dim=0).to(args.device)

    return source, target

def save_image(args, image, i):
        save_path = os.path.join(args.save_path, os.path.basename(args.image_source).replace('.', '_{}.'.format(i)))
        misc.save_image(image.data.cpu(), save_path)

if __name__ == "__main__":
    # arguments
    args = get_arguments()

    # logging
    # misc.mkdir(args.save_path)
    os.makedirs(args.save_path)
    misc.setup_logging(os.path.join(args.save_path, 'log.txt'))
    logging.info(args)

    # models
    model = ImageOpt().to(args.device)

    if args.style_type == 'style':
        criterion = StyleLoss(features_to_compute=args.features, base_scale=args.base_scale).to(args.device)
    else:
        criterion = RecurrentStyleLoss(features_to_compute=args.features, scales=args.style_scales, base_scale=args.base_scale).to(args.device)

    # optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(args.iterations // 3), gamma=0.5)

    # read-images
    source, target = read_images(args)

    # run optimization
    for i in range(args.iterations + 1):
        optimizer.zero_grad()

        output = model(source)
        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()
        scheduler.step(epoch=i)

        # some details
        if (i % args.print_every == 0):
            logging.info('iter {}/{}, loss: {:.8f}'.format(i, args.iterations, float(loss.data.item())))
            save_image(args, output, i)


    
    
