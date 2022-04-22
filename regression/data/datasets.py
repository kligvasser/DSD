import torch
import os
import glob
import random
import pandas as pd
import numpy as np
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image
from utils.core import imresize

class DatasetSR(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', scale=4, training=True, crop_size=64, max_size=None):
        self.root = root
        self.scale = scale if (scale % 1) else int(scale)
        self.training = training
        self.crop_size = crop_size
        self.max_size = max_size

        self._init()

    def _init(self):
        # data paths
        inputs = glob.glob(os.path.join(self.root, 'img_x{}'.format(self.scale), '*.*'))[:self.max_size]
        targets = [x.replace('img_x{}'.format(self.scale), 'img') for x in inputs]
        self.paths = {'input' : inputs, 'target' : targets}

        # transforms
        t_list = [transforms.ToTensor()]
        if self.training:
            t_list.append(lambda x: ((255. * x) + torch.zeros_like(x).uniform_(0., 1.)) / 256.,)
        self.image_transform = transforms.Compose(t_list)

    def _get_augment_params(self, size):
        random.seed(random.randint(0, 12345))

        # position
        w_size, h_size = size
        
        # parameters
        if self.training:
            x = random.randint(0, max(0, w_size - self.crop_size))
            y = random.randint(0, max(0, h_size - self.crop_size))
            flip = random.random() > 0.5
        else:
            x = w_size // 2
            y = h_size // 2
            flip = False

        return {'crop_pos': (x, y), 'flip': flip}

    def _augment(self, image, aug_params, scale=1):
        x, y = aug_params['crop_pos']
        image = image.crop((x * scale, y * scale, x * scale + self.crop_size * scale, y * scale + self.crop_size * scale))
        if aug_params['flip']:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def __getitem__(self, index):
        # input image
        input = Image.open(self.paths['input'][index]).convert('RGB')

        # target image
        target = Image.open(self.paths['target'][index]).convert('RGB')

        # augment
        aug_params = self._get_augment_params(input.size)
        input = self._augment(input, aug_params)
        target = self._augment(target, aug_params, self.scale)

        input = self.image_transform(input)
        target = self.image_transform(target)

        return {'input': input, 'target': target, 'path': self.paths['target'][index]}

    def __len__(self):
        return len(self.paths['input'])

class DatasetEval(torch.utils.data.dataset.Dataset):
    def __init__(self, root_sr='', root_resized='', scale=4, crop_size=80, batch_size=8, max_size=None):
        self.root_sr = root_sr
        self.root_resized = root_resized
        self.scale = scale
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.max_size = max_size
        self.index = 0

        self._init()

    def _init(self):
        # data paths
        sr = (sorted(glob.glob(os.path.join(self.root_sr, '*.*'))))[:self.max_size]
        resized = (sorted(glob.glob(os.path.join(self.root_resized, '*.*'))))[:self.max_size]
        self.paths = {'sr': sr, 'resized': resized}

        # transforms
        t_list = [transforms.ToTensor()]
        t_list.append(lambda x: ((255. * x) + torch.zeros_like(x).uniform_(0., 1.)) / 256.,)
        self.image_transform = transforms.Compose(t_list)

    def _get_augment_params(self, size):
        random.seed(random.randint(0, 12345))

        # position
        w_size, h_size = size
        
        # parameters
        x = random.randint(0, max(0, w_size - self.crop_size))
        y = random.randint(0, max(0, h_size - self.crop_size))
        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}

    def _augment(self, image, aug_params, scale=1):
        x, y = aug_params['crop_pos']
        image = image.crop((x * scale, y * scale, x * scale + self.crop_size * scale, y * scale + self.crop_size * scale))
        if aug_params['flip']:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        return image

    def __getitem__(self, index):
        i = index // self.batch_size

        # open images
        resized = Image.open(self.paths['resized'][i]).convert('RGB')
        sr = Image.open(self.paths['sr'][i]).convert('RGB')

        # augment
        aug_params = self._get_augment_params(resized.size)
        resized = self._augment(resized, aug_params)
        sr = self._augment(sr, aug_params, self.scale)
        resized = self.image_transform(resized)
        sr = self.image_transform(sr)

        return {'sr': sr, 'resized': resized, 'path': self.paths['sr'][i]}

    def __len__(self):
        if self.max_size:
            return self.max_size * self.batch_size
        else:
            return len(self.paths['sr']) * self.batch_size

class DatasetPieAPP(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', csv='', data_type='test'):
        self.root = root
        self.data_type = data_type
        self.df = pd.read_csv(csv)

        self._init()

    def _init(self):
        # transforms
        t_list = [transforms.ToTensor()]
        t_list.append(lambda x: ((255. * x) + torch.zeros_like(x).uniform_(0., 1.)) / 256.,)
        self.image_transform = transforms.Compose(t_list)

    def __getitem__(self, index):
        # open images
        ref_name = self.df.loc[index, 'ref. image']
        distortion_a = self.df.loc[index, ' distorted image A']
        distortion_b = self.df.loc[index, ' distorted image B']
        score = self.df.loc[index, ' preference for A']
        
        ref_path = os.path.join(self.root, 'reference_images', self.data_type, ref_name)
        image_ref = Image.open(ref_path).convert('RGB')
        if ref_name == distortion_a:
            image_a = image_ref
            a_path = ref_path
        else:
            a_path = os.path.join(self.root, 'distorted_images', self.data_type, self.df.loc[index, 'ref. image'].split('.')[0], distortion_a)
            image_a = Image.open(a_path).convert('RGB')
        b_path = os.path.join(self.root, 'distorted_images', self.data_type, self.df.loc[index, 'ref. image'].split('.')[0], distortion_b)
        image_b = Image.open(b_path).convert('RGB')

        # transform
        image_ref = self.image_transform(image_ref)
        image_a = self.image_transform(image_a)
        image_b = self.image_transform(image_b)

        return {'image_ref': image_ref, 'image_a': image_a, 'image_b': image_b, 'ref_name':ref_name, 'distortion_a': distortion_a, 'distortion_b': distortion_b, 'score': score, 'ref_path': ref_path, 'a_path': a_path, 'b_path': b_path}

    def __len__(self):
        return len(self.df)

class DatasetBAPPS(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', max_size=None):
        self.root = root
        self.max_size = max_size

        self._init()

    def _init(self):
        # data paths
        judge = sorted(glob.glob(os.path.join(self.root, 'judge', '*.*')))[:self.max_size]
        p0 = sorted(glob.glob(os.path.join(self.root, 'p0', '*.*')))[:self.max_size]
        p1 = sorted(glob.glob(os.path.join(self.root, 'p1', '*.*')))[:self.max_size]
        ref = sorted(glob.glob(os.path.join(self.root, 'ref', '*.*')))[:self.max_size]
        self.paths = {'judge': judge, 'p0': p0, 'p1': p1, 'ref': ref}

        # transforms
        t_list = [transforms.ToTensor()]
        t_list.append(lambda x: ((255. * x) + torch.zeros_like(x).uniform_(0., 1.)) / 256.,)
        self.image_transform = transforms.Compose(t_list)

    def __getitem__(self, index):
        # open images
        print(self.paths['p0'][index], self.paths['p1'][index], self.paths['ref'][index], np.load(self.paths['judge'][index]))
        p0 = Image.open(self.paths['p0'][index]).convert('RGB')
        p1 = Image.open(self.paths['p1'][index]).convert('RGB')
        ref = Image.open(self.paths['ref'][index]).convert('RGB')
        judge = np.load(self.paths['judge'][index])
        path = self.paths['ref'][index]

        # transform
        p0 = self.image_transform(p0)
        p1 = self.image_transform(p1)
        ref = self.image_transform(ref)

        return {'p0': p0, 'p1': p1, 'ref': ref, 'judge': judge, 'path': path}

    def __len__(self):
        return len(self.paths['judge'])


class DatasetPIPAL(torch.utils.data.dataset.Dataset):
    def __init__(self, root='', data_type='train', max_size=None):
        self.root = root
        self.data_type = data_type
        self.max_size = max_size

        self._init()

    def _init(self):
        # data paths
        self._set_data_paths()

        # transforms
        t_list = [transforms.ToTensor()]
        t_list.append(lambda x: ((255. * x) + torch.zeros_like(x).uniform_(0., 1.)) / 256.,)
        self.image_transform = transforms.Compose(t_list)

    def _set_data_paths(self):
        # distortions
        distortions = sorted(glob.glob(os.path.join(self.root, self.data_type, 'distortion', '*.*')))[:self.max_size]
        
        # referance and labels
        referances, labels = [], []
        for p in distortions:
            basename = os.path.basename(p).split('_')[0]
            df = pd.read_csv(os.path.join(self.root, self.data_type, 'label', '{}.txt'.format(basename)), header=None).T
            df.columns = df.iloc[0]
            labels.append(float(df[os.path.basename(p)].iloc[1]))
            referances.append(os.path.join(self.root, self.data_type, 'ref', '{}.bmp'.format(basename)))
            del df
        self.paths = {'distortion': distortions, 'referance': referances, 'label': labels}

    def __getitem__(self, index):
        # paths
        ref_path = self.paths['referance'][index]
        dist_path = self.paths['distortion'][index]

        # open images
        ref = Image.open(ref_path).convert('RGB')
        dist = Image.open(dist_path).convert('RGB')

        # transform
        ref = self.image_transform(ref)
        dist = self.image_transform(dist)

        # label
        label = self.paths['label'][index]

        return {'ref': ref, 'dist': dist, 'label': label, 'ref_path': ref_path, 'dist_path': dist_path}

    def __len__(self):
        return len(self.paths['distortion'])
