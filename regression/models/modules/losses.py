import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import MultiVGGFeaturesExtractor
from utils.core import imresize
from collections import OrderedDict

class RMSELoss(nn.Module):
    def __init__(self, reduce=True):
        super(RMSELoss, self).__init__()
        self.reduce = reduce
        self.mse = torch.nn.MSELoss(reduction='none')

    def forward(self, inputs, targets):
        mse = self.mse(inputs, targets)
        loss = torch.sqrt(mse)

        if self.reduce:
            loss = loss.mean()

        return loss

class PerceptualLoss(nn.Module):
    def __init__(self, features_to_compute=['conv5_4'], criterion=torch.nn.L1Loss()):
        super(PerceptualLoss, self).__init__()
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        loss = 0
        for key in inputs_fea.keys():
            loss += self.criterion(inputs_fea[key], targets_fea[key].detach())

        return loss

class ConsistencyLoss(nn.Module):
    def __init__(self, scale=0.5, criterion=torch.nn.L1Loss()):
        super(ConsistencyLoss, self).__init__()
        self.scale = scale
        self.criterion = criterion

    def forward(self, inputs, targets):
        inputs = imresize(inputs, scale=self.scale)
        targets = imresize(targets, scale=self.scale)
        loss = self.criterion(inputs, targets.detach())

        return loss

class Style(nn.Module):
    def __init__(self, features_to_compute, batch_size, criterion=torch.nn.L1Loss(reduction='none')):
        super(Style, self).__init__()
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()
        self.register_buffer('loss', torch.zeros(batch_size, len(features_to_compute)))

    def forward(self, inputs, targets):
        self.loss.zero_()

        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        for i, key in enumerate(inputs_fea.keys()):
            inputs_gram = self._gram_matrix(inputs_fea[key])
            with torch.no_grad():
                targets_gram = self._gram_matrix(targets_fea[key]).detach()
            self.loss[:, i] = self.criterion(inputs_gram, targets_gram).flatten(start_dim=1).mean(dim=1)

        return self.loss

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

class SlicedWasserstein(nn.Module):
    def __init__(self, features_to_compute, criterion=torch.nn.MSELoss()):
        super(SlicedWasserstein, self).__init__()
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
        loss = 0

        inputs_fea = self.features_extractor(inputs)
        with torch.no_grad():
            targets_fea = self.features_extractor(targets)

        for  key in inputs_fea.keys():
            loss += self._wasserstein1d_loss(inputs_fea[key], targets_fea[key])

        return loss

    def _wasserstein1d_loss(self, inputs, targets):
        inputs = inputs.flatten(start_dim=2)
        targets = targets.flatten(start_dim=2)
        targets = targets[:, :, torch.randperm(targets.size(2))[:(inputs.size(2))]]
        sorrted_inputs, _ = torch.sort(inputs, dim=-1)
        sorrted_stargets, _ = torch.sort(targets, dim=-1)
        return self.criterion(sorrted_inputs, sorrted_stargets.detach())

class RecurrentStyleLoss(nn.Module):
    def __init__(self, features_to_compute=['relu1_2', 'relu2_1', 'relu3_1'], scales=[0.5], criterion=torch.nn.L1Loss()):
        super(RecurrentStyleLoss, self).__init__()
        self.scales = scales
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
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

class MultiStyleLoss(nn.Module):
    def __init__(self, features_to_compute=['relu1_2', 'relu2_1', 'relu3_1'], scales=[1.0, 0.5], criterion=torch.nn.L1Loss(), shave_edge=None):
        super(MultiStyleLoss, self).__init__()
        self.scales = scales
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
        loss = 0.
        for scale in self.scales:
            if scale != 1.0:
                inputs_scaled = imresize(inputs, scale=scale)
                targets_scaled = imresize(targets, scale=scale)
                inputs_fea = self.features_extractor(inputs_scaled)
                with torch.no_grad():
                    targets_fea = self.features_extractor(targets_scaled)
            else:
                inputs_fea = self.features_extractor(inputs)
                with torch.no_grad():
                    targets_fea = self.features_extractor(targets)
            
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

class RecurrentWassersteinLoss(nn.Module):
    def __init__(self, features_to_compute=['relu1_2', 'relu2_1', 'relu3_1'], scales=[0.25], criterion=torch.nn.MSELoss()):
        super(RecurrentWassersteinLoss, self).__init__()
        self.scales = scales
        self.criterion = criterion
        self.features_extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute).eval()

    def forward(self, inputs, targets):
        loss = 0.
        for scale in self.scales:
            inputs_scaled = imresize(inputs, scale=scale)
            targets_scaled = imresize(targets, scale=scale)

            inputs_wasserstein = self._compute_wasserstein(inputs, inputs_scaled)
            with torch.no_grad():
                itargets_wasserstein = self._compute_wasserstein(targets, targets_scaled)
            
            for key in inputs_wasserstein.keys():
                loss += self.criterion(inputs_wasserstein[key], itargets_wasserstein[key].detach())
        
        return loss

    def _compute_wasserstein(self, inputs, targets):
        inputs_fea = self.features_extractor(inputs)
        targets_fea = self.features_extractor(targets)
        
        wasserstein = OrderedDict()
        for key in inputs_fea.keys():
            diff = self._sliced_wasserstein1d(inputs_fea[key], targets_fea[key])
            wasserstein.update({key: diff})
        
        return wasserstein

    def _sliced_wasserstein1d(self, inputs, targets):
        inputs = inputs.flatten(start_dim=2)
        targets = targets.flatten(start_dim=2)
        inputs = inputs[:, :, torch.randperm(inputs.size(2))[:(targets.size(2))]]
        sorrted_inputs, _ = torch.sort(inputs, dim=-1)
        sorrted_targets, _ = torch.sort(targets, dim=-1)
        diff = sorrted_inputs - sorrted_targets
        return diff

class PatchesStyleLoss(nn.Module):
    def __init__(self, kernel_sizes=[(5, 5), (7, 7), (11, 11)], reduce_mean=True, criterion=torch.nn.L1Loss()):
        super(PatchesStyleLoss, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.reduce_mean = reduce_mean
        self.criterion = criterion

    def forward(self, inputs, targets):
        loss = 0.
        for kernel_size in self.kernel_sizes:
            inputs_pch = self._image_to_patches(inputs, kernel_size=kernel_size)
            inputs_gram = self._gram_matrix(inputs_pch)

            with torch.no_grad():
                targets_pch = self._image_to_patches(targets, kernel_size=kernel_size)
                targets_gram = self._gram_matrix(targets_pch).detach()

            loss += self.criterion(inputs_gram, targets_gram)

        return loss

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

    def _image_to_patches(self, x, kernel_size):
        x = F.unfold(x, kernel_size=kernel_size, padding=0).unsqueeze(dim=-1)
        if self.reduce_mean:
            x = x - x.mean(dim=1, keepdim=True)
        return x

class RecurrentPatchesStyleLoss(nn.Module):
    def __init__(self, kernel_sizes=[(5, 5), (7, 7), (11, 11)], reduce_mean=True, scales=[0.25], criterion=torch.nn.L1Loss()):
        super(RecurrentPatchesStyleLoss, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.reduce_mean = reduce_mean
        self.scales = scales
        self.criterion = criterion

    def forward(self, inputs, targets):
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
        style = OrderedDict()
        for kernel_size in self.kernel_sizes:
            inputs_pch = self._image_to_patches(inputs, kernel_size)
            targets_pch = self._image_to_patches(targets, kernel_size)

            inputs_gram = self._gram_matrix(inputs_pch)
            targets_gram = self._gram_matrix(targets_pch)

            diff = inputs_gram - targets_gram
            style.update({'k{}'.format(kernel_size[0]): diff})
        
        return style

    def _gram_matrix(self, x):
        a, b, c, d = x.size()
        features = x.view(a, b, c * d)
        gram = features.bmm(features.transpose(1, 2))
        return gram.div(b * c * d)

    def _image_to_patches(self, x, kernel_size):
        x = F.unfold(x, kernel_size=kernel_size, padding=0).unsqueeze(dim=-1)
        if self.reduce_mean:
            x = x - x.mean(dim=1, keepdim=True)
        return x

class ContextualLoss(nn.Module):
    def __init__(self, features_to_compute=['relu2_1'], h=0.5, eps=1e-5):
        super(ContextualLoss, self).__init__()
        self.h = h
        self.eps = eps
        self.extractor = MultiVGGFeaturesExtractor(target_features=features_to_compute, requires_grad=False, use_input_norm=True).eval()

    def forward(self, inputs, targets):
        inputs_fea = self.extractor(inputs)
        with torch.no_grad():
            targets_fea = self.extractor(targets)

        loss = 0
        for key in inputs_fea.keys():
            loss += self._contextual_loss(inputs_fea[key], targets_fea[key])

        return loss
    
    def _contextual_loss(self, inputs, targets):
        dist = self._cosine_dist(inputs, targets)
        dist_min, _ = torch.min(dist, dim=2, keepdim=True)

        # Eq (2)
        dist_tilde = dist / (dist_min + self.eps)
        
        # Eq (3)
        w = torch.exp((1 - dist_tilde) / self.h)

        # Eq (4)
        cx_ij = w / torch.sum(w, dim=2, keepdim=True)       # (N, H*W, H*W)

        # Eq (1)
        cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)  # (N, )
        loss = torch.mean(-torch.log(cx + self.eps))
    
        return loss

    def _cosine_dist(self, x, y):
        # reduce mean
        y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
        x_centered = x - y_mu
        y_centered = y - y_mu

        # L2 normalization
        x_normalized = F.normalize(x_centered, p=2, dim=1)
        y_normalized = F.normalize(y_centered, p=2, dim=1)

        # channel-wise vectorization
        x_normalized = x_normalized.flatten(start_dim=2)  # (N, C, H*W)
        y_normalized = y_normalized.flatten(start_dim=2)  # (N, C, H*W)

        # consine similarity
        cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)  # (N, H*W, H*W)

        # convert to distance
        dist = 1 - cosine_sim

        return dist