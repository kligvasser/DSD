import logging
import models
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from data import get_loaders
from ast import literal_eval
from utils.recorderx import RecoderX
from models.modules.losses import Style, SlicedWasserstein, RMSELoss

class Trainer():
    def __init__(self, args):
        # parameters
        self.args = args
        self.print_model = True
        
        if self.args.use_tb:
            self.tb = RecoderX(log_dir=args.save_path)

        # initialize
        self._init()

    def _init_model(self):
        # initialize model
        if self.args.model_config != '':
            model_config = dict({}, **literal_eval(self.args.model_config))
        else:
            model_config = {}

        model = models.__dict__[self.args.model]
        self.model = model(**model_config)

        # print model
        if self.print_model:
            logging.info(self.model)
            logging.info('Number of parameters in model: {}\n'.format(sum([l.nelement() for l in self.model.parameters()])))
            self.print_model = False

        # loading weights
        if self.args.model_to_load != '':
            logging.info('\nLoading model...')
            self.model.load_state_dict(torch.load(self.args.model_to_load, map_location='cpu'))

        # gpu
        self.model = self.model.to(self.args.device)

        # parallel
        if self.args.device_ids and len(self.args.device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, self.args.device_ids)

    def _init_optim(self):
        # initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=self.args.betas)

        # initialize scheduler
        self.scheduler = StepLR(self.optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        # initialize criterion
        if self.args.wasserstein1d:
            self.criterion = SlicedWasserstein(features_to_compute=self.args.features, batch_size=self.args.batch_size).to(self.args.device)
        else:
            self.criterion = Style(features_to_compute=self.args.features, batch_size=self.args.batch_size).to(self.args.device)
        self.mse = torch.nn.MSELoss(reduction='none')
        self.rmse = RMSELoss(reduce=False)

    def _init(self):
        # init parameters
        self.train_steps = 0
        self.eval_steps = 0
        self.losses = {}
        self.losses['train'] = {'all': [], 'f0': [], 'f1': [], 'f2': [], 'f3': []}
        self.losses['eval'] = {'all': [], 'f0': [], 'f1': [], 'f2': [], 'f3': []}

        # initialize model
        self._init_model()

        # initialize optimizer
        self._init_optim()

    def _save_model(self, epoch):
        # save models
        torch.save(self.model.state_dict(), os.path.join(self.args.save_path, '{}_e{}.pt'.format(self.args.model, epoch + 1)))

    def _train_iteration(self, data):
        # set inputs
        self.train_steps += 1
        inputs = data['input'].to(self.args.device)
        targets = data['target'].to(self.args.device)
        regs = self.criterion(inputs, targets).detach()

        # zero grads
        self.optimizer.zero_grad()

        # get predictions
        outputs = self.model(inputs)
        loss = self.mse(outputs, regs).mean(dim=0)

        # weighted
        loss[0] /= (self.args.weights[0]) ** 2
        loss[1] /= (self.args.weights[1]) ** 2
        loss[2] /= (self.args.weights[2]) ** 2
        loss[3] /= (self.args.weights[3]) ** 2
        
        # record loss
        self.losses['train']['f0'].append(loss[0].data.item())
        self.losses['train']['f1'].append(loss[1].data.item())
        self.losses['train']['f2'].append(loss[2].data.item())
        self.losses['train']['f3'].append(loss[3].data.item())

        loss = loss.sum()
        self.losses['train']['all'].append(loss.data.item())

        # backward loss
        loss.backward()
        self.optimizer.step()

        # logging
        if self.train_steps % self.args.print_every == 0:
            line2print = 'Train: {}'.format(self.train_steps)
            line2print += ', all: {:.8f}, f0: {:.8f}, f1: {:.8f},  f2: {:.8f},  f3: {:.8f}'.format(self.losses['train']['all'][-1], self.losses['train']['f0'][-1],self.losses['train']['f1'][-1], self.losses['train']['f2'][-1], self.losses['train']['f3'][-1])
            logging.info(line2print)

        # plots for tensorboard
        if self.args.use_tb:
            self.tb.add_scalar('data/train/all', self.losses['train']['all'][-1], self.train_steps)
            self.tb.add_scalar('data/train/f0', self.losses['train']['f0'][-1], self.train_steps)
            self.tb.add_scalar('data/train/f1', self.losses['train']['f1'][-1], self.train_steps)
            self.tb.add_scalar('data/train/f2', self.losses['train']['f2'][-1], self.train_steps)
            self.tb.add_scalar('data/train/f3', self.losses['train']['f3'][-1], self.train_steps)

    def _eval_iteration(self, data, df):
        # set inputs
        self.eval_steps += 1
        inputs = data['input'].to(self.args.device)
        targets = data['target'].to(self.args.device)
        path = data['path'][0]

        regs = self.criterion(inputs, targets).detach()

        # get predictions
        outputs = self.model(inputs)
        # loss = self.mse(outputs, regs).mean(dim=0)
        loss = self.rmse(outputs, regs).mean(dim=0)

        # weighted
        loss[0] /= (self.args.weights[0]) ** 2
        loss[1] /= (self.args.weights[1]) ** 2
        loss[2] /= (self.args.weights[2]) ** 2
        loss[3] /= (self.args.weights[3]) ** 2
        
        # record loss
        self.losses['eval']['f0'].append(loss[0].data.item())
        self.losses['eval']['f1'].append(loss[1].data.item())
        self.losses['eval']['f2'].append(loss[2].data.item())
        self.losses['eval']['f3'].append(loss[3].data.item())

        loss = loss.sum()
        self.losses['eval']['all'].append(loss.data.item())

        # logging
        if self.eval_steps % self.args.print_every == 0:
            line2print = 'Eval: {}'.format(self.eval_steps)
            line2print += ', all: {:.8f}, f0: {:.8f}, f1: {:.8f},  f2: {:.8f},  f3: {:.8f}'.format(self.losses['eval']['all'][-1], self.losses['eval']['f0'][-1],self.losses['eval']['f1'][-1], self.losses['eval']['f2'][-1], self.losses['eval']['f3'][-1])
            logging.info(line2print)

        # plots for tensorboard
        if self.args.use_tb:
            self.tb.add_scalar('data/eval/all', self.losses['eval']['all'][-1], self.train_steps)
            self.tb.add_scalar('data/eval/f0', self.losses['eval']['f0'][-1], self.train_steps)
            self.tb.add_scalar('data/eval/f1', self.losses['eval']['f1'][-1], self.train_steps)
            self.tb.add_scalar('data/eval/f2', self.losses['eval']['f2'][-1], self.train_steps)
            self.tb.add_scalar('data/eval/f3', self.losses['eval']['f3'][-1], self.train_steps)

        # data frames
        df['true'] = df['true'].append(self._get_df_line(path, regs), ignore_index=True)
        df['preds'] = df['preds'].append(self._get_df_line(path, outputs), ignore_index=True)
        df['diff'] = df['diff'].append(self._get_df_line(path, torch.abs(regs-outputs)), ignore_index=True)

        return df
    
    def _get_df_line(self, path, outputs):
        d = {'image': path}
        for i, feature in enumerate(self.args.features):
            d.update({feature: outputs[0][i].data.item()})
        return d

    def _train_epoch(self, loader):
        self.model.train()

        # train over epochs
        for _, data in enumerate(loader):
            self._train_iteration(data)

    def _eval_epoch(self, loader):
        # data-frames
        columns = ['image'] + self.args.features
        df = {}
        df['true'] = pd.DataFrame(columns=columns)
        df['preds'] = pd.DataFrame(columns=columns)
        df['diff'] = pd.DataFrame(columns=columns)

        self.model.eval()
        # eval over epoch
        for _, data in enumerate(loader):
            df = self._eval_iteration(data, df)

        # describes
        logging.info('Eval True: \n{}'.format(df['true'].describe()))
        logging.info('Eval Predictions: \n{}'.format(df['preds'].describe()))
        logging.info('Eval difference: \n{}'.format(df['diff'].describe()))

        # save data-frames
        df['true'].to_csv(os.path.join(self.args.save_path, 'true.csv'))
        df['preds'].to_csv(os.path.join(self.args.save_path, 'pred.csv'))

    def _train(self, loaders):
        # run epoch iterations
        for self.epoch in range(self.args.epochs):
            logging.info('\nEpoch {}'.format(self.epoch + 1))

            # train
            self._train_epoch(loaders['train'])

            # scheduler
            self.scheduler.step(epoch=self.epoch )

            # evaluation
            if ((self.epoch + 1) % self.args.eval_every == 0) or ((self.epoch + 1) == self.args.epochs):
                self._eval_epoch(loaders['eval'])
                self._save_model(self.epoch)

    def train(self):
        # get loader
        loaders = get_loaders(self.args)

        # run training
        self._train(loaders)

        # close tensorboard
        if self.args.use_tb:
            self.tb.close()

    def eval(self):
        # get loader
        loaders = get_loaders(self.args)

        # evaluation
        logging.info('\nEvaluating...')
        self._eval_epoch(loaders['eval'])

        # close tensorboard
        if self.args.use_tb:
            self.tb.close()