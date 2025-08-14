import pdb
import torch
import numpy as np
import torch.optim as optim


class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        
        # Separate parameter groups for different learning rates
        spatial_params = []
        temporal_params = []
        
        for name, param in model.named_parameters():
            if 'spatial_branch' in name:
                spatial_params.append(param)
            else:
                temporal_params.append(param)
        
        # Set spatial LR to match temporal LR
        spatial_lr = self.optim_dict['base_lr'] * 1.0  # Same as temporal
        temporal_lr = self.optim_dict['base_lr']
        
        param_groups = [
            {'params': temporal_params, 'lr': temporal_lr, 'name': 'temporal'},
            {'params': spatial_params, 'lr': spatial_lr, 'name': 'spatial'}
        ]
        
        if self.optim_dict["optimizer"] == 'SGD':
            self.optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                nesterov=self.optim_dict['nesterov'],
                weight_decay=self.optim_dict['weight_decay']
            )
        elif self.optim_dict["optimizer"] == 'Adam':
            self.optimizer = optim.Adam(
                param_groups,
                weight_decay=self.optim_dict['weight_decay']
            )
        else:
            raise ValueError()
        self.scheduler = self.define_lr_scheduler(self.optimizer, self.optim_dict['step'])

    def define_lr_scheduler(self, optimizer, milestones):
        if self.optim_dict["optimizer"] in ['SGD', 'Adam']:
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
            return lr_scheduler
        else:
            raise ValueError()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
