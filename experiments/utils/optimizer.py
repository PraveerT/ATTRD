import pdb
import torch
import numpy as np
import torch.optim as optim


class Optimizer(object):
    def __init__(self, model, optim_dict):
        self.optim_dict = optim_dict
        self.spatial_prefix = self.optim_dict.get('spatial_prefix', 'spatial_branch').strip('.')
        self.temporal_prefix = self.optim_dict.get('temporal_prefix', 'temporal_branch').strip('.')

        # Separate parameter groups for different learning rates
        spatial_params = []
        temporal_params = []
        fusion_params = []

        fusion_lr_mult = self.optim_dict.get('fusion_lr_mult', 1.0)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            normalized_name = name[7:] if name.startswith('module.') else name
            if self.spatial_prefix and (
                normalized_name == self.spatial_prefix
                or normalized_name.startswith(self.spatial_prefix + '.')
            ):
                spatial_params.append(param)
            elif self.temporal_prefix and (
                normalized_name == self.temporal_prefix
                or normalized_name.startswith(self.temporal_prefix + '.')
            ):
                temporal_params.append(param)
            else:
                fusion_params.append(param)

        base_lr = self.optim_dict['base_lr']

        param_groups = []
        if temporal_params:
            param_groups.append({'params': temporal_params, 'lr': base_lr, 'name': 'temporal'})
        if spatial_params:
            param_groups.append({'params': spatial_params, 'lr': base_lr, 'name': 'spatial'})
        if fusion_params:
            param_groups.append({'params': fusion_params, 'lr': base_lr * fusion_lr_mult, 'name': 'fusion'})
        if not param_groups:
            raise ValueError("No trainable parameters were found for optimizer setup.")
        
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
        scheduler_type = self.optim_dict.get('scheduler', 'multistep')
        if scheduler_type == 'cosine':
            num_epochs = self.optim_dict.get('num_epochs', milestones[-1] if milestones else 200)
            start_epoch = self.optim_dict.get('start_epoch', 0)
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - start_epoch,
                eta_min=self.optim_dict['base_lr'] * 0.01,
            )
        if scheduler_type == 'constant_then_cosine':
            # Constant base_lr until cosine_start, then cosine decay to num_epochs.
            num_epochs = self.optim_dict['num_epochs']
            cosine_start = self.optim_dict['cosine_start']
            eta_min = self.optim_dict['base_lr'] * self.optim_dict.get('eta_min_ratio', 0.01)
            constant = optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=cosine_start,
            )
            cosine = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - cosine_start, eta_min=eta_min,
            )
            return optim.lr_scheduler.SequentialLR(
                optimizer, [constant, cosine], milestones=[cosine_start],
            )
        if self.optim_dict["optimizer"] in ['SGD', 'Adam']:
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        raise ValueError()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)
