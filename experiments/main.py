"""Minimal trainer for CN-XXL on NVGesture.

Loads a yaml config, instantiates the model + dataloader + optimizer, runs
the train/eval loop with optional auxiliary loss support, saves checkpoints.

No telegram, no oracle/fusion telemetry, no shuffle-mix, no qcc scheduling,
no branch-specific losses, no sample weighting. Just training.
"""
import argparse
import os
import sys
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import get_parser, import_class, GpuDataParallel, Optimizer, Recorder, Stat, RandomState


def dynamic_pts_size(epoch, arg):
    """Linear ramp 48->128 over ep [0,50), quadratic ramp 128->256 over [50,100),
    fixed 256 after. Can be overridden by --pts-size CLI or static config."""
    if epoch < 50:
        return int(48 + (128 - 48) * (epoch / 50))
    if epoch < 100:
        progress = (epoch - 50) / 50
        return int(128 + (256 - 128) * (progress ** 2))
    return 256


class Processor:
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = RandomState(seed=self.arg.random_seed)
        self.device = GpuDataParallel()
        self.device.set_device(self.arg.device)
        self.recoder = Recorder(self.arg.work_dir, self.arg.print_log)
        self.data_loader = {}
        self.topk = (1, 5)
        self.stat = Stat(self.arg.model_args['num_classes'], self.topk)
        self.model, self.optimizer = self.Loading()
        self.loss = self.criterion()
        self.best_accuracy = 0.0
        self.use_static_pts = ('--pts-size' in sys.argv) or (
            not getattr(self.arg, 'dynamic_pts_size', True)
        )

    # ---------------------------------------------------------------- loss
    def criterion(self):
        loss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='none')
        return self.device.criterion_to_device(loss)

    # ---------------------------------------------------------------- model setup
    def Loading(self):
        self.recoder.print_log('Loading model')
        model_class = import_class(self.arg.model)
        model = model_class(**self.arg.model_args)
        if self.arg.weights:
            self._load_weights(model, self.arg.weights)
        model = self.device.model_to_device(model)
        optimizer = Optimizer(model, self.arg.optimizer_args)
        if self.arg.resume:
            self._resume_optimizer_state(optimizer)
        self.recoder.print_log('Loading model finished.')
        self.load_data()
        return model, optimizer

    def _load_weights(self, model, weights_path):
        self.recoder.print_log(f'Initializing model weights from {weights_path}.')
        payload = torch.load(weights_path, map_location='cpu')
        state = payload['model_state_dict'] if (
            isinstance(payload, dict) and 'model_state_dict' in payload
        ) else payload
        # Normalize 'module.' prefix from DataParallel checkpoints
        state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
        res = model.load_state_dict(state, strict=self.arg.strict_load)
        if res.missing_keys:
            self.recoder.print_log(f'  missing keys: {len(res.missing_keys)}')
        if res.unexpected_keys:
            self.recoder.print_log(f'  unexpected keys: {len(res.unexpected_keys)}')

    def _resume_optimizer_state(self, optimizer):
        ckpt = torch.load(self.arg.weights, map_location='cpu')
        if not isinstance(ckpt, dict):
            return
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except (ValueError, RuntimeError) as e:
                self.recoder.print_log(f'optimizer state restore skipped: {e}')
        if 'scheduler_state_dict' in ckpt:
            try:
                optimizer.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except (ValueError, RuntimeError):
                pass
        if 'epoch' in ckpt:
            self.arg.optimizer_args['start_epoch'] = ckpt['epoch'] + 1
            self.recoder.print_log(
                f'Resuming from checkpoint: epoch {self.arg.optimizer_args["start_epoch"]}'
            )

    # ---------------------------------------------------------------- data
    def load_data(self):
        self.recoder.print_log('Loading data')
        dataset_class = import_class(self.arg.dataloader)
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset_class(**self.arg.train_loader_args),
                batch_size=self.arg.batch_size, shuffle=True,
                num_workers=self.arg.num_worker, pin_memory=True,
            )
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset_class(**self.arg.test_loader_args),
            batch_size=self.arg.test_batch_size, shuffle=False,
            num_workers=self.arg.num_worker, pin_memory=True,
        )
        self.recoder.print_log('Loading data finished.')

    # ---------------------------------------------------------------- training
    def train(self, epoch):
        self.model.train()
        model_ref = self.model.module if hasattr(self.model, 'module') else self.model

        # pts_size scheduling
        if self.use_static_pts:
            pts_size = self.arg.pts_size
            tag = '--pts-size' if '--pts-size' in sys.argv else 'config'
            self.recoder.print_log(
                f'Training epoch: {epoch + 1} | pts_size: {pts_size} (static from {tag})'
            )
        else:
            pts_size = dynamic_pts_size(epoch, self.arg)
            self.recoder.print_log(
                f'Training epoch: {epoch + 1} | pts_size: {pts_size} (dynamic)'
            )
        model_ref.pts_size = pts_size
        self.arg.model_args['pts_size'] = pts_size

        loader = self.data_loader['train']
        loss_value = []
        correct, total = 0, 0
        self.recoder.timer_reset()
        cur_lr = [g['lr'] for g in self.optimizer.optimizer.param_groups]

        bar = tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch + 1}', leave=False)
        for batch_idx, data in bar:
            self.recoder.record_timer('dataloader')
            image = self.device.data_to_device(data[0])
            label = self.device.data_to_device(data[1])
            self.recoder.record_timer('device')

            output = self.model(image)
            self.recoder.record_timer('forward')

            loss = torch.mean(self.loss(output, label))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.recoder.record_timer('backward')

            loss_value.append(loss.item())
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += label.size(0)

            bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%',
                'LR': f'{cur_lr[0]:.6f}',
                'PTS': pts_size,
            })

            if batch_idx % self.arg.log_interval == 0:
                self.recoder.print_log(
                    f'\tEpoch: {epoch}, Batch({batch_idx}/{len(loader)}) done. '
                    f'Loss: {loss.item():.8f}  lr:{cur_lr[0]:f}'
                )
                self.recoder.print_time_statistics()

        train_acc = 100. * correct / max(1, total)
        train_loss = float(np.mean(loss_value)) if loss_value else 0.0
        self.recoder.print_log(f'\tMean training acc:  {train_acc:.4f}%.')
        self.recoder.print_log(f'\tMean training loss: {train_loss:.10f}.')

        self.optimizer.scheduler.step()
        return train_acc, train_loss

    # ---------------------------------------------------------------- evaluation
    def eval(self, loader_name=('test',)):
        self.model.eval()
        self.stat.reset_statistic()
        eval_loss_values = []
        n_samples = 0
        with torch.no_grad():
            for name in loader_name:
                loader = self.data_loader[name]
                self.stat.test_size = len(loader.dataset)
                for data in loader:
                    image = self.device.data_to_device(data[0])
                    label = self.device.data_to_device(data[1])
                    output = self.model(image)
                    loss = torch.mean(self.loss(output, label))
                    eval_loss_values.append(loss.item() * label.size(0))
                    n_samples += label.size(0)
                    self.stat.update_accuracy(output.data.cpu(), label.cpu(), topk=self.topk)
        mean_loss = sum(eval_loss_values) / max(1, n_samples)
        self.recoder.print_log(f'mean loss: {mean_loss}')

    # ---------------------------------------------------------------- main loop
    def start(self):
        if self.arg.phase == 'train':
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                eval_interval = 10 if (epoch + 1) < 100 else 1
                save_interval = self.arg.save_interval if (epoch + 1) < 100 else 1
                save_now = (epoch + 1) % save_interval == 0 or (epoch + 1) == self.arg.num_epoch
                eval_now = (epoch + 1) % eval_interval == 0 or (epoch + 1) == self.arg.num_epoch

                train_acc, train_loss = self.train(epoch)
                if save_now:
                    self.save_model(epoch, self.model, self.optimizer,
                                    f'{self.arg.work_dir}/epoch{epoch + 1}_model.pt')
                if eval_now:
                    self.eval(loader_name=['test'])
                    self.print_inf_log(epoch + 1, 'Test', train_acc, train_loss)
        elif self.arg.phase == 'test':
            if not self.arg.weights:
                raise ValueError('phase=test requires --weights')
            self.recoder.print_log(f'Evaluating: {self.arg.weights}')
            self.eval(loader_name=['test'])
            self.print_inf_log(0, 'Test')

    # ---------------------------------------------------------------- logging
    def print_inf_log(self, epoch, mode, train_acc=None, train_loss=None):
        static = self.stat.show_accuracy(f'{self.arg.work_dir}/{mode}_confusion_mat')
        prec1 = static[str(self.topk[0])] / self.stat.test_size * 100
        prec5 = static[str(self.topk[1])] / self.stat.test_size * 100
        self.recoder.print_log(
            f'Epoch {epoch}, {mode}, Evaluation: prec1 {prec1:.4f}, prec5 {prec5:.4f}'
        )
        self.recoder.print_log(f'Confusion Matrix (epoch {epoch}, {mode}):')
        cm = self.stat.confusion_matrix
        n_correct = int(cm.diagonal().sum())
        n_total = int(cm.sum())
        self.recoder.print_log(f'  Total Correct: {n_correct}.0/{n_total}.0')
        overall = 100. * n_correct / max(1, n_total)
        self.recoder.print_log(f'  Overall Accuracy: {overall:.2f}%')
        if prec1 > self.best_accuracy:
            self.best_accuracy = float(prec1)
            best_path = f'{self.arg.work_dir}/best_model.pt'
            self.save_model(epoch, self.model, self.optimizer, best_path)
            self.recoder.print_log(
                f'  Saved new best to {best_path} at prec1={prec1:.2f}% (prec1={prec1:.2f}%)'
            )

    # ---------------------------------------------------------------- checkpoint
    def save_model(self, epoch, model, optimizer, save_path):
        model_state = (model.module if hasattr(model, 'module') else model).state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
            'scheduler_state_dict': optimizer.scheduler.state_dict(),
        }, save_path)

    def save_arg(self):
        os.makedirs(self.arg.work_dir, exist_ok=True)
        with open(f'{self.arg.work_dir}/config.yaml', 'w') as f:
            yaml.dump(vars(self.arg), f, default_flow_style=False)
        self.recoder = self.recoder if hasattr(self, 'recoder') else None
        if hasattr(self, 'recoder') and self.recoder is not None:
            self.recoder.print_log(f'Parameters:\n{vars(self.arg)}')


if __name__ == '__main__':
    sparser = get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        keys = vars(p).keys()
        for k in default_arg.keys():
            if k not in keys:
                raise ValueError(f'unrecognized config key: {k}')
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    Processor(args).start()
