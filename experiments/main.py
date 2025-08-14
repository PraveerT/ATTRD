import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import sys
import pdb
import yaml
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm

sys.path.append("../..")
from utils import get_parser, import_class, GpuDataParallel, Optimizer, Recorder, Stat, RandomState


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = RandomState(seed=self.arg.random_seed)
        self.device = GpuDataParallel()
        self.recoder = Recorder(self.arg.work_dir, self.arg.print_log)
        self.data_loader = {}
        self.topk = (1, 5)
        self.stat = Stat(self.arg.model_args['num_classes'], self.topk)
        self.model, self.optimizer = self.Loading()
        self.loss = self.criterion()
        
        # Check if pts_size was explicitly provided via command line
        # by checking if --pts-size appears in sys.argv
        self.use_static_pts = '--pts-size' in sys.argv

    def criterion(self):
        # Add label smoothing for regularization
        loss = nn.CrossEntropyLoss(label_smoothing=0.1, reduction="none")
        return self.device.criterion_to_device(loss)

    def train(self, epoch):
        self.model.train()
        
        # Check if pts_size was provided as command line argument
        if self.use_static_pts:
            # Use static pts_size from command line
            pts_size = self.arg.pts_size
            self.model.pts_size = pts_size
            # Also update model_args to ensure consistency
            self.arg.model_args['pts_size'] = pts_size
            self.recoder.print_log('Training epoch: {} | pts_size: {} (static from --pts-size)'.format(epoch + 1, pts_size))
        else:
            # Dynamic pts_size scheduling
            # Epoch 0-50: 96 -> 128 (slow increase)
            # Epoch 50-100: 128 -> 256 (fast increase)
            if epoch < 50:
                # Slow linear increase from 96 to 128 over 50 epochs
                pts_size = int(48 + (128 - 48) * (epoch / 50))
            elif epoch < 100:
                # Fast exponential-like increase from 128 to 256 over 50 epochs
                progress = (epoch - 50) / 50  # 0 to 1
                # Use quadratic progression for faster increase
                pts_size = int(128 + (256 - 128) * (progress ** 2))
            else:
                # Keep at maximum after epoch 100
                pts_size = 256
            
            # Update model's pts_size
            self.model.pts_size = pts_size
            self.recoder.print_log('Training epoch: {} | pts_size: {} (dynamic)'.format(epoch + 1, pts_size))
        
        loader = self.data_loader['train']
        loss_value = []
        temporal_loss_values = []
        spatial_loss_values = []
        correct = 0
        total = 0
        self.recoder.timer_reset()
        current_learning_rate = [group['lr'] for group in self.optimizer.optimizer.param_groups]
        
        # Add progress bar for training batches
        loader_with_progress = tqdm(enumerate(loader), total=len(loader), 
                                   desc=f"Epoch {epoch+1}", leave=False)
        
        for batch_idx, data in loader_with_progress:
            self.recoder.record_timer("dataloader")
            image = self.device.data_to_device(data[0])
            label = self.device.data_to_device(data[1])
            self.recoder.record_timer("device")
            output = self.model(image)
            self.recoder.record_timer("forward")
            loss = torch.mean(self.loss(output, label))
            
            # Compute separate branch losses for monitoring
            if hasattr(self.model, 'temporal_logits') and hasattr(self.model, 'spatial_logits'):
                with torch.no_grad():
                    temporal_loss = torch.mean(self.loss(self.model.temporal_logits, label))
                    spatial_loss = torch.mean(self.loss(self.model.spatial_logits, label))
                    
                    # Track separate losses for epoch mean calculation
                    temporal_loss_values.append(temporal_loss.item())
                    spatial_loss_values.append(spatial_loss.item())
                    
                    # Print branch losses every 50 batches
                    if batch_idx % 50 == 0:
                        # Get current learning rates
                        temporal_lr = self.optimizer.optimizer.param_groups[0]['lr']
                        spatial_lr = self.optimizer.optimizer.param_groups[1]['lr'] if len(self.optimizer.optimizer.param_groups) > 1 else temporal_lr
                        print(f"\n[Branch Losses] Temporal: {temporal_loss.item():.4f} (lr={temporal_lr:.6f}), "
                              f"Spatial: {spatial_loss.item():.4f} (lr={spatial_lr:.6f}), "
                              f"Combined: {loss.item():.4f}, "
                              f"Alpha: {getattr(self.model, 'alpha_value', 'N/A')}")
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.recoder.record_timer("backward")
            loss_value.append(loss.item())
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += label.size(0)
            current_acc = 100. * correct / total
            
            # Update progress bar
            loader_with_progress.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{current_learning_rate[0]:.6f}',
                'PTS': pts_size
            })
            
            if batch_idx % self.arg.log_interval == 0:
                # self.viz.append_loss(epoch * len(loader) + batch_idx, loss.item())
                self.recoder.print_log(
                    '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                        .format(epoch, batch_idx, len(loader), loss.item(), current_learning_rate[0]))
                self.recoder.print_time_statistics()
        self.optimizer.scheduler.step()
        self.recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
        
        # Print separate branch loss means if available
        if temporal_loss_values and spatial_loss_values:
            self.recoder.print_log('\tMean temporal loss: {:.10f}.'.format(np.mean(temporal_loss_values)))
            self.recoder.print_log('\tMean spatial loss: {:.10f}.'.format(np.mean(spatial_loss_values)))

    def eval(self, loader_name):
        self.model.eval()
        for l_name in loader_name:
            loader = self.data_loader[l_name]
            loss_mean = []
            for batch_idx, data in enumerate(loader):
                image = self.device.data_to_device(data[0])
                label = self.device.data_to_device(data[1])
                # Cal = CalculateParasAndFLOPs()
                # Cal.reset()
                # Cal.calculate_all(self.model, image)
                with torch.no_grad():
                    output = self.model(image)
                # loss = torch.mean(self.loss(output, label))
                loss_mean += self.loss(output, label).cpu().detach().numpy().tolist()
                self.stat.update_accuracy(output.data.cpu(), label.cpu(), topk=self.topk)
            self.recoder.print_log('mean loss: ' + str(np.mean(loss_mean)))

    def Loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        if self.arg.model:
            model_class = import_class(self.arg.model)
            # Override pts_size in model_args if provided via command line
            if '--pts-size' in sys.argv:
                self.arg.model_args['pts_size'] = self.arg.pts_size
                print(f"Using pts_size={self.arg.pts_size} from command line")
            model = self.device.model_to_device(model_class(**self.arg.model_args))
            if self.arg.weights:
                try:
                    print("Loading pretrained model...")
                    state_dict = torch.load(self.arg.weights)
                    for w in self.arg.ignore_weights:
                        if state_dict.pop(w, None) is not None:
                            print('Sucessfully Remove Weights: {}.'.format(w))
                        else:
                            print('Can Not Remove Weights: {}.'.format(w))
                    model.load_state_dict(state_dict, strict=True)
                    optimizer = Optimizer(model, self.arg.optimizer_args)
                except RuntimeError:
                    print("Loading from checkpoint...")
                    state_dict = torch.load(self.arg.weights)
                    self.rng.set_rng_state(state_dict['rng_state'])
                    self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
                    self.recoder.print_log("Resuming from checkpoint: epoch {}".
                                           format(self.arg.optimizer_args['start_epoch']))
                    model = self.device.load_weights(model, self.arg.weights, self.arg.ignore_weights)
                    optimizer = Optimizer(model, self.arg.optimizer_args)
                    optimizer.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                    optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])
            else:
                optimizer = Optimizer(model, self.arg.optimizer_args)
        else:
            raise ValueError("No Models.")
        print("Loading model finished.")
        self.load_data()
        return model, optimizer

    def load_data(self):
        print("Loading data")
        Feeder = import_class(self.arg.dataloader)
        self.data_loader = dict()
        if self.arg.train_loader_args != {}:
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_loader_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.arg.num_worker,
            )
        if self.arg.valid_loader_args != {}:
            self.data_loader['valid'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.valid_loader_args),
                batch_size=self.arg.test_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.arg.num_worker,
            )
        if self.arg.test_loader_args != {}:
            test_dataset = Feeder(**self.arg.test_loader_args)
            self.stat.test_size = len(test_dataset)
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=self.arg.test_batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=self.arg.num_worker,
            )
        print("Loading data finished.")

    def start(self):
        if self.arg.phase == 'train':
            self.recoder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or \
                             (epoch + 1 == self.arg.num_epoch)
                eval_model = ((epoch + 1) % self.arg.eval_interval == 0) or \
                             (epoch + 1 == self.arg.num_epoch)
                self.train(epoch)
                if save_model:
                    model_path = '{}/epoch{}_model.pt'.format(self.arg.work_dir, epoch + 1)
                    self.save_model(epoch, self.model, self.optimizer, model_path)
                if eval_model:
                    if self.arg.valid_loader_args != {}:
                        self.stat.reset_statistic()
                        self.eval(loader_name=['valid'])
                        self.print_inf_log(epoch + 1, "Valid")
                    if self.arg.test_loader_args != {}:
                        self.stat.reset_statistic()
                        self.eval(loader_name=['test'])
                        self.print_inf_log(epoch + 1, "Test")
        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.weights))
            if self.arg.valid_loader_args != {}:
                self.stat.reset_statistic()
                self.eval(loader_name=['valid'])
                self.print_inf_log(self.arg.optimizer_args['start_epoch'], "Valid")
            if self.arg.test_loader_args != {}:
                self.stat.reset_statistic()
                self.eval(loader_name=['test'])
                self.print_inf_log(self.arg.optimizer_args['start_epoch'], "Test")
            self.recoder.print_log('Evaluation Done.\n')

    def print_inf_log(self, epoch, mode):
        static = self.stat.show_accuracy('{}/{}_confusion_mat'.format(self.arg.work_dir, mode))
        prec1 = static[str(self.topk[0])] / self.stat.test_size * 100
        prec5 = static[str(self.topk[1])] / self.stat.test_size * 100
        self.recoder.print_log("Epoch {}, {}, Evaluation: prec1 {:.4f}, prec5 {:.4f}".
                               format(epoch, mode, prec1, prec5),
                               '{}/{}.txt'.format(self.arg.work_dir, self.arg.phase))

    def save_model(self, epoch, model, optimizer, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.optimizer.state_dict(),
            'scheduler_state_dict': optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state()
        }, save_path)

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)


if __name__ == '__main__':
    sparser = get_parser()
    p = sparser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    processor = Processor(args)
    processor.start()
