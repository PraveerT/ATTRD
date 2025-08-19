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
import requests

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
        
        # Telegram Bot Configuration
        self.telegram_bot_token = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"
        self.telegram_chat_id = None
        self.best_accuracy = 0.0  # Track best accuracy within current run
        
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
        train_loss = np.mean(loss_value)
        train_acc = 100. * correct / total
        self.recoder.print_log('\tMean training loss: {:.10f}.'.format(train_loss))
        return train_acc, train_loss

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
                    # Test-Time Augmentation: average predictions from 3 runs
                    outputs = []
                    for _ in range(3):
                        output = self.model(image)
                        outputs.append(output)
                    output = torch.stack(outputs).mean(dim=0)
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
            # Send initial Telegram message to establish chat
            try:
                self.send_initial_telegram_message("🚀 Training started!")
            except:
                pass  # Ignore if we can't send the initial message
            
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                save_model = ((epoch + 1) % self.arg.save_interval == 0) or \
                             (epoch + 1 == self.arg.num_epoch)
                eval_model = ((epoch + 1) % self.arg.eval_interval == 0) or \
                             (epoch + 1 == self.arg.num_epoch)
                train_acc, train_loss = self.train(epoch)
                if save_model:
                    model_path = '{}/epoch{}_model.pt'.format(self.arg.work_dir, epoch + 1)
                    self.save_model(epoch, self.model, self.optimizer, model_path)
                if eval_model:
                    if self.arg.valid_loader_args != {}:
                        self.stat.reset_statistic()
                        self.eval(loader_name=['valid'])
                        self.print_inf_log(epoch + 1, "Valid", train_acc, train_loss)
                    if self.arg.test_loader_args != {}:
                        self.stat.reset_statistic()
                        self.eval(loader_name=['test'])
                        self.print_inf_log(epoch + 1, "Test", train_acc, train_loss)
        elif self.arg.phase == 'test':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.recoder.print_log('Model:   {}.'.format(self.arg.model))
            self.recoder.print_log('Weights: {}.'.format(self.arg.weights))
            # Send initial Telegram message to establish chat
            try:
                self.send_initial_telegram_message("🚀 Testing started!")
            except:
                pass  # Ignore if we can't send the initial message
            
            if self.arg.valid_loader_args != {}:
                self.stat.reset_statistic()
                self.eval(loader_name=['valid'])
                self.print_inf_log(self.arg.optimizer_args['start_epoch'], "Valid", None, None)
            if self.arg.test_loader_args != {}:
                self.stat.reset_statistic()
                self.eval(loader_name=['test'])
                self.print_inf_log(self.arg.optimizer_args['start_epoch'], "Test", None, None)
            self.recoder.print_log('Evaluation Done.\n')

    def print_inf_log(self, epoch, mode, train_acc=None, train_loss=None):
        static = self.stat.show_accuracy('{}/{}_confusion_mat'.format(self.arg.work_dir, mode))
        prec1 = static[str(self.topk[0])] / self.stat.test_size * 100
        prec5 = static[str(self.topk[1])] / self.stat.test_size * 100
        self.recoder.print_log("Epoch {}, {}, Evaluation: prec1 {:.4f}, prec5 {:.4f}".
                               format(epoch, mode, prec1, prec5),
                               '{}/{}.txt'.format(self.arg.work_dir, self.arg.phase))
        
        # Display confusion matrix
        try:
            import numpy as np
            cm = self.stat.confusion_mat
            if cm is not None:
                self.recoder.print_log(f"Confusion Matrix (epoch {epoch}, {mode}):")
                # Print a simplified version of the confusion matrix
                # Show only the diagonal elements (correct predictions) and some key stats
                diagonal = np.diag(cm)
                total_correct = np.sum(diagonal)
                total_samples = np.sum(cm)
                self.recoder.print_log(f"  Total Correct: {total_correct}/{total_samples}")
                self.recoder.print_log(f"  Overall Accuracy: {total_correct/total_samples*100:.2f}%")
        except Exception as e:
            self.recoder.print_log(f"Failed to display confusion matrix: {e}")
        
        # Send Telegram message with evaluation results
        try:
            # Check if this is a new best
            is_new_best = prec1 > self.best_accuracy
            if is_new_best:
                self.best_accuracy = prec1
            
            # Format message as: Train: train acc train loss Test: test acc test loss
            if train_acc is not None and train_loss is not None:
                message = f"📊 Epoch {epoch}\n"
                message += f"Train: {train_acc:.1f} {train_loss:.2f}\n"
                message += f"Test: {prec1:.1f} {prec5:.1f}"
                if is_new_best:
                    message += f" 🏆 New Best: {self.best_accuracy:.1f}%"
            else:
                # For test phase without training data
                message = f"📊 Epoch {epoch} {mode}\n"
                message += f"Test: {prec1:.1f} {prec5:.1f}\n"
                if is_new_best:
                    message += f"🏆 New Best: {self.best_accuracy:.1f}%\n"
            
            # Send message
            self.send_telegram_message(message)
        except Exception as e:
            self.recoder.print_log(f"Failed to send Telegram message: {e}")

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

    def get_telegram_chat_id(self):
        """Get chat ID from the most recent message to the bot"""
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates"
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            
            if data["ok"] and data["result"]:
                # Get the most recent message chat ID (no need for /start command)
                chat_id = data["result"][-1]["message"]["chat"]["id"]
                return chat_id
        except Exception as e:
            self.recoder.print_log(f"Failed to get Telegram chat ID: {e}")
        return None

    def send_telegram_message(self, message):
        """Send message to Telegram - simplified version"""
        try:
            # Just try to send the message - if there's no chat, it will fail silently
            url = f"https://api.telegram.org/bot{self.telegram_bot_token}/getUpdates"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if data["ok"] and data["result"]:
                # Get the most recent chat ID
                chat_id = data["result"][-1]["message"]["chat"]["id"]
                
                # Send the actual message
                send_url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
                send_data = {
                    "chat_id": chat_id,
                    "text": message,
                    "parse_mode": "HTML"
                }
                
                send_response = requests.post(send_url, data=send_data, timeout=10)
                return send_response.json()["ok"]
        except:
            # If anything fails, just silently continue without sending message
            pass
        return False


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
