import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
from utils import *
import time
import logging
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import os
import time

from dataset import *
import torch.optim as optim

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


class Solver():
    def __init__(self, model, hyp_params):
        # self.model = nn.DataParallel(model)
        self.model = model.double()
        self.model = self.model.to(hyp_params.device)
        self.epoch_num = hyp_params.num_epochs
        self.batch_size = hyp_params.batch_size
        self.batch_chunk = hyp_params.batch_size
        self.loss = hyp_params.criterion
        self.optimizer = getattr(optim, hyp_params.optim)(
            self.model.parameters(), lr=hyp_params.lr, weight_decay=hyp_params.reg_par)
        self.device = hyp_params.device
        print(self.device)
        self.schedule = ReduceLROnPlateau(self.optimizer, mode='min', patience=hyp_params.when,
                                             factor=0.1, verbose=True)
        self.log_interval = hyp_params.log_interval

        # self.outdim_size1 = outdim_size1
        # self.outdim_size2 = outdim_size2

        self.train_loader = None
        self.test_loader = None
        self.valid_loader = None

        self.hyp_params = hyp_params

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("Lenet.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        self.logger.info(self.model)
        self.logger.info(self.optimizer)
        self.writer = SummaryWriter()
        self.epoch = -1

    def fit(self, train_loader, valid_loader, test_loader, checkpoint='checkpoint.model'):
        """

        x1, x2 are the vectors needs to be make correlated
        dim=[batch_size, feats]

        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader
        best_valid = 1e+08
        for epoch in range(1, self.epoch_num+1):
            self.epoch = epoch
            start = time.time()
            train_loss  = self.train(epoch)
            val_loss, _, _ = self.test(test=False)
            test_loss, _, _ = self.test(test=True)
            
            end = time.time()
            duration = end-start
            self.schedule.step(val_loss)    # Decay learning rate by validation loss
            print("-"*50)
            print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, train_loss, val_loss, test_loss))
            print("-"*50)

            if val_loss < best_valid:
                name = self.hyp_params.name
                save_model(self.hyp_params, self.model, name=name)
                print(f"Saved model at pre_trained_models/{name}.pt")
                best_valid = val_loss
        
        self.writer.close()
        self.model = load_model(self.hyp_params, name=self.hyp_params.name)
        _, results, truths = self.test(test=True)
        eval_mosei_senti(results, truths, True)
        return results, truths
        

        ####################################
        ###### functions defenition #######
        ####################################

    def train(self, epoch):
        model=self.model.to(self.device)
        optimizer=self.optimizer
        criterion=self.loss
        epoch_loss = 0
        device = self.device
        model.train()
        num_batches = self.hyp_params.n_train // self.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        batch_size = self.batch_size
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(self.train_loader):
            sample_ind, text, audio, vision = batch_X
            y = eval_attr = batch_Y.to(device).double().squeeze()  
            model.zero_grad()

            text, audio, vision = text.to(device).double(), audio.to(device).double(), vision.to(device).double()
            batch_chunk = text.size(0)
                
            combined_loss = 0
            # net = nn.DataParallel(model) if self.batch_size > 10 else model
            net = model
        
            
            raw_loss = combined_loss = 0
            text, audio, vision = text.unsqueeze(1), audio.unsqueeze(1), vision.unsqueeze(1)
            y_pred = net(text, audio, vision)
            raw_loss_i = criterion(y_pred, y)
            raw_loss += raw_loss_i
            assert torch.isnan(raw_loss).sum().item() == 0
            raw_loss_i.backward()
            raw_loss = raw_loss 
            combined_loss = raw_loss
            
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.hyp_params.clip)

            optimizer.step()

            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % self.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                    format(epoch, i_batch, num_batches, elapsed_time * 1000 / self.hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
                self.writer.add_graph(self.model, (text, audio, vision))

        for name, param in self.model.LeNet1.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.epoch)
        for name, param in self.model.LeNet2.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.epoch)
        for name, param in self.model.LeNet3.named_parameters():
            self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.epoch) 
        self.writer.add_scalar('Loss/train', avg_loss, self.epoch) 
        return avg_loss

    def evaluate(self, test=False, _loader=None, scoring=False):
        model=self.model.to(self.device)
        device = self.device
        criterion=self.loss    
        model.eval()
        if _loader is None:
            loader = self.test_loader if test else self.valid_loader
        else : loader = _loader
        total_loss = 0.0
        net = model
        outputs, losses, targets = [], [], []
        labels = []
        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                _, text, audio, vision = batch_X
                text, audio, vision = text.to(device).double(), audio.to(device).double().to(device).double(), vision.to(device).double()
                batch_chunk = text.size(0)
                
                y = eval_attr = batch_Y.squeeze(-1).to(device).double()
                text, audio, vision = text.unsqueeze(1), audio.unsqueeze(1), vision.unsqueeze(1)
                y_pred = net(text, audio, vision)
                y_pred = y_pred.squeeze()
                outputs.append(y_pred)
                targets.append(y)
                loss = criterion(y_pred, y)
                # total_loss += loss.item()
                losses.append(loss)
                # print(loss)
                # labels.append(eval_attr)
            
            # labels = torch.cat(labels, dim=0)
            outputs = torch.cat(outputs)
            targets = torch.cat(targets)
            losses = torch.tensor(losses).double()
        
        if loader is self.train_loader:
            loss_name="train_eval"
        elif test:
            loss_name='test_eval'
        else : loss_name ='valid_eval'
        # mat = np.concatenate([
        #         outputs[0].cpu().numpy(),
        #         outputs[1].cpu().numpy(),
        #     ], axis=1)
        # labels = np.concatenate([labels.data,labels.data], axis=1)
        # self.writer.add_embedding(mat,
        #                         metadata=labels,
        #                         global_step=self.epoch
        #                         # tag=["text", "audio"]
        #                         )
        if not scoring:
            self.writer.add_scalar(f'Loss/{loss_name}', losses.mean(), self.epoch)
        print(f'loss shape : {losses.shape}')
        return losses, outputs, targets

    def test(self, test=False, loader=None):
        losses, outputs, targets = self.evaluate(test=test, _loader=loader)
        return torch.mean(losses), outputs, targets


