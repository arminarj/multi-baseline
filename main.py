import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import torch.optim as optim

import numpy as np
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
import time
import logging
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import gzip
import numpy as np
import torch.nn as nn
import os
import time

from dataset import *
import Models 

from solver import Solver
import argparse

torch.set_default_tensor_type(torch.DoubleTensor)


parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MultimodalLeNet',
                    help='name of the model to use (MultimodalLeNet, etc.)')

parser.add_argument('--aligned', default=True,
                    help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_20seq_emo',
                    help='dataset to use (default: mosei_20seq_emo)')
parser.add_argument('--data_path', type=str, default='/content/multi-baseline/data',
                    help='path for storing the dataset')

# Dropouts
parser.add_argument('--dropout', type=float, default=0.3,
                    help='attention dropout')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')



# Tuning
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 400)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='number of epochs (default: 10)')
parser.add_argument('--when', type=int, default=10,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=24,
                    help='number of chunks per batch (default: 1)')
parser.add_argument('--reg_par', type=float, default=1e-5,
                    help='the regularization parameter of the network')
# Logistics
parser.add_argument('--log_interval', type=int, default=60,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='LiNet',
                    help='name of the trial (default: "LeNet")')
parser.add_argument('--nlevels', type=int, default=3,
                    help='n hidden layer')

parser.add_argument('--loss', type=str, default='BCELoss',
                    help='Model loss')
args = parser.parse_args()


output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8
}

criterion_dict = {
    'iemocap': 'CrossEntropyLoss'
}

use_cuda = False
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print("Using", torch.cuda.device_count(), "GPUs")
        use_cuda = True

####################################################################
#
# Load the dataset 
#
####################################################################

print("Start loading the data....")
dataset = args.dataset

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')
   
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = getattr(nn, hyp_params.loss)
hyp_params.device = 'cuda' if use_cuda else 'cpu'



if __name__ == '__main__':

    seq_number = 20
    input_shape1 = 300*seq_number
    input_shape2 = 74*seq_number
    input_shape3 = 35*seq_number
    
    model = getattr(Models, hyp_params.model)(input_shape1, input_shape2, input_shape3, seq_number).double()

    solver = Solver(model, hyp_params)

    results, truths = solver.fit(train_loader, valid_loader, test_loader)