import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

import socket
hname = socket.gethostname()
if hname[0:4] == 'andy':
    dataroot = '/home/mtayyab/visionnas_data/'
elif hname[0:6] == 'tayyab':
    dataroot = '/home/tayyab/visionnas_data/'
else:
    dataroot = '/home/mtayyab/visionnas_data/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='Newsplit | ECCVsplit | ShanghaiTech')
parser.add_argument('--outf', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--gpuId', type=int, required=True, help='The ID of the specified GPU')

parser.add_argument('--dataroot', default=dataroot, help='path to dataset')
parser.add_argument('--trainBatchSize', type=int, default=32, help='input train batch size')
parser.add_argument('--testBatchSize', type=int, default=32, help='input test batch size')
parser.add_argument('--displayAfter', type=int, default=50, help='print status after processing (n) batches')
parser.add_argument('--sampleSize', type=int, default=30000, help='sample size for samplar class')
parser.add_argument('--numEpochs', type=int, default=100, help='input number of epoch')
parser.add_argument('--netP', default='', help="path to net (to continue training)")

opt = parser.parse_args()
print(opt)