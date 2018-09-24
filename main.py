import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import argparse
import socket
import torch
import os

import socket
hname = socket.gethostname()
if hname[0:4] == 'andy':
    dataroot = '/home/mtayyab/visionnas_data/'
elif hname[0:6] == 'tayyab':
    dataroot = '/home/tayyab/visionnas_data/'
else:
    dataroot = '/home/mtayyab/visionnas_data/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='UCF-QNRF-ECCV18 | ShanghaiTech')
parser.add_argument('--trPatchs', required=True, help='name of the folder containing training patches')
parser.add_argument('--tsPatches', required=True, help='name of the folder containing testing patches')
parser.add_argument('--outf', required=True, help='folder to output images and model checkpoints')
parser.add_argument('--gpuId', type=int, required=True, help='The ID of the specified GPU')

parser.add_argument('--trValFile', required=True, help='train val file to import')
parser.add_argument('--dsetFile', required=True, help='dataset class to import')
parser.add_argument('--netFile', required=True, help='network file to import')

parser.add_argument('--dataroot', default=dataroot, help='path to dataset')
parser.add_argument('--trainBatchSize', type=int, default=32, help='input train batch size')
parser.add_argument('--testBatchSize', type=int, default=32, help='input test batch size')
parser.add_argument('--displayAfter', type=int, default=50, help='print status after processing (n) batches')
parser.add_argument('--sampleSize', type=int, default=30000, help='sample size for samplar class')
parser.add_argument('--numEpochs', type=int, default=100, help='input number of epoch')
parser.add_argument('--netP', default='', help="path to net (to continue training)")
parser.add_argument('--graphDir', default='', help="path to write tensorboard graph")

opt = parser.parse_args()
print(opt)

trainval = __import__(opt.trValFile)
myNet = __import__(opt.netFile)

components = opt.dsetFile.split('.')
mod = __import__(components[0])
dataset = getattr(mod, components[1])

######################################################################################################################

model = myNet.DenseNet()

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuId)

if opt.graphDir == '':
    opt.graphDir = opt.outf + '/tensorboard'
else:
    opt.graphDir = opt.graphDir + '/' + opt.outf

if opt.netP != '':
    model.load_state_dict(torch.load(opt.netP))

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5, last_epoch=-1)

swriter = SummaryWriter(log_dir=opt.graphDir)
model.cuda()
######################################################################################################################

if os.path.exists(opt.outf) == False:
    os.makedirs(opt.outf)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

trainPatchPath = opt.dataroot + opt.dataset + '/Train/' + opt.trPatchs
trainData = dataset(root_dir=trainPatchPath, transform=transform)
print(len(trainData))

testPatchPath = opt.dataroot + opt.dataset + '/Test/' + opt.tsPatches
testData = dataset(root_dir=testPatchPath, transform=transform)
print(len(testData))

######################################################################################################################

# imshow(data[n, :, :, :], counts[n])
# data_iter = iter(train_loader)
# patch, patch_count, im_name, patch_name = data_iter.next()
# patch, count, c1, c2, c3, c4, img_name, patch_name = train_data[15]

######################################################################################################################

def imshow(img, count):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

######################################################################################################################

for epoch in range(opt.numEpochs):  # loop over the dataset multiple times

    trainval.train(opt.outf, model, trainData, opt.sampleSize, opt.trainBatchSize, scheduler, swriter, opt.displayAfter, epoch)
    trainval.test(opt.outf, model, testData, opt.testBatchSize, swriter, opt.displayAfter, epoch)

print('Finished Training')
del model
swriter.close()