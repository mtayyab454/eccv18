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

######################################################################################################################

def loadCheckpoint(opt):
    if opt.netP:
        if os.path.isfile(opt.netP):
            print("=> loading checkpoint '{}'".format(opt.netP))
            checkpoint = torch.load(opt.netP)

            opt.startEpoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])

            print("=> loaded checkpoint '{}' (epoch {})".format(opt.netP, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opt.netP))

def saveCheckpoint(outf, model, scheduler, opt, epoch_num):

    cp = {'epoch': epoch_num,
        'model': model.state_dict(),
        'optimizer': scheduler.optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'options': opt,}

    torch.save(cp, outf + '/' + outf + '_' + str(epoch_num) + '.pth')

def imshow(img, count):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

######################################################################################################################

hname = socket.gethostname()
if hname[0:4] == 'andy':
    dataroot = '/home/andy/mtayyab/data/'
elif hname[0:6] == 'tayyab':
    dataroot = '/home/tayyab/crcv_data/'
else:
    dataroot = '/home/mtayyab/data/'

parser = argparse.ArgumentParser()
parser.add_argument('--outf', required=True, help='folder to output images and model checkpoints')

parser.add_argument('--gpuId', type=int, default=0, help='The ID of the specified GPU')

parser.add_argument('--trValFile', default='trainval_eccv', help='train val file to import')
parser.add_argument('--dsetFile', default='mydatasets.CCMatDataECCV', help='dataset class to import')
parser.add_argument('--netFile', default='myDenseNet_eccv', help='network file to import')

parser.add_argument('--dataset', default='UCF-QNRF-ECCV18', help='UCF-QNRF-ECCV18 | ShanghaiTech')
parser.add_argument('--trPatches', default='224112r', help='name of the folder containing training patches')
parser.add_argument('--tsPatches', default='224c', help='name of the folder containing testing patches')

parser.add_argument('--dataroot', default=dataroot, help='path to dataset')
parser.add_argument('--trBatchSize', type=int, default=24, help='input train batch size')
parser.add_argument('--tsBatchSize', type=int, default=24, help='input test batch size')
parser.add_argument('--displayAfter', type=int, default=50, help='print status after processing (n) batches')
parser.add_argument('--sampleSize', type=int, default=300000, help='sample size for samplar class')
parser.add_argument('--numEpochs', type=int, default=100, help='input number of epoch')
parser.add_argument('--netP', default='', help="path to net (to continue training)")
parser.add_argument('--graphDir', default='', help="path to write tensorboard graph")

# opt = parser.parse_args(['--outf', 'eccv18_smallsample', '--sampleSize', '10', '--trBatchSize', '2', '--displayAfter', '1'])
# opt = parser.parse_args(['--outf', 'eccv18_onech', '--netFile', 'myDenseNet_onech', '--dsetFile', 'mydatasets.CCMatDataOneCh'])
# opt = parser.parse_args(['--outf', 'eccv18_test'])
opt = parser.parse_args()

opt.startEpoch = 1
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

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.5, last_epoch=-1)

swriter = SummaryWriter(log_dir=opt.graphDir)
model.cuda()
######################################################################################################################

if os.path.exists(opt.outf) == False:
    os.makedirs(opt.outf)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

trainPatchPath = opt.dataroot + opt.dataset + '/Train/' + opt.trPatches
trainData = dataset(root_dir=trainPatchPath, transform=transform)
print(len(trainData))

testPatchPath = opt.dataroot + opt.dataset + '/Test/' + opt.tsPatches
testData = dataset(root_dir=testPatchPath, transform=transform)
print(len(testData))

trainData[99]
######################################################################################################################

# imshow(data[n, :, :, :], counts[n])
# data_iter = iter(train_loader)
# patch, patch_count, im_name, patch_name = data_iter.next()
# patch, count, c1, c2, c3, c4, img_name, patch_name = train_data[15]

######################################################################################################################
loadCheckpoint(opt)

for epoch in range(opt.startEpoch, opt.numEpochs):  # loop over the dataset multiple times

    trainval.train(opt.outf, model, trainData, opt.sampleSize, opt.trBatchSize, scheduler, swriter, opt.displayAfter, epoch)
    saveCheckpoint(opt.outf, model, scheduler, opt, epoch)
    trainval.test(opt.outf, model, testData, opt.tsBatchSize, swriter, opt.displayAfter, epoch)

print('Finished Training')
del model
swriter.close()