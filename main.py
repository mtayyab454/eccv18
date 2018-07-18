from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import socket
import torch
import scipy
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
parser.add_argument('--dataset', required=True, help='Newsplit | ECCVsplit | ShanghaiTech')
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
# dataset = __import__(opt.dsetFile)
myNet = __import__(opt.netFile)

components = opt.dsetFile.split('.')
mod = __import__(components[0])
dataset = getattr(mod, components[1])

model = myNet.DenseNet()

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpuId)

if opt.dataset == 'Newsplit':
    opt.dataset = 'UCF-QNRF-Newsplit'
elif opt.dataset == 'ECCVsplit':
    opt.dataset = 'UCF-QNRF-ECCVsplit'

if opt.graphDir == '':
    opt.graphDir = opt.outf + '/tensorboard'
else:
    opt.graphDir = opt.graphDir + '/' + opt.outf

if opt.netP != '':
    model.load_state_dict(torch.load(opt.netP))
print(model)

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5, last_epoch=-1)

model.train()
######################################################################################################################

if os.path.exists(opt.outf) == False:
    os.makedirs(opt.outf)

trainPatchPath = opt.dataroot + opt.dataset + '/Train/' + opt.trPatchs
trainData = dataset(root_dir=trainPatchPath, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))
print(len(trainData))

testPatchPath = opt.dataroot + opt.dataset + '/Test/' + opt.tsPatches
testData = dataset(root_dir=testPatchPath, transform=transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]))

testLoader = DataLoader(testData, batch_size=opt.testBatchSize, shuffle=False, num_workers=4, drop_last=False)
print(len(testData))

testData[99]
######################################################################################################################

swriter = SummaryWriter(log_dir=opt.graphDir)
model.cuda()

mseCriterion = nn.MSELoss()
mseCriterion.cuda()

ceCriterion = nn.CrossEntropyLoss()
ceCriterion.cuda()


######################################################################################################################

# imshow(data[n, :, :, :], counts[n])
# data_iter = iter(train_loader)
# patch, patch_count, im_name, patch_name = data_iter.next()
# patch, count, c1, c2, c3, c4, img_name, patch_name = train_data[15]

######################################################################################################################

def save_everything():
    scipy.io.savemat(opt.outf + '/' + opt.outf + '_%d.mat' % (epoch + 1),
                     mdict={'train_loss': train_loss_mat, 'train_mae': train_mae_mat, 'test_loss': test_loss_mat,
                            'test_mae': test_mae_mat, 'mat_counts': testInfo['mat_counts'],
                            'mat_outputs': testInfo['mat_outputs'], 'mat_files': testInfo['mat_files'],
                            'im_error': testInfo['im_error'], 'im_error_vec': im_error_vec, 'train_time': str(t1 - t0),
                            'test_time': str(t3 - t2), 'im_files': testInfo['im_files'],
                            'mat_patches': testInfo['mat_patches']})

    text_file = open(opt.outf + '/' + opt.outf + '_%d.txt' % (epoch + 1), "w")
    text_file.write(log)
    text_file.close()


def print_everything():
    train_time_str = '\nTraining time: ' + str(t1 - t0)
    test_time_str = '\nTraining time: ' + str(t3 - t2)
    logEntery = train_time_str + test_time_str + (
                '\nTest MAE (over images): %.3f \nTraining Loss: %.3f \nTest MAE: %.3f \n' % (
        np.mean(testInfo['im_error']), np.mean(train_info['loss_vec']), np.mean(testInfo['mae_vec'])))

    print
    logEntery
    return logEntery

def get_pltimg(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0));


def imshow(img, count):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    print
    count

######################################################################################################################

train_loss_mat = []
train_mae_mat = []

test_loss_mat = []
test_mae_mat = []

im_error_vec = []
log = str()

for epoch in range(opt.numEpochs):  # loop over the dataset multiple times

    temp = np.random.permutation(range(len(trainData)))
    idx = temp[0: min(opt.sampleSize, len(trainData))]
    samplar = torch.utils.data.sampler.SubsetRandomSampler(idx)
    train_loader = DataLoader(trainData, batch_size=opt.trainBatchSize, sampler=samplar, num_workers=4, drop_last=False)

    t0 = datetime.now()
    train_info = trainval.train(opt.outf, model, trainData, train_loader, scheduler, mseCriterion, ceCriterion,
                                swriter, opt.displayAfter, epoch)
    t1 = datetime.now()
    train_loss_mat.append(train_info['loss_vec'])
    train_mae_mat.append(train_info['mae_vec'])
    log = log + train_info['log'] + '\n'

    t2 = datetime.now()
    testInfo = trainval.test(opt.outf, model, testData, testLoader, mseCriterion, swriter, opt.displayAfter, epoch)
    t3 = datetime.now()
    test_loss_mat.append(testInfo['loss_vec'])
    test_mae_mat.append(testInfo['mae_vec'])
    log = log + testInfo['log'] + '\n'
    im_error_vec.append(np.mean(testInfo['im_error']))

    print
    im_error_vec

    log_entery = print_everything()
    log = log + log_entery
    save_everything()

print('Finished Training')
del model
swriter.close()