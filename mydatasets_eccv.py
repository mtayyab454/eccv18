import torchvision.transforms as transforms
from torch.utils.data import Dataset
import scipy.io as sio
from skimage import io
import numpy as np
import torch
import os

def get_count(patch_path, gt_available):
    count = 0.0
    img_name = 'nill!'
    if gt_available:
        count = float(patch_path.split('_')[-1].split('.')[0])
        temp = patch_path[::-1].split('/')[0]
        id = [i for i, n in enumerate(temp) if n == '_'][1]
        temp = temp[id + 1:]
        img_name = temp[::-1]
    return count, img_name

class CC(Dataset):

    def __init__(self, root_dir, gt_available=True, transform=None):
        
        temp = root_dir[::-1].split('/', 1)
        folder_name = temp[0][::-1]
        dir_path = temp[1][::-1]
        
        text_file = open(dir_path + '/' + folder_name + '_fnames.txt', 'r')
        name_data = text_file.read()
        files = name_data.split('\n')
        
        if not files[-1]:
            files = files[0:-1]
        
        self.root_dir = root_dir
        self.transform = transform
        self.filetype = 'jpg'
        self.files = files
        self.gt_available = gt_available
        
    def __len__(self):
        return len(self.files)

class CCMatDataOneCh(CC):

    def __init__(self, root_dir, gt_available=True, transform=None):
        super(CCMatDataOneCh, self).__init__(root_dir, gt_available, transform)
        self.transform = transforms.Compose( [transforms.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        patch_path = os.path.join(self.root_dir, self.files[idx])
        patch_name = self.files[idx]
        patch = io.imread(patch_path)

        count, img_name = get_count(patch_path, self.gt_available)
        count = torch.FloatTensor([count])

        try:
            mat_data = sio.loadmat(patch_path[:-4:] + '.mat')
        except:
            print('Problem loading >>> ' + patch_path, idx)

        den = mat_data['d28']
        d1 = torch.FloatTensor(den[:, :, 0])
        d2 = torch.FloatTensor(den[:, :, 1])
        d3 = torch.FloatTensor(den[:, :, 2])

        temp = mat_data['d224']
        patch = temp[:, :, 2]
        patch = np.expand_dims(patch, axis=2)
        # patch = torch.unsqueeze(patch, 2)

        if self.transform is not None:
            patch = self.transform(patch)

        patch = patch.type(torch.FloatTensor)
        return patch, count, d1, d2, d3, img_name, patch_name

class CCMatDataECCV(CC):

    def __init__(self, root_dir, gt_available=True, transform=None):
        super(CCMatDataECCV,self).__init__(root_dir, gt_available, transform)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        patch_path = os.path.join(self.root_dir, self.files[idx])
        patch_name = self.files[idx]
        patch = io.imread(patch_path)

        count, img_name = get_count(patch_path, self.gt_available)
        count = torch.FloatTensor([count])

        try:
            mat_data = sio.loadmat(patch_path[:-4:] + '.mat')
        except:
            print('Problem loading >>> ' + patch_path, idx)

        den = mat_data['d28']
        d1 = torch.FloatTensor(den[:, :, 0])
        d2 = torch.FloatTensor(den[:, :, 1])
        d3 = torch.FloatTensor(den[:, :, 2])

        if self.transform is not None:
            patch = self.transform(patch)
        
        return patch, count, d1, d2, d3, img_name, patch_name  
    
class CCMatDataD201(CC):

    def __init__(self, root_dir, gt_available=True, transform=None):
        super(CCMatDataD201,self).__init__(root_dir, gt_available, transform)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        patch_path = os.path.join(self.root_dir, self.files[idx])
        patch_name = self.files[idx]
        patch = io.imread(patch_path)

        count, img_name = get_count(patch_path, self.gt_available)
        count = torch.FloatTensor([count])

        if self.transform is not None:
            patch = self.transform(patch)
        
        return patch, count, img_name, patch_name