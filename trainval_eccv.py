from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable
from datetime import datetime
import torch.nn as nn
import numpy as np
import torch

###############################################################################

mse_criterion = nn.MSELoss()
mse_criterion.cuda()

def train(outf, model, data, sample_size, batch_size, scheduler, swriter, disp_after, epoch_num):

    temp = np.random.permutation(range(len(data)))
    idx = temp[0: min(sample_size, len(data))]
    samplar = torch.utils.data.sampler.SubsetRandomSampler(idx)
    loader = DataLoader(data, batch_size=batch_size, sampler=samplar, num_workers=4, drop_last=False)

    model.train()
    scheduler.step()

    print('\n')
    print('Learning rate: ', scheduler.get_lr())
    log = '\n' + 'Learning rate: ' + str(scheduler.get_lr())

    batch_loss = 0.0
    batch_mae = 0.0
    loss_vec = []
    mae_vec = []

    t0 = datetime.now()
    for itr, batch in enumerate(loader, 0):
        patches, counts, d1, d2, d3, _, _ = batch
        patches, counts, d1, d2, d3 = patches.cuda(), counts.cuda(), d1.cuda(), d2.cuda(), d3.cuda()
        patches, counts, d1, d2, d3 = Variable(patches), Variable(counts), Variable(d1), Variable(d2), Variable(d3)
        
        scheduler.optimizer.zero_grad()

        outputs, o1, o2, o3 = model(patches)
        
        l0 = mse_criterion(outputs, counts) # l1 is MSE loss
        l1 = mse_criterion(o1, d1)
        l2 = mse_criterion(o2, d2)
        l3 = mse_criterion(o3, d3)
        
        loss = 0.001*l0 + l1 + l2 + l3        
        
        loss.backward()
        scheduler.optimizer.step()

        mae = (outputs - counts).abs().mean()
        mae = mae.item()

        batch_mae += mae
        mae_vec.append(mae)
        batch_loss += loss.item()
        loss_vec.append(loss.item())

        if itr % disp_after == disp_after-1:
            log_entery = ('[%d, %5d of %5d] Training bMSE: %5.3f MSE: %5.3f bMAE: %4.2f MAE: %4.2f ' % (epoch_num,
                itr+1, len(loader), batch_loss/disp_after, np.mean(loss_vec), batch_mae/disp_after, np.mean(mae_vec) ))

            swriter.add_scalars(outf+'-train-bw', {'bMSE':batch_loss/disp_after, 'MSE':np.mean(loss_vec),
                'bMAE':batch_mae/disp_after, 'MAE':np.mean(mae_vec)}, ((epoch_num-1)*loader.__len__()) + itr)

            batch_loss = 0.0
            batch_mae = 0.0
            log = log + '\n' + log_entery
            print(log_entery)

    t1 = datetime.now()

    swriter.add_scalars(outf+'-train', {'loss':np.mean(loss_vec), 'mae':np.mean(mae_vec)}, epoch_num)
    swriter.add_text(outf+'-train-log', log, epoch_num)
    swriter.add_text(outf+'-train-time', str(t1-t0), epoch_num)

    model.eval()

    return model

###############################################################################

def test(outf, model, data, batch_size, swriter, disp_after, epoch_num):

    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    model.eval()
    print('\n')
    log = str();    
    batch_loss = 0.0
    batch_mae = 0.0
    
    loss_vec = []
    mae_vec = []
    
    mat_counts = []
    mat_outputs = []
    mat_files = []
    mat_patches = []

    t0 = datetime.now()
    for itr, batch in enumerate(loader, 0):

        patches, counts, _, _, _, im_names, patch_names = batch        
        patches, counts = patches.cuda(), counts.cuda()
        patches, counts = Variable(patches), Variable(counts)

        outputs, _, _, _ = model(patches)
        loss = mse_criterion(outputs, counts)

        mae = (outputs - counts).abs().mean()
        mae = mae.item()
        
        batch_mae += mae
        mae_vec.append(mae)
        batch_loss += loss.item()
        loss_vec.append(loss.item())
        
        temp_counts = counts.data.cpu().numpy()
        mat_counts.extend(temp_counts[:,0].tolist())
        
        temp_outputs = outputs.data.cpu().numpy()
        mat_outputs.extend(temp_outputs[:,0].tolist())
        mat_files.extend(im_names)
        mat_patches.extend(patch_names)
        
        if itr % disp_after == disp_after-1:
            log_entery = ('[%d, %5d of %5d] Testing bMSE: %5.3f MSE: %5.3f bMAE: %4.2f MAE: %4.2f ' % (epoch_num,
                itr+1, len(loader), batch_loss/disp_after, np.mean(loss_vec), batch_mae/disp_after, np.mean(mae_vec) ))

            swriter.add_scalars(outf + '-test-bw', {'bMSE': batch_loss / disp_after, 'MSE': np.mean(loss_vec),
                'bMAE': batch_mae / disp_after, 'MAE': np.mean(mae_vec)}, ((epoch_num-1) * loader.__len__()) + itr)
            batch_loss = 0.0
            batch_mae = 0.0
            log = log + '\n' + log_entery
            print(log_entery)

    t1 = datetime.now()

    u_files = set(mat_files)
    im_error = []
    for ii, f in enumerate(u_files):
        g_counts = 0.0
        e_counts = 0.0
        
        indices = [jj for jj, x in enumerate(mat_files) if x == f]
        for kk in indices:
            g_counts += mat_counts[kk]
            e_counts += mat_outputs[kk]
        im_error.append(abs(g_counts-e_counts))

    info = {'im_error':im_error, 'im_files':list(u_files), 'mae_vec':mae_vec, 'loss_vec':loss_vec, 
            'mat_counts':mat_counts, 'mat_outputs':mat_outputs, 'mat_files':mat_files, 'mat_patches':mat_patches, 'log':log}
    
    swriter.add_scalars(outf+'-test', {'im_error':np.mean(im_error), 'loss':np.mean(loss_vec), 'mae':np.mean(mae_vec)}, epoch_num)
    swriter.add_text(outf+'-test-log', log, epoch_num)
    swriter.add_text(outf+'-train-time', str(t1-t0), epoch_num)
    
    return info
