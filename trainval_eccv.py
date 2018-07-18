from torch.autograd import Variable
import numpy as np
import torch

###############################################################################

def train(model_name, model, data, loader, scheduler, mse_criterion, ce_criterion, swriter, disp_after, epoch_num):
    model.train()
    print('\n')
    scheduler.step()
    print('Learning rate: ', scheduler.get_lr())
    log = '\n' + 'Learning rate: ' + str(scheduler.get_lr())
    batch_loss = 0.0
    batch_mae = 0.0
    
    loss_vec = []
    mae_vec = []
    
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
        mae = mae.data[0]

        batch_mae += mae
        mae_vec.append(mae)
        batch_loss += loss.data[0]
        loss_vec.append(loss.data[0])

        if itr % disp_after == disp_after-1:
            log_entery = ('[%d, %5d of %5d] Training bMSE: %5.3f MSE: %5.3f bMAE: %4.2f MAE: %4.2f ' %
                  (epoch_num+1, itr+1, len(loader), batch_loss/disp_after, np.mean(loss_vec), batch_mae/disp_after, np.mean(mae_vec) ))

#            swriter.add_scalars('data/training-batchwise', {'bMSE':batch_loss/disp_after, 'MSE':np.mean(loss_vec), 'bMAE':batch_mae/disp_after, 'MAE':np.mean(mae_vec)}, itr)
            
            batch_loss = 0.0
            batch_mae = 0.0
            log = log + '\n' + log_entery
            print(log_entery)
            
    model.eval()
    torch.save(model.state_dict(), model_name + '/' + model_name + '_%d.pth' % (epoch_num+1))
    info = {'mae_vec':mae_vec, 'loss_vec':loss_vec, 'log':log}
    
    swriter.add_scalars(model_name+'-train', {'loss':np.mean(loss_vec), 'mae':np.mean(mae_vec)}, epoch_num)
    
    return info

###############################################################################
    
def test(model_name, model, data, loader, criterion, swriter, disp_after, epoch_num):
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
    
    
    for itr, batch in enumerate(loader, 0):

        patches, counts, _, _, _, im_names, patch_names = batch        
        patches, counts = patches.cuda(), counts.cuda()
        patches, counts = Variable(patches, volatile=True), Variable(counts)

        outputs, _, _, _ = model(patches)
        loss = criterion(outputs, counts)

        mae = (outputs - counts).abs().mean()
        mae = mae.data[0]
        
        batch_mae += mae
        mae_vec.append(mae)
        batch_loss += loss.data[0]
        loss_vec.append(loss.data[0])
        
        temp_counts = counts.data.cpu().numpy()
        mat_counts.extend(temp_counts[:,0].tolist())
        
        temp_outputs = outputs.data.cpu().numpy()
        mat_outputs.extend(temp_outputs[:,0].tolist())
        mat_files.extend(im_names)
        mat_patches.extend(patch_names)
        
        if itr % disp_after == disp_after-1:
            log_entery = ('[%d, %5d of %5d] Testing bMSE: %5.3f MSE: %5.3f bMAE: %4.2f MAE: %4.2f ' %
                  (epoch_num+1, itr+1, len(loader), batch_loss/disp_after, np.mean(loss_vec), batch_mae/disp_after, np.mean(mae_vec) ))
            batch_loss = 0.0
            batch_mae = 0.0
            log = log + '\n' + log_entery
            print(log_entery)
            
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
    
    swriter.add_scalars(model_name+'-test', {'im_error':np.mean(im_error), 'loss':np.mean(loss_vec), 'mae':np.mean(mae_vec)}, epoch_num)    
    
    return info
