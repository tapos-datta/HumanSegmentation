import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os
import cv2
from PIL import Image

from data_loader import Rescale
from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET
from model import U2NETP

# ------- 1. define loss function --------

bce_loss = nn.BCELoss(reduction='mean')


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    loss0 = bce_loss(d0,labels_v)
    loss1 = bce_loss(d1,labels_v)
    loss2 = bce_loss(d2,labels_v)
    loss3 = bce_loss(d3,labels_v)
    loss4 = bce_loss(d4,labels_v)
    loss5 = bce_loss(d5,labels_v)
    loss6 = bce_loss(d6,labels_v)

    return (loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6), loss0

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def normPRED_batch(d_batch):
    for dn in range(0, d_batch.shape[0]):
        d_batch[dn] = normPRED(d_batch[dn])
    return d_batch

def save_blend_images(maskImg, originalImg, counter, output_dir):
    for indx in range(0, originalImg.shape[0]):
        oImg = originalImg[indx].transpose((1, 2, 0))
        blend = oImg.copy()
        mask = maskImg[indx].transpose((1,2,0))
        mask = mask[:,:,0].copy()
        blend[np.where(mask > 127)] = [206,55,230]
        oImg = cv2.addWeighted(oImg, 0.4, blend, 0.6, 0.0)
        im_pil = Image.fromarray(oImg)
        im_pil.save(output_dir + str(counter) + '.jpg')
        counter += 1

        del oImg, blend, mask, im_pil


def validate( model, validate_salobj_dataloader, output_dir):
    #switch mode to evaluate
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    running_val_loss = 0.0
    d0_val_loss = 0.0
    with torch.no_grad():
        
        data_iter = iter(validate_salobj_dataloader)
        
        next_batch = next(data_iter) # start loading the first batch
        inputs, labels = next_batch['image'], next_batch['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        next_batch_img, next_batch_lab =  Variable(inputs.cuda(non_blocking = True), requires_grad=False), Variable(labels.cuda(non_blocking= True),
                                                                                    requires_grad=False)
        iteration = 0
        _batches = []
        _g_precision = []
        _g_recall = []
        _g_f1_score = []
        counter = 0
        for i in range(len(validate_salobj_dataloader)):
            batch_img = next_batch_img
            batch_lab = next_batch_lab
            if i + 2 != len(validate_salobj_dataloader): 
                # start copying data of next batch
                next_batch = next(data_iter)
                inputs, labels = next_batch['image'], next_batch['label']
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
                next_batch_img, next_batch_lab =  Variable(inputs.cuda(non_blocking = True), requires_grad=False), Variable(labels.cuda(non_blocking = True), requires_grad=False)


            # forward
            d0, d1, d2, d3, d4, d5, d6 = model(batch_img)
            
            #calculate loss
            val_loss, d0_loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, batch_lab)
            
    
            # # print statistics
            running_val_loss += val_loss.data.item()
            d0_val_loss += d0_loss.data.item()

            ## measuring f1-score
            res = normPRED_batch(d0) * 255
            res = res.cpu().data.numpy()
            res = np.array(res, dtype=np.uint8)
            
            lab = normPRED_batch(batch_lab) * 255
            lab = lab.cpu().data.numpy()
            lab = np.array(lab, dtype=np.uint8)

            org = normPRED_batch(batch_img) * 255
            org = org.cpu().data.numpy()
            org = np.array(org, dtype=np.uint8)

            # save blend result
            save_blend_images(res, org, counter, output_dir)
            counter += res.shape[0] 

            _ac_tp = 0;
            _ac_fn = 0;
            _ac_fp = 0;
            
            _batches.append(lab.shape[0])
            
            for ind in range(0, lab.shape[0]):
                _lab = lab[ind]
                actual_positive = np.sum(_lab > 127)

                _TP = np.sum(np.logical_and(_lab > 127, res[ind] > 127))
                _FN = np.sum(np.logical_and(_lab  > 127 , res[ind]  <= 127))
                _FP = np.sum(np.logical_and(_lab  <= 127 , res[ind]  > 127))
                
                if(_TP / actual_positive >= 0.95):
                    _ac_tp += 1
                elif (_FN >= _FP):
                    _ac_fn += 1
                else:
                    _ac_fp += 1
            if _ac_tp + _ac_fp > 0.01:
                precision = _ac_tp / (_ac_tp + _ac_fp)
            else:
                precision = 0.0
            
            if _ac_tp + _ac_fn > 0.01:
                recall = _ac_tp / (_ac_tp + _ac_fn)
            else:
                recall = 0.0

            _g_precision.append( precision)
            _g_recall.append( recall)
            if precision + recall > 0.01:
                f1_score = 2.0 * ((precision * recall) / (precision + recall))
            else:
                f1_score = 0.0
            _g_f1_score.append(f1_score)
            
            iteration += 1
            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, val_loss
        
        _g_f1_score = np.array(_g_f1_score)
        _g_precision = np.array(_g_precision)
        _g_recall = np.array(_g_recall)
        total_sample = np.sum(_batches)
        f1_weighted_avg = np.sum(_g_f1_score * _batches) / total_sample
        precision_weighted_avg = np.sum(_g_precision * _batches) / total_sample
        recall_weighted_avg = np.sum(_g_recall * _batches) / total_sample
        print("Validation batch %3f , running loss %3f \n" % ( running_val_loss / iteration, d0_val_loss / iteration))
        
    return running_val_loss / iteration , d0_val_loss / iteration , f1_weighted_avg, precision_weighted_avg, recall_weighted_avg




def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    if(torch.cuda.is_available()):
        model.cuda()
    
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])

    # return model, optimizer, epoch value
    return model, optimizer, checkpoint['epoch']




def save_model(epoch, model, optimizer, running_train_loss, tr_loss, run_val_loss, val_loss, precision=0.0, recall=0.0, f1_score=0.0):

    file1 = open(log_writter, "a+")  # append mode
    file1.write("%3f , %3f , %3f, %3f \n" %(running_train_loss, tr_loss, run_val_loss, val_loss))
    file1.close()   
    
    checkpoint = {
        'epoch': epoch + 1,
        'tr_loss_G': running_train_loss,
        'tr_loss_D': tr_loss,
        'val_loss_G': run_val_loss,
        'val_loss_D': val_loss,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'f1-score' : f1_score,
        'precision': precision,
        'recall': recall
    }

    torch.save(checkpoint, model_dir + model_name+"_ckp_ep_%d_tr_loss_%3f_td0_loss_%3f_va_loss_%3f_td0_loss_%3f_f1_%3f.ckpt" % (epoch + 1, running_train_loss, tr_loss, run_val_loss, val_loss, f1_score))

def find_image_path(src_dir, file_name) -> str:
    f_name = os.path.splitext(file_name)
    ext = f_name[-1]
    f_name = f_name[0]
    full_path = os.path.join(src_dir, f_name + ext)
    if os.path.exists(full_path):
        return full_path
    else:
        return os.path.join(src_dir, f_name + '.png')
    

if __name__ == '__main__':
    
    # ------- 2. set the directory of training dataset --------
    
    model_name = 'u2netp'
    
    log_writter =  os.path.join ("./", 'log' + os.sep)
    log_writter =  os.path.join (log_writter , 'val_log.txt')
    
    val_data_dir = os.path.join ('./', 'val_data' + os.sep)
    val_image_dir = os.path.join('imgs' + os.sep)
    val_label_dir = os.path.join('masks' + os.sep)
    
    
    model_dir = './selected_models/'
    
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    # model_dir = os.path.join(model_dir, 'u2netp_bce_itr_6000_train_0.210024_tar_0.210024' + '.pth')
    #saved_check_point = os.path.join(model_dir, 'u2net_ckp_ep_86_tr_loss_0.137484_td0_loss_0.013057_va_loss_0.000000_td0_loss_0.000000_f1_0.000000' + '.ckpt')
    
    batch_size_val = 20
    val_num = 0
    
    val_img_name_list = glob.glob(val_data_dir + val_image_dir + '*')
    
    val_lbl_name_list = []
    for img_path in val_img_name_list:
        img_name = img_path.split(os.sep)[-1]
        val_lbl_name_list.append(find_image_path(val_data_dir + val_label_dir, img_name))
    
    print("---")
    
    print("test images: ", len(val_img_name_list))
    print("test labels: ", len(val_lbl_name_list))
    print("---")
    
    
    val_salobj_dataset = SalObjDataset(img_name_list = val_img_name_list,
                                        lbl_name_list = val_lbl_name_list,
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    val_salobj_dataloader = DataLoader(val_salobj_dataset,
                                        batch_size=batch_size_val,
                                        shuffle=False,
                                        num_workers=1,
                                        pin_memory=True)
    
   
    
    # ------- 3. define model --------
    # define the net 
    if(model_name=='u2netp'):
        net = U2NETP(3,1)
    else:
        net = U2NET(3,1)
     
    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0)
    
    print("---start validation...")
    
    for (root,dir,files) in os.walk(model_dir, topdown=False):
        
        for ckpt in files:
            if(model_name=='u2netp'):
                net = U2NETP(3,1)
            else:
                net = U2NET(3,1)
            ckpt_path = os.path.join(model_dir + ckpt)
            _, _, epoch = load_ckp(ckpt_path, net, optimizer)
            out_dir = os.path.join(val_data_dir + os.sep, ckpt.split('.')[0] + os.sep)
            
            val_loss, d0_val_loss, f1_score, precision, recall = validate(net, validate_salobj_dataloader=val_salobj_dataloader, output_dir=out_dir)

            file = open(log_writter, "a+")  # append mode
            file.write(f'{val_loss} , {d0_val_loss} , {f1_score} , {precision} , {recall}, {ckpt}\n')
            file.close()


            del net, file

        
    
