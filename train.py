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

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	# print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    if not os.path.exists(checkpoint_fpath):
        return model, optimizer, 0
    print(checkpoint_fpath)
    # load check point
    checkpoint = torch.load(checkpoint_fpath, map_location=torch.device('cuda'))
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    if(torch.cuda.is_available()):
        model.cuda()
    
    # initialize optimizer from checkpoint to optimizer
    
   

    # return model, optimizer, epoch value 
    return model, optimizer, checkpoint['epoch']

def convert_to_variable(tensorA, tensorB):
    
    tensorA = tensorA.type(torch.FloatTensor)
    tensorB = tensorB.type(torch.FloatTensor)
    
    if torch.cuda.is_available():
        return  Variable(tensorA.cuda(), requires_grad=False), Variable(tensorB.cuda(),requires_grad=False)
    else:
        return Variable(tensorA, requires_grad=False), Variable(tensorB, requires_grad=False)
    


if __name__ == '__main__':
    # ------- 2. set the directory of training dataset --------

    log_writter =  os.path.join ("./", 'log' + os.sep)
    log_writter =  os.path.join (log_writter , 'training_log.txt')

    model_name = 'u2netp' #'u2netp'

    data_dir = os.path.join("./", 'P3M-10k/train' + os.sep)
    tra_image_dir = 'images/'
    tra_label_dir = 'labels/'

    image_ext = '.jpg'
    label_ext = '.png'

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

    
    
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    print(model_dir)
    saved_check_point = os.path.join(model_dir, 'file_name' + '.ckpt') # update the saved check-point file name
    

    epoch_num = 100000
    batch_size_train = 20
    batch_size_val = 1
    train_num = 0
    val_num = 0

    tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)

    print("---")
    print("train images: ", len(tra_img_name_list))
    print("train labels: ", len(tra_lbl_name_list))
    print("---")

    train_num = len(tra_img_name_list)

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([
            RescaleT(320),
            RandomCrop(288),
            ToTensorLab(flag=0)]))
    salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=8,  pin_memory=True)

    # ------- 3. define model --------
    # define the net
    if(model_name=='u2net'):
        net = U2NET(3, 1)
    elif(model_name=='u2netp'):
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.cuda()

    # ------- 4. define optimizer --------
    print("---define optimizer...")
    optimizer = optim.Adam(net.parameters(), lr=0.0009, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    net, optimizer, _epoch = load_ckp(saved_check_point, net, optimizer)

    # ------- 5. training process --------
    print(f"---start training... \n From epoch {_epoch}\n")
    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0
    ite_num4val = 0
    save_frq = 1500 # save the model every 1500 iterations

    for epoch in range(_epoch, epoch_num):
        net.train()
        
        data_iter = iter(salobj_dataloader)
        
        next_batch = next(data_iter) # start loading the first batch
        inputs, labels = next_batch['image'], next_batch['label']
        
        next_batch_img, next_batch_lab = convert_to_variable(inputs, labels)

        for i in range(len(salobj_dataloader)):
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1
            
            batch_img = next_batch_img
            batch_lab = next_batch_lab
            
            if i + 2 != len(salobj_dataloader): 
                # start copying data of next batch
                next_batch = next(data_iter)
                inputs, labels = next_batch['image'], next_batch['label']
                next_batch_img, next_batch_lab = convert_to_variable(inputs, labels)

            # y zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            d0, d1, d2, d3, d4, d5, d6 = net(batch_img)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, batch_lab)

            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.data.item()
            running_tar_loss += loss2.data.item()

            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss

            print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

            if ite_num % save_frq == 0:
                saved_path = model_dir + model_name+"_epoch_%d_bce_itr_%d_train_%3f_tar_%3f.ckpt" % (epoch, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val)

                checkpoint = {
                    'epoch': epoch + 1,
                    'running_train_loss': running_tar_loss,
                    'train_loss': running_loss,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                
                torch.save(checkpoint, saved_path) 
                file1 = open(log_writter, "a+")  # append mode
                file1.write("%3f, %3f \n" %(running_loss / ite_num4val, running_tar_loss / ite_num4val))
                file1.close()  
                
                running_loss = 0.0
                running_tar_loss = 0.0
                net.train()  # resume train
                ite_num4val = 0

