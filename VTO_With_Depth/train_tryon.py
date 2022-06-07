import sys
import os

import threading
import torch
from torch.autograd import Variable
import torch.utils.data
from lr_scheduler import *

import numpy
from AverageMeter import  *
from loss_function import *
import datasets
import balancedsampler
import networks
from my_args import args
import Resblock

# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger")
# debugpy.wait_for_client()
# print("Attached! :)")



def train():
    torch.manual_seed(args.seed)
    device = torch.cuda.current_device()
    input_channels = 19
    model = networks.__dict__['Try_on'](device=device, inp_channels=input_channels)
    if args.use_cuda:
        print("Turn the model into CUDA")
        model = model.cuda()

    
    train_set = datasets.__dict__['VITON_Data'](train=True)
    val_set = datasets.__dict__['VITON_Data'](train=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size = 16,
        num_workers= args.workers, pin_memory=True if args.use_cuda else False)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size = 4,
        num_workers= args.workers, pin_memory=True if args.use_cuda else False)
    optimizer = torch.optim.Adamax([
                {'params': model.conv1.parameters(), 'lr': 0.001},
                {'params': model.conv2.parameters(), 'lr':0.001},
                {'params': model.residue_net.parameters(), 'lr': args.rectify_lr}
            ],
                lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer, 'min',factor=args.factor, patience=args.patience,verbose=True)
    training_losses = AverageMeter()
    val_losses = AverageMeter()
    print("*********Start Training********")
    print("LR is: "+ str(float(optimizer.param_groups[0]['lr'])))
    print("EPOCH is: "+ str(int(len(train_set) / args.batch_size )))
    print("Num of EPOCH is: "+ str(args.numEpoch))

    for t in range(250):
        print("Training for epoch: ", str(t+1))
        model = model.train()
        
        # import sys
        # sys.exit()
        for i, (inp_1, inp_2, inp_3) in enumerate(train_loader):
            diffs, res = model(inp_1.to(device), inp_2.to(device), inp_3.to(device)) 
            #  [negPSNR_loss(diff, args.epsilon) for diff in diffs]
            pixel_loss = [charbonier_loss(diff, args.epsilon) for diff in diffs]
            # total_loss = sum(x*y if x > 0 else 0 for x,y in zip(args.alpha, pixel_loss))
            total_loss = sum(pixel_loss)
            training_losses.update(total_loss.item(), args.batch_size)
            if i % 10 == 0:
                print("\tAvg. Loss: " + str([round(training_losses.avg, 5)]  ))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        for i, (inp_1, inp_2, inp_3) in enumerate(val_loader):
            diffs, res = model(inp_1.to(device), inp_2.to(device), inp_3.to(device)) 
            #  [negPSNR_loss(diff, args.epsilon) for diff in diffs]
            pixel_loss = [charbonier_loss(diff, args.epsilon) for diff in diffs]
            # total_loss = sum(x*y if x > 0 else 0 for x,y in zip(args.alpha, pixel_loss))
            total_loss = sum(pixel_loss)
            val_losses.update(total_loss.item(), args.batch_size)
        
        print("Average val loss = "+str([round(training_losses.avg, 5)]))
        
        if t % 10 == 0:
            print("Inside saving")
            save_file_name = os.path.join("tryon_checkpoints", "tryon_epoch_res" + str(t+1) + "_" + str([round(training_losses.avg, 5)]) + "_" + str([round(val_losses.avg, 5)]) + ".pth")
            print(save_file_name)
            torch.save(model.state_dict(), save_file_name)

        training_losses.reset()
        scheduler.step(val_losses.avg)
       


if __name__ == '__main__':
    train()
