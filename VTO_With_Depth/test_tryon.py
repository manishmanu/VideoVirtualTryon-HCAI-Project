import sys
import os

import threading
import torch
from torch.autograd import Variable
import torch.utils.data
from lr_scheduler import *
import numpy
import numpy as np
import pickle as pkl
from scipy.misc import imread, imsave


import numpy
from AverageMeter import  *
from loss_function import *
import datasets
import balancedsampler
import networks
from my_args import args
import Resblock
import datasets
import os

# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger")
# debugpy.wait_for_client()
# print("Attached! :)")



SAVED_MODEL = "tryon_checkpoints/tryon_epoch_res241_[0.03265]_[0.06273].pth"
res_folder = "results_tryon_model_250"
os.makedirs(res_folder, exist_ok=True)

def test():
    torch.manual_seed(args.seed)
    device = torch.cuda.current_device()
    input_channels = 19
    model = networks.__dict__['Try_on'](device=device, inp_channels=input_channels)

    # Save model results
    model_dict = model.state_dict()
    pretrained_dict = torch.load(SAVED_MODEL)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.cuda()
    model = model.eval() 

    # loop over the video files and save it over here
    video_name = "bj721d01m-k11"
    pkl_file_name = video_name + ".pkl"
    info_folder = "/home/vishnu1911/test2/dataset_generated_new/dataset_generated"
    mi_path = os.path.join(info_folder, video_name, "masked_input_images")
    pkl_dict = pkl.load(open(pkl_file_name, "rb" ))
    out_path = os.path.join(info_folder, video_name, "input_images")

            # read each image and also transform
    with torch.no_grad():
        for f in os.listdir(mi_path):
            im2 = f.split(".jpg")[0]
            if im2 not in pkl_dict:
                continue
            im1 = os.path.join(mi_path, f)
            im3 = os.path.join(out_path, f)
            
            im1, im3 = datasets.loader(im1, im3)
            im1 = torch.tensor(im1).unsqueeze(0).to(device)
            im3 = torch.tensor(im3).unsqueeze(0).to(device)
            im2 = torch.tensor(pkl_dict[im2]).to(device)
            # self.tup.append([im1, pkl_dict[im2].squeeze(), im3])
            res = model(im1, im2, im3)

            # saving image in folder
            save_path = os.path.join(res_folder, f)
            res_img = np.round(np.transpose(255.0 * res.squeeze().cpu().numpy(), (1, 2, 0))).astype(np.uint8)
            imsave(save_path, res_img)




if __name__ == '__main__':
    test()
