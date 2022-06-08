import time
import os
import sys
from os.path import join as osp
from os.path import exists as ose
from torch.autograd import Variable
import math
import torch
import cv2 

import random
import numpy as np
import numpy
import networks
from my_args import  args
import pickle

from scipy.misc import imread, imsave
from AverageMeter import  *

# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger")
# debugpy.wait_for_client()
# print("Attached! :)")

torch.backends.cudnn.benchmark = True # to speed up the
pickle_dp = "pickle_files"
os.makedirs("pickle_files", exist_ok=True)
data_path = "/home/vishnu1911/test2/dataset_gen/dataset_generated"
vid_names = os.listdir(data_path)
ctr = 0

def save_img(name, res):
    res_ = np.round(np.transpose(255.0 * res.unsqueeze(0).cpu().numpy(),(1, 2,0))).astype(np.uint8)
    print(res_.shape)
    print(cv2.imwrite(name, res_))


# DO_MiddleBurryOther = True
# MB_Other_DATA = "./MiddleBurySet/other-data/"
# MB_Other_RESULT = "./MiddleBurySet/other-result-author/"
# MB_Other_GT = "./MiddleBurySet/other-gt-interp/"
# if not os.path.exists(MB_Other_RESULT):
#     os.mkdir(MB_Other_RESULT)



model = networks.__dict__['DAIN_small'](channel=args.channels,
                            filter_size = args.filter_size ,
                            timestep=args.time_step,
                            training=False)

if args.use_cuda:
    model = model.cuda()

args.SAVED_MODEL = './model_weights/best.pth'
if os.path.exists(args.SAVED_MODEL):
    print("The testing model weight is: " + args.SAVED_MODEL)
    if not args.use_cuda:
        pretrained_dict = torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(args.SAVED_MODEL, map_location=lambda storage, loc: storage))
    else:
        pretrained_dict = torch.load(args.SAVED_MODEL)
        # model.load_state_dict(torch.load(args.SAVED_MODEL))

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    # 4. release the pretrained dict for saving memory
    pretrained_dict = []
else:
    print("*****************************************************************")
    print("**** We don't load any trained weights **************************")
    print("*****************************************************************")

model = model.eval() # deploy mode

img_file_name1='frame_1.jpg'
img_file_name2='frame_3.jpg'


use_cuda=args.use_cuda
save_which=args.save_which
dtype = args.dtype
unique_id =str(random.randint(0, 100000))
print("The unique id for current testing is: " + str(unique_id))

arguments_strFirst = img_file_name1
arguments_strSecond = img_file_name2

X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype) # shape 3, 480, 640
X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)

# arguments_strOut = os.path.join(gen_dir, dir, "frame10i11.png")
y_ = torch.FloatTensor()

assert (X0.size(1) == X1.size(1))
assert (X0.size(2) == X1.size(2))

intWidth = X0.size(2)
intHeight = X0.size(1)
channel = X0.size(0)
# if not channel == 3:
#     continue

if intWidth != ((intWidth >> 7) << 7):
    intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
    intPaddingLeft =int(( intWidth_pad - intWidth)/2)
    intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
else:
    intWidth_pad = intWidth
    intPaddingLeft = 32
    intPaddingRight= 32

if intHeight != ((intHeight >> 7) << 7):
    intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
    intPaddingTop = int((intHeight_pad - intHeight) / 2)
    intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
else:
    intHeight_pad = intHeight
    intPaddingTop = 32
    intPaddingBottom = 32

pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

torch.set_grad_enabled(False)
X0 = Variable(torch.unsqueeze(X0,0))
X1 = Variable(torch.unsqueeze(X1,0))
X0 = pader(X0)
X1 = pader(X1)

if use_cuda:
    X0 = X0.cuda()
    X1 = X1.cuda()
proc_end = time.time()

res = model(torch.stack((X0, X1),dim = 0))
# print(res[0].shape)
print(type(res))
print(type(res[0]))
print(type(res[0][0]))

save_img('warp_00.jpg',res[0][0][0][0])
save_img('warp_01.jpg',res[0][0][0][1])

save_img('warp_10.jpg',res[1][0][0][0])
save_img('warp_11.jpg',res[1][0][0][1])

import sys
sys.exit()



for vn in vid_names:
    vn_path = osp(data_path, vn, "input_images")
    pickle_fn = vn+".pkl"
    a = 10
    res = {}
    with open(pickle_fn, 'wb') as handle:     
    # print(vn_path)
        # find the len of the list
        file_list = os.listdir(vn_path)
        for idx in range(len(file_list)-1):
            print("Running for:", vn + str(idx))
            arguments_strFirst = osp(vn_path, "frame_"+str(idx)+".jpg")
            arguments_strSecond = osp(vn_path, "frame_"+str(idx+1)+".jpg")
            if(not ose(arguments_strFirst) or not ose(arguments_strSecond)):
                continue
            arg_out = "frame_"+str(idx)
            X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype) # shape 3, 480, 640
            X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)

            # arguments_strOut = os.path.join(gen_dir, dir, "frame10i11.png")

            X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype) # shape 3, 480, 640
            X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)


            y_ = torch.FloatTensor()

            assert (X0.size(1) == X1.size(1))
            assert (X0.size(2) == X1.size(2))

            intWidth = X0.size(2)
            intHeight = X0.size(1)
            channel = X0.size(0)
            if not channel == 3:
                continue

            if intWidth != ((intWidth >> 7) << 7):
                intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
                intPaddingLeft =int(( intWidth_pad - intWidth)/2)
                intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
            else:
                intWidth_pad = intWidth
                intPaddingLeft = 32
                intPaddingRight= 32

            if intHeight != ((intHeight >> 7) << 7):
                intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
                intPaddingTop = int((intHeight_pad - intHeight) / 2)
                intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
            else:
                intHeight_pad = intHeight
                intPaddingTop = 32
                intPaddingBottom = 32

            pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

            torch.set_grad_enabled(False)
            X0 = Variable(torch.unsqueeze(X0,0))
            X1 = Variable(torch.unsqueeze(X1,0))
            X0 = pader(X0)
            X1 = pader(X1)

            if use_cuda:
                X0 = X0.cuda()
                X1 = X1.cuda()
            proc_end = time.time()
            
            # print("Calling forward")
            res[arg_out] = model(torch.stack((X0, X1),dim = 0)).cpu().numpy()
            # print("Finished calling forward")
            # res[arg_out] = model.return_forward(torch.stack((X0, X1),dim = 0))

                
                # sys.exit()
        pickle.dump(res, handle)
        
# interp_error = AverageMeter()
# if DO_MiddleBurryOther:
#     subdir = os.listdir(MB_Other_DATA)
#     gen_dir = os.path.join(MB_Other_RESULT, unique_id)
#     os.mkdir(gen_dir)

#     tot_timer = AverageMeter()
#     proc_timer = AverageMeter()
#     end = time.time()
#     for dir in subdir:
#         print(dir)
#         os.mkdir(os.path.join(gen_dir, dir))
#         arguments_strFirst = os.path.join(MB_Other_DATA, dir, "frame10.png") # pass frame 1
#         arguments_strSecond = os.path.join(MB_Other_DATA, dir, "frame11.png") # pass frame 2
#         arguments_strOut = os.path.join(gen_dir, dir, "frame10i11.png")
#         gt_path = os.path.join(MB_Other_GT, dir, "frame10i11.png")

#         X0 =  torch.from_numpy( np.transpose(imread(arguments_strFirst) , (2,0,1)).astype("float32")/ 255.0).type(dtype) # shape 3, 480, 640
#         X1 =  torch.from_numpy( np.transpose(imread(arguments_strSecond) , (2,0,1)).astype("float32")/ 255.0).type(dtype)


#         y_ = torch.FloatTensor()

#         assert (X0.size(1) == X1.size(1))
#         assert (X0.size(2) == X1.size(2))

#         intWidth = X0.size(2)
#         intHeight = X0.size(1)
#         channel = X0.size(0)
#         if not channel == 3:
#             continue

#         if intWidth != ((intWidth >> 7) << 7):
#             intWidth_pad = (((intWidth >> 7) + 1) << 7)  # more than necessary
#             intPaddingLeft =int(( intWidth_pad - intWidth)/2)
#             intPaddingRight = intWidth_pad - intWidth - intPaddingLeft
#         else:
#             intWidth_pad = intWidth
#             intPaddingLeft = 32
#             intPaddingRight= 32

#         if intHeight != ((intHeight >> 7) << 7):
#             intHeight_pad = (((intHeight >> 7) + 1) << 7)  # more than necessary
#             intPaddingTop = int((intHeight_pad - intHeight) / 2)
#             intPaddingBottom = intHeight_pad - intHeight - intPaddingTop
#         else:
#             intHeight_pad = intHeight
#             intPaddingTop = 32
#             intPaddingBottom = 32

#         pader = torch.nn.ReplicationPad2d([intPaddingLeft, intPaddingRight , intPaddingTop, intPaddingBottom])

#         torch.set_grad_enabled(False)
#         X0 = Variable(torch.unsqueeze(X0,0))
#         X1 = Variable(torch.unsqueeze(X1,0))
#         X0 = pader(X0)
#         X1 = pader(X1)

#         if use_cuda:
#             X0 = X0.cuda()
#             X1 = X1.cuda()
#         proc_end = time.time()

#         # now we need to add information from the model
#         model.forward_tryon(torch.stack((X0, X1),dim = 0))