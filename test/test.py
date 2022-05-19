import time
from options.test_options import TestOptions
from data.data_loader_test import CreateDataLoader
from models.networks import ResUnetGenerator, load_checkpoint
from models.afwm import AFWM
import torch.nn as nn
import os
import numpy as np
import torch
import cv2
import sys
import torch.nn.functional as F
from torchvision import utils
import torchvision.transforms as T
from util import flow_util
from pathlib import Path
from data.flownet_wrapper import FlowNetWrapper

Path("../results").mkdir(parents=True, exist_ok=True)
Path("../results/our_t_results").mkdir(parents=True, exist_ok=True)
Path("../results/im_gar_flow_wg").mkdir(parents=True, exist_ok=True)

opt = TestOptions().parse()

# import debugpy
# debugpy.listen(5690)
# print("Waiting for debugger")
# debugpy.wait_for_client()
# print("Attached! :)")

def de_offset(s_grid):
    [b,_,h,w] = s_grid.size()


    x = torch.arange(w).view(1, -1).expand(h, -1).float()
    y = torch.arange(h).view(-1, 1).expand(-1, w).float()
    x = 2*x/(w-1)-1
    y = 2*y/(h-1)-1
    grid = torch.stack([x,y], dim=0).float().cuda()
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)

    offset = grid - s_grid

    offset_x = offset[:,0,:,:] * (w-1) / 2
    offset_y = offset[:,1,:,:] * (h-1) / 2

    offset = torch.cat((offset_y,offset_x),0)
    
    return  offset

start_epoch, epoch_iter = 1, 0

f2c = flow_util.flow2color()

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print(dataset_size)
#import ipdb; ipdb.set_trace()
warp_model = AFWM(opt, 3)
print(warp_model)
warp_model.eval()
warp_model.cuda()
load_checkpoint(warp_model, opt.warp_checkpoint)

gen_model = ResUnetGenerator(7, 4, 5, ngf=64, norm_layer=nn.BatchNorm2d)
#print(gen_model)
gen_model.eval()
gen_model.cuda()
print("loading checkpoint...")
load_checkpoint(gen_model, opt.gen_checkpoint)
print("checkpoint loaded")

total_steps = (start_epoch-1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size / opt.batchSize

flownet_wrapper = FlowNetWrapper()

for epoch in range(1,2):

    out_frames = []
    p_tryon_tmp = None
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        real_image = data['image']
        clothes = data['clothes']
        ##edge is extracted from the clothes image with the built-in function in python
        edge = data['edge']
        edge = torch.FloatTensor((edge.detach().numpy() > 0.5).astype(np.int))
        clothes = clothes * edge        
        if step > 0:
            flow = data['flow']
            warped_by_flow = flownet_wrapper.getSample(p_tryon_tmp,flow.cuda())
            # print("warped_by_flow generated")

        flow_out = warp_model(real_image.cuda(), clothes.cuda())
        warped_cloth, last_flow, = flow_out
        warped_edge = F.grid_sample(edge.cuda(), last_flow.permute(0, 2, 3, 1),
                          mode='bilinear', padding_mode='zeros')

        gen_inputs = torch.cat([real_image.cuda(), warped_cloth, warped_edge], 1)
        gen_outputs = gen_model(gen_inputs)
        p_rendered, m_composite = torch.split(gen_outputs, [3, 1], 1)
        p_rendered = torch.tanh(p_rendered)
        m_composite = torch.sigmoid(m_composite)
        m_composite = m_composite * warped_edge
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)
        p_tryon_tmp = p_tryon.clone().detach()

        path = 'results/' + opt.name
        os.makedirs(path, exist_ok=True)
        #sub_path = path + '/PFAFN'
        #os.makedirs(sub_path,exist_ok=True)
        print(data['p_name'])

        if step == 1:
            utils.save_image(
                p_rendered,
                os.path.join('../results/', 'p_rendered.jpg'),
                nrow=int(1),
                normalize=True,
                range=(-1,1),
            )
            utils.save_image(
                p_tryon,
                os.path.join('../results/', 'p_tryon.jpg'),
                nrow=int(1),
                normalize=True,
                range=(-1,1),
            )
            utils.save_image(
                warped_by_flow,
                os.path.join('../results/', 'warped_flow.jpg'),
                nrow=int(1),
                normalize=True,
                range=(-1,1),
            )

        if step % 1 == 0:

            ## write into video
            out_frames.append(os.path.join('../results/our_t_results', data['p_name'][0]))
            
            ## save try-on image only

            utils.save_image(
                p_tryon,
                os.path.join('../results/our_t_results', data['p_name'][0]),
                nrow=int(1),
                normalize=True,
                range=(-1,1),
            )
            
            ## save person image, garment, flow, warped garment, and try-on image
            
            a = real_image.float().cuda()
            b = clothes.cuda()
            flow_offset = de_offset(last_flow)
            flow_color = f2c(flow_offset).cuda()
            c= warped_cloth.cuda()
            d = p_tryon
            combine = torch.cat([a[0],b[0], flow_color, c[0], d[0]], 2).squeeze()
            utils.save_image(
                combine,
                os.path.join('../results/im_gar_flow_wg', data['p_name'][0]),
                nrow=int(1),
                normalize=True,
                range=(-1,1),
            )
            

        step += 1
        if epoch_iter >= dataset_size:
            break

    out_video = cv2.VideoWriter('../results/result_aug.avi',fourcc=cv2.VideoWriter_fourcc(*'DIVX'), fps=30, frameSize=(192,256))
    for frame in out_frames:
        out_video.write(cv2.imread(frame))

    out_video.release()


