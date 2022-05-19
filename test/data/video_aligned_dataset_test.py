import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import linecache
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import utils

from data.flownet_wrapper import FlowNetWrapper

class VideoAlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.fine_height=256
        self.fine_width=192

        self.flownet_wrapper = FlowNetWrapper()
        self.flow_norm = transforms.Normalize((0.5, 0.5), (0.5, 0.5))

        self.c_name = os.path.join(opt.dataroot, "cloth.jpg")
        self.e_name = os.path.join(opt.dataroot, "edge.jpg")
        self.v_name = os.path.join(opt.dataroot, "video.mp4")

        self.capture = cv2.VideoCapture(self.v_name)
        self.frames_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("frames count: ",self.frames_count)
        self.frame_id = 0
        self.frames = []
        while(self.capture.isOpened()):
            ret, frame = self.capture.read()
            if ret:
                self.frames.append(frame)
            else:
                self.capture.release()
        print("frames len",len(self.frames))

        self.flows = []
        for index in range(self.frames_count):
            self.frames[index] = cv2.cvtColor(self.frames[index], cv2.COLOR_BGR2RGB)
            self.frames[index] = Image.fromarray(self.frames[index])
            if index > 0:
                flow_ = self.flownet_wrapper.getFlow(self.frames[index-1],self.frames[index])
                self.flows.append(flow_)

    def __getitem__(self, index):        
        I = self.frames[index]

        if index > 0:
            flow_tensor = torch.from_numpy(self.flows[index-1]).permute(2, 0, 1)
            flow_tensor = self.flow_norm(flow_tensor)
            # print("flow tensor shape: ",flow_tensor.size())

        params = get_params(self.opt, I.size)
        # print(params)
        transform = get_transform(self.opt, params)
        transform_E = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

        I_tensor = transform(I)
        #import ipdb; ipdb.set_trace()
        C_path = os.path.join(self.c_name)
        #print(self.c_name[index])
        C = Image.open(C_path).convert('RGB')
        C_tensor = transform(C)

        E_path = os.path.join(self.e_name)
        E = Image.open(E_path).convert('L')
        E_tensor = transform_E(E)

        input_dict = { 
            'image': I_tensor,
            'clothes': C_tensor,
            'edge': E_tensor,
            'p_name':'frame_'+str(index)+'.jpg'
        }

        if index > 0:
            input_dict['flow'] = flow_tensor
            
        self.frame_id += 1
        return input_dict

    def __len__(self):
        return self.frames_count 

    def name(self):
        return 'VideoAlignedDataset'
