import PWCNet
import S2D_models
import Resblock
import MegaDepth
import time
import torch
import torch.nn as nn

class Try_on(torch.nn.Module):
    def __init__(self, device, inp_channels, depth_inp_channels=16, training=False):
        super(Try_on, self).__init__()
        self.inp_channels = inp_channels
        self.depth_inp_channels = depth_inp_channels
        self.conv1 = nn.Conv2d(in_channels=8, out_channels=self.depth_inp_channels//2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=self.depth_inp_channels//2, kernel_size=1, stride=1, padding=0)
        self.residue_net = Resblock.__dict__['MultipleBasicBlock_4'](inp_channels, 128)
        self._initialize_weights()
        self.device = device
        self.training = training
    
    def _initialize_weights(self):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # print(m)
                count+=1
                # print(count)
                # weight_init.xavier_uniform(m.weight.data)
                nn.init.xavier_uniform_(m.weight.data)
                # weight_init.kaiming_uniform(m.weight.data, a = 0, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, inp1, inp2, inp3):
        # inp1 = inp1.to(self.device)
        # inp2 = inp2.to(self.device)
        # inp3 = inp3.to(self.device)
        losses = []
        inp2_1 = self.conv1(inp2[:, :self.depth_inp_channels//2])
        inp2_2 = self.conv2(inp2[:, self.depth_inp_channels//2:])

        cur_input = torch.cat((inp2_1, inp2_2, inp1), dim=1)

        res = self.residue_net(cur_input)

        if self.training == True:
            losses += [res - inp3]
            return losses, res
        else:
            return res
        


