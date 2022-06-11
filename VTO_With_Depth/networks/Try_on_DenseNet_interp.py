import torch
import torch.nn as nn
import math
from torchsummary import summary

# referred from https://towardsdatascience.com/simple-implementation-of-densely-connected-convolutional-networks-in-pytorch-3846978f2f36

class Dense_Block(nn.Module):
    def __init__(self, in_channels):
        super(Dense_Block, self).__init__()

        self.relu = nn.ReLU(inplace = True)
        self.bn = nn.BatchNorm2d(num_features = in_channels)

        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

    
    def forward(self, x):

        bn = self.bn(x)
        conv1 = self.relu(self.conv1(bn))

        conv2 = self.relu(self.conv2(conv1))
        c2_dense = self.relu(torch.cat([conv1, conv2], 1))

        conv3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))

        conv4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))

        conv5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))

        return c5_dense


class Transition_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer, self).__init__()

        self.relu = nn.ReLU(inplace = True)
        self.bn = nn.BatchNorm2d(num_features = out_channels)
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)

    def forward(self, x):

        out = self.bn(self.relu(self.conv(x)))
        # out = bn
        # out = self.avg_pool(bn)

        return out


class DenseNet(nn.Module):
    def __init__(self, nr_classes, in_channels=24):
        super(DenseNet, self).__init__()

        self.lowconv = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 7, padding = 3, bias = False)
        self.relu = nn.ReLU()

        # Make Dense Blocks
        self.denseblock1 = self._make_dense_block(Dense_Block, 64)
        self.denseblock2 = self._make_dense_block(Dense_Block, 128)
        self.denseblock3 = self._make_dense_block(Dense_Block, 128)

        # Make transition Layers
        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128)
        self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128)
        self.transitionLayer3 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 64)

        # Classifier
        self.bn = nn.BatchNorm2d(num_features = 64)
        self.pre_classifier = nn.Linear(64*4*4, 512)
        # self.classifier = nn.Linear(512, nr_classes)
        self.final_block = nn.Sequential(*[nn.Conv2d(64, 3 , (3, 3), 1, (1, 1))])


    def _make_dense_block(self, block, in_channels):
        layers = []
        layers.append(block(in_channels))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, layer, in_channels, out_channels):
        modules = []
        modules.append(layer(in_channels, out_channels))
        return nn.Sequential(*modules)

    def forward(self, x):
        out = self.relu(self.lowconv(x))

        out = self.denseblock1(out)
        out = self.transitionLayer1(out)

        out = self.denseblock2(out)
        out = self.transitionLayer2(out)

        out = self.denseblock3(out)
        out = self.transitionLayer3(out)
    
        out = self.bn(out)

        out = self.final_block(out)

        # Shardul made change here
        # out = out.view(-1, 64*4*4)

        # out = self.pre_classifier(out)
        # out = self.classifier(out)

        return out


class Try_on_DenseNet_interp(torch.nn.Module):
    def __init__(self, device, inp_channels, depth_inp_channels=16, training=False):
        super(Try_on_DenseNet_interp, self).__init__()
        self.inp_channels = inp_channels # inp_channels to main densenet after concatenation
        self.depth_inp_channels = depth_inp_channels
        
        self.conv1_1 = nn.Conv2d(in_channels=self.depth_inp_channels//2, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=self.depth_inp_channels//2, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2_1 = nn.Conv2d(in_channels=self.depth_inp_channels//2, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2_2 = nn.Conv2d(in_channels=self.depth_inp_channels//2, out_channels=3, kernel_size=1, stride=1, padding=0)

        
        self.residue_net = DenseNet(10, self.inp_channels)
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

    # inp1 --> masked body image
    # inp2 --> depth map (16 channels)
    # inp3 --> warped cloth
    # inp4 --> original output image
    def forward(self, inp1_1, inp1_2, inp1_3, inp3_1, inp3_2, inp3_3, im2_orig):
        # inp1 = inp1.to(self.device)
        # inp2 = inp2.to(self.device)
        # inp3 = inp3.to(self.device)
        losses = []


        inp1_2_1 = self.conv1_1(inp1_2[:, :self.depth_inp_channels//2])
        inp1_2_2 = self.conv1_2(inp1_2[:, self.depth_inp_channels//2:])
        inp3_2_1 = self.conv2_1(inp3_2[:, :self.depth_inp_channels//2])
        inp3_2_2 = self.conv2_2(inp3_2[:, self.depth_inp_channels//2:])

        cur_input = torch.cat((inp1_2_1, inp1_2_2, inp3_2_1, inp3_2_2, inp1_1, inp1_3, inp3_1, inp3_3), dim=1)
        # cur_input = torch.cat((inp2_1, inp2_2, inp1, inp3), dim=1)
        res = self.residue_net(cur_input)
        if self.training == True:
            losses += [res - im2_orig]
            return losses, res
        else:
            return res