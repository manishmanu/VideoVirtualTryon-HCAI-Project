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
    def __init__(self, nr_classes, in_channels=19):
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

model = DenseNet(10, in_channels=12)
model = model.cuda()

summary(model, (12, 256, 320))