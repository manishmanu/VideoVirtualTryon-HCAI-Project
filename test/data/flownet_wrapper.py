import torch

from data.flownet import FlowNet
from torchvision import transforms

class FlowNetWrapper:

    def __init__(self) -> None:
        self.flownet = FlowNet()
        self.to_tensor_and_norm_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def getFlow(self, im1, im2):
        im1 = self.to_tensor_and_norm_rgb(im1).unsqueeze(0).cuda()
        im2 = self.to_tensor_and_norm_rgb(im2).unsqueeze(0).cuda()
        flow_, conf_ = self.flownet(im1, im2)
        return flow_.squeeze().data.cpu().numpy().transpose(1, 2, 0)