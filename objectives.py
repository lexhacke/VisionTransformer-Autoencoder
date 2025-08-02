import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class Discriminator(nn.Module):
    def __init__(self, img_shape, filters=[256,512]):
        super().__init__()
        module_list = [nn.Conv2d(img_shape[0], filters[0], kernel_size=3, stride=2, padding=1),
                       nn.BatchNorm2d(filters[0]),
                       nn.LeakyReLU(0.2)]
        for i in range(1,len(filters)):
            module_list += [nn.Conv2d(filters[i-1], filters[i], kernel_size=3, stride=2, padding=1),
                            nn.BatchNorm2d(filters[i]),
                            nn.LeakyReLU(0.2)]

        self.convs = nn.Sequential(*module_list)
        self.mlp = nn.Sequential(nn.Conv2d(filters[-1], 1, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        x = self.convs(x)
        x = self.mlp(x)
        return x

class vgg_builder(nn.Module):
    def __init__(self):
        super(vgg_builder, self).__init__()
        convs = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.N_slices = 5
        self.slices = nn.ModuleList(list(nn.Sequential() for _ in range(self.N_slices)))
        for x in range(4):
            self.slices[0].add_module(str(x), convs[x])
        for x in range(4, 9):
            self.slices[1].add_module(str(x), convs[x])
        for x in range(9, 16):
            self.slices[2].add_module(str(x), convs[x])
        for x in range(16, 23):
            self.slices[3].add_module(str(x), convs[x])
        for x in range(23, 30):
            self.slices[4].add_module(str(x), convs[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat_map = []
        x = (x+1)/2
        x = self.slices[0](x)
        feat_map.append(x)
        x = self.slices[1](x)
        feat_map.append(x)
        x = self.slices[2](x)
        feat_map.append(x)
        x = self.slices[3](x)
        feat_map.append(x)
        x = self.slices[4](x)
        feat_map.append(x)
        return feat_map