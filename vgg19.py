import torch
import torch.nn as nn

from torchvision import models

class VGG19(nn.Module):
    def __init__(self, required_grad=False):
        super(VGG19, self).__init__()
        self.required_grad = required_grad

        self.vgg19 = models.vgg19(weights='IMAGENET1K_V1', progress=True)
        self.feature_maps = list(self.vgg19.children())[0]
        self.conv_layers = nn.Sequential(*self.feature_maps)

        for layers, params in self.vgg19.named_parameters():
            if not self.required_grad:
                params.requires_grad = False

    def forward(self, x, mode='style'):
        feature_maps = []

        if mode == 'style':
            layers = [0, 5, 10, 19, 28]
            for i in range(len(self.feature_maps)):
                x = self.feature_maps[i](x)
                if i in layers:
                    feature_maps.append(x)
            return feature_maps

        elif mode == 'content':
            layer = 21
            for i in range(len(self.feature_maps)):
                x = self.feature_maps[i](x)
                if i == layer:
                    return x

    def get_feature_maps(self, image):
        feature_maps = []
        for i in range(len(self.conv_layers)):
            image = self.conv_layers[i](image)
            if type(self.conv_layers[i]) == nn.Conv2d:
                feature_maps.append(image)
        return feature_maps
