import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as transforms
from vgg19 import *


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=mean, std=std)
])

def load_json_config(file_path):
    with open(file_path, 'r') as file:
        config = json.load(file)
    return config

config = load_json_config('../Neural_Style_Transfer/config.json')

def load_image(image_link):
    image = cv2.imread(image_link)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def content_loss(original_image, stylied_image):
    return  1/2 * (torch.sum((original_image - stylied_image) ** 2))

def gram_matrix(feature_map):
    b, c, h, w = feature_map.size()
    feature_map = feature_map.view(b*c, -1)
    return torch.mm(feature_map, feature_map.t())

def style_loss(style_image, stylied_image):
    style_gram_matrix = gram_matrix(style_image)
    stylied_gram_matrix = gram_matrix(stylied_image)

    Nl, Ml = style_gram_matrix.size()

    return (1/(4 * Nl**2 * Ml**2)) * torch.sum((style_gram_matrix - stylied_gram_matrix) ** 2)

def convert_image_origin_size(image, height, width):
    image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    image = image * std + mean
    image = (image * 255.0).clip(0, 255).astype(np.uint8)
    image = cv2.resize(image, (width, height))
    return image

def visualize_feature_map_layer(model, image):
    feature_maps = model.get_feature_maps(image)
    fig, axis = plt.subplots(len(feature_maps), 16, figsize=(32, 32))
    for i in range(len(feature_maps)):
        feature_map = feature_maps[i]
        for j in range(16):
            feature_map_in_block = feature_map.squeeze(0)[j].detach().numpy()
            axis[i][j].imshow(feature_map_in_block)
            axis[i][j].axis('off')
            axis[i][j].set_title(f'Block {i+1}')
    plt.show()

class NeuralStyleTransfer(nn.Module):
    def __init__(self):
        super(NeuralStyleTransfer, self).__init__()
        self.model = VGG19()
        
    def forward(self, content_image, style_image, step_sizes):
        self.model.to(device).eval()
        original_height, original_width = content_image.shape[:2] 

        content_image = trans(content_image).unsqueeze(0)
        style_image = trans(style_image).unsqueeze(0)

        content_image = content_image.to(device)
        style_image = style_image.to(device)
            
        generated_image = content_image.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([generated_image], lr=config['parameters']['learning_rate'])

        for step_size in tqdm(range(step_sizes)):
            
            generated_style_feature_maps = self.model(generated_image, mode='style')
            generated_content_feature_map = self.model(generated_image, mode='content')
            
            content_feature_map = self.model(content_image, mode='content')
            style_feature_maps = self.model(style_image, mode='style')
            
            contentloss = 0
            styleloss = 0
            total_loss = 0
            for generated_style_feature_map, style_feature_map in zip(generated_style_feature_maps, style_feature_maps):
                styleloss += config['parameters']['wl'] * style_loss(style_feature_map, generated_style_feature_map)
            
            contentloss = content_loss(content_feature_map, generated_content_feature_map)
            
            total_loss = config['parameters']['alpha'] * contentloss + config['parameters']['beta'] * styleloss
            optimizer.zero_grad()
            total_loss.backward()
                
            optimizer.step()
            
        generated_image = transforms.Resize((original_height, original_width))(generated_image)
        vutils.save_image(generated_image, f'../Neural_Style_Transfer/generated_image_{step_size+1}.jpg', normalize=True)
        styled_image = convert_image_origin_size(generated_image, original_height, original_width)
        
        return styled_image
    
# content_image = load_image('../Neural_Style_Transfer/content_image.jpg')
# style_image = load_image('../Neural_Style_Transfer/style_image.jpg')

# neural_style_transfer = NeuralStyleTransfer()
# styled_image = neural_style_transfer(content_image, style_image, 2)
# print(styled_image.shape)
# plt.imshow(styled_image)
# plt.axis('off')
# plt.show()

