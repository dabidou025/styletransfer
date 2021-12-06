import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 256 if torch.cuda.is_available() else 28

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("picasso.JPG")
content_img = image_loader("hubert.JPG")

gen_img = content_img.clone().requires_grad_(True)

unloader = transforms.ToPILImage()  # reconvert into PIL image

def compute_content_loss(gen_feat,orig_feat):
    return torch.mean((gen_feat-orig_feat)**2)

def compute_style_loss(gen, style):
    batch_size, n_feature_maps, height, width = gen.size()

    G_features = gen.view(n_feature_maps, height*width)
    G = torch.mm(G_features, G_features.t())

    A_features = style.view(n_feature_maps, height*width)
    A = torch.mm(A_features, A_features.t())

    return torch.mean((G-A)**2) #/ (4*channel*(height*width)**2)

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.req_features= ['0','5','10','19','28'] 
        #Since we need only the 5 layers in the model so we will be dropping all the rest layers from the features of the model
        self.model=models.vgg19(pretrained=True).features[:29] #model will contain the first 29 layers

    def forward(self,x):
        # normalize input for vgg
        # x = (x - cnn_normalization_mean) / cnn_normalization_std
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if (str(i) in self.req_features):
                features.append(x)

        return features

def compute_loss(gen_features, content_features, style_features):
    style_loss = content_loss = 0
    for gen, content, style in zip(gen_features, content_features, style_features):
        content_loss += compute_content_loss(gen, content)
        style_loss += compute_style_loss(gen, style)

    total_loss = alpha*content_loss + beta*style_loss
    return total_loss

model = VGG().to(device).eval() 

EPOCHS = 100
lr = 5e-3
alpha = 0.5
beta = 1 - alpha

optimizer = optim.Adam([gen_img],lr=lr)

for epoch in range(EPOCHS):

    gen_features = model(gen_img)
    content_features = model(content_img)
    style_features = model(style_img)

    total_loss = compute_loss(gen_features, content_features, style_features)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print('epoch :', epoch, '| loss =', total_loss.item())
        save_image(gen_img, "gen.png")
