import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps

# https://www.kaggle.com/datasets/boltuzamaki/tom-and-jerrys-face-detection-dateset?select=images


CROP_WIDTH = 256
CROP_HEIGHT = 160
CHANNELS = 3

RESIZE_WIDTH = int((1280 / 720) * CROP_HEIGHT)
RESIZE_HEIGHT = CROP_HEIGHT

LAYER_1 = 4096
LAYER_2 = 2048
LAYER_3 = 1024
LAYER_4 = 512
LAYER_5 = 256

USE_CNN = True

device = torch.device('cpu')

def center_crop(image):
    width = image.size[0]
    left = (width / 2) - (CROP_WIDTH / 2)
    right = left + CROP_WIDTH

    height = image.size[1]
    top = (height / 2) - (CROP_HEIGHT / 2)
    bottom = top + CROP_HEIGHT
    return image.crop((left, top, right, bottom))

def preprocess(f, flip=False):
    image = Image.open(f)

    if CHANNELS == 1:
        image = ImageOps.grayscale(image)

    image = image.resize((RESIZE_WIDTH, RESIZE_HEIGHT))

    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    image = center_crop(image)

    a = np.array(image) / 255.0
    
    if USE_CNN:
        a = a.transpose((2, 0, 1))
        a = a.reshape((1, 3, CROP_HEIGHT, CROP_WIDTH))
    else:
        a = a.reshape(CROP_HEIGHT * CROP_WIDTH * CHANNELS)

    return a

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)
    
class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

class FLWRNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        if USE_CNN:
            self.model = nn.Sequential(
                nn.Conv2d(3, 256, kernel_size=4, stride=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
 
                nn.Conv2d(256, 3, kernel_size=4, stride=2),
                nn.ReLU(),
                

                View(7068),
                nn.Linear(7068, 1),
                nn.Sigmoid()
            )
            
        else:
            self.model = nn.Sequential(
                nn.Linear(CROP_HEIGHT * CROP_WIDTH * CHANNELS, LAYER_1),
                nn.Sigmoid(),
                nn.Linear(LAYER_1, LAYER_2),
                nn.Sigmoid(),
                nn.Linear(LAYER_2, LAYER_3),
                nn.Sigmoid(),
                nn.Linear(LAYER_3, LAYER_4),
                nn.Sigmoid(),
                nn.Linear(LAYER_4, LAYER_5),
                nn.Sigmoid(),
                nn.Linear(LAYER_5, 2),
                nn.Sigmoid()
            )

        self.loss_function = nn.BCELoss()

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.00001)

        self.to(device)



    def forward(self, inputs):
        inputs = torch.Tensor(inputs).to(device)
        return self.model(inputs)
    
    def train(self, inputs, target):
        target = torch.Tensor(target).to(device)
        outputs = self.forward(inputs)

        loss = self.loss_function(outputs, target)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

