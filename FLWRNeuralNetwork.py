import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps

# https://www.kaggle.com/datasets/boltuzamaki/tom-and-jerrys-face-detection-dateset?select=images


RESIZE_WIDTH = 128
RESIZE_HEIGHT = 128
CHANNELS = 3

LAYER_1 = 1000

device = torch.device('cpu')


def preprocess(f, flip=False):
    image = Image.open(f)
    image = image.resize((RESIZE_WIDTH, RESIZE_HEIGHT))

    a = np.array(image) / 255.0
    a = a.reshape(RESIZE_HEIGHT * RESIZE_WIDTH * CHANNELS)

    return a


class FLWRNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        
        self.model = nn.Sequential(
                nn.Linear(RESIZE_HEIGHT * RESIZE_WIDTH * CHANNELS, LAYER_1),
                nn.Sigmoid(),
                nn.Linear(LAYER_1, 3),
                nn.Sigmoid(),
            )

        self.loss_function = nn.MSELoss()

        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

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