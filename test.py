import numpy as np
import torch
import os
import random
from datetime import datetime

from FLWRNeuralNetwork import preprocess, FLWRNeuralNetwork


def get_label(f):
    label = 0
    temp = f.replace('.jpg', '.xml')
    with open(temp, 'rt') as file:
        content = file.read()
        if '<name>jerry</name>' in content:
            label = 1
    return label


epochs = 5

n = TJNeuralNetwork()


print("Start:", datetime.now())

directory = "datasets/train/"

file_list = os.listdir(directory)


for epoch in range(epochs):
    
    random.shuffle(file_list)

    print("Epoch:", epoch)

    flip = False
    #if epoch % 2 == 1:
    #   flip = True

    count = 0
    for filename in file_list:
        if not filename.endswith(".jpg"):
            continue

        f = directory + filename
        #print(f)

        img = preprocess(f, flip)
        
        target = np.zeros(1)
        label = get_label(f)

        if label == 1:
            target[0] = 1.0

        n.train(img, target)

        count += 1
        if count % 100 == 0:
            print(count)

      

torch.save(n.state_dict(), 'flower.pth')
print("End:", datetime.now())