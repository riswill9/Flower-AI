import numpy as np
import torch
import os
import random
from datetime import datetime

from FLWRNeuralNetwork import preprocess, FLWRNeuralNetwork



directory = "ClassificationProject/dataset/train/"

epochs = 3

n = FLWRNeuralNetwork()

types = ["rose", "tulip", "water_lily"]
num_classes = len(types)

file_lists = []
for t in types:
    dir = directory + t + '/'
    files = os.listdir(dir)
    file_lists.append(files)
    


print("Start:", datetime.now())


for epoch in range(epochs):
    
    print("Epoch:", epoch)

    for i in range(800):
        for label in range(num_classes):
            dir = directory + types[label] + '/'
            files = file_lists[label]
            f = dir + files[i]

            img = preprocess(f)
            
            target = np.zeros(3)
            target[label] = 1.0

            n.train(img, target)

        
        if i % 100 == 0:
            print(i)

      

torch.save(n.state_dict(), 'fl.pth')
print("End:", datetime.now())