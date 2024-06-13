import numpy as np
import torch
import os
import random
from datetime import datetime

from FLWRNeuralNetwork import preprocess, FLWRNeuralNetwork



directory = "dataset/test/"

epochs = 1

n = FLWRNeuralNetwork()
n.load_state_dict(torch.load('fl.pth'))

types = ["rose", "tulip", "water_lily"]
num_classes = len(types)

file_lists = []
for t in types:
    dir = directory + t + '/'
    files = os.listdir(dir)
    file_lists.append(files)
    


total_labels = [0, 0, 0]
total_correct = [0, 0, 0]

total = 0
correct = 0

for i in range(10):
    for label in range(num_classes):
        dir = directory + types[label] + '/'
        files = file_lists[label]
        f = dir + files[i]

        img = preprocess(f)
        
        output = n.forward(img).detach()
        guess = np.argmax(output)
        
        total += 1
        total_labels[label] += 1
        
        if guess == label:
            correct += 1
            total_correct[label] += 1
            
            
print("Accuract:", correct / total)
print(total_labels)
print(total_correct)