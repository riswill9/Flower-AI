import os
import numpy as np
import torch
from PIL import Image

from TAJNeuralNetwork import TAJNeuralNetwork

# turns this file into whatever we need it to do (crop, edit)
def preprocess(f):
    image = Image.open(f)
    image = image.resize((64, 64))
    a = np.array(image) / 255.0 # coverts array into numpy array
    a = a.reshape( 64 * 64 * 3) #length time width times 3 for rgb scale
    return a

# directory = "datasets/train/mango"
train = "datasets/train/"

types = ["mango", "pineapple", "watermelon", "guava", "coconut"]

stop_at = 700

n = TAJNeuralNetwork()

for i in range(len(types)):
    directory = train + types[i]
    counter = 0
    for filename in os.listdir(directory):
        counter = counter + 1
        if counter == stop_at:
            break
        f = os.path.join(directory, filename)
        if not os.path.isfile(f):
            continue

        print(f)
        img = preprocess(f)
        label = i
            
        targets = np.zeros(len(types))
        targets[label] = 1.0
    
        
        n.train(img, targets)
        
torch.save(n.state_dict(), 'tomAndJerry.pth')