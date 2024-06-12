import torch
import os
import numpy as np
from PIL import Image

from TAJNeuralNetwork import TAJNeuralNetwork

n = TAJNeuralNetwork()

def preprocess(f):
    image = Image.open(f)
    image = image.resize((64, 64))
    a = np.array(image) / 255.0 # coverts array into numpy array
    a = a.reshape( 64 * 64 * 3) #length time width times 3 for rgb scale
    return a

n.load_state_dict(torch.load('tomAndJerry.pth'))


test = "datasets/test/"
types = ["mango", "pineapple", "watermelon", "guava", "coconut"]

stop_at = 100

correct = 0
# rows = idk
total = 0
for i in range(len(types)):
    directory = test + types[i]
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
        
        output = n.forward(img).detach().numpy()
        #print(output)
        
        total += 1
        
        guess = np.argmax(output)
        
        if (guess == label):
            correct += 1
           
            
print("Accuracy: ", correct/ total)