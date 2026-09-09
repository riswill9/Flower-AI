# Flower-AI 🌺

## AI Flower Classification & Generation

![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch)

Flower-AI is a deep learning project developed during Pace University's Artificial Intelligence Internship Experience. The project explores two applications of neural networks in computer vision: **classifying flower species** and **generating new flower images using a Generative Adversarial Network (GAN)**.

The project was developed by **Kerisa Williams and Jonathan Moore** and provided hands-on experience with neural network architecture, model training, image preprocessing, generative AI, dataset experimentation, and evaluating model performance.

*Link to the slide presentation below*

https://docs.google.com/presentation/d/1Hy2o5rCYvl11UM5iSc8jWyXby_F6_wDpvbOt0xy21Cw/edit?usp=sharing


---

## 🌟 Project Overview

Flower-AI contains two main deep learning projects:

### 🌷 Flower Classification
A neural network trained to distinguish between three flower species:

- Roses
- Tulips
- Water lilies

### 🎨 Flower Generation
A Generative Adversarial Network (GAN) trained on thousands of flower images to generate new synthetic flower imagery from random noise.

Together, the projects explore both **discriminative AI**, where a model learns to identify an image, and **generative AI**, where a model learns patterns from existing images to create new ones.

---

## 🧠 Flower Classification

The classification model is a custom feedforward neural network implemented with PyTorch.

Images are resized to **128 × 128 pixels** with three RGB color channels before being flattened and passed through the neural network.

### Architecture

The network uses multiple fully connected layers to progressively reduce the image representation before producing predictions for the three flower classes.

The project separates the classification workflow into:

- `FLWRNeuralNetwork.py` - neural network architecture and image preprocessing
- `train.py` - model training
- `test.py` - evaluation using unseen flower images

### Training

The classifier was trained on:

- **3 flower classes**
- **800 training images**
- **3 epochs**
- **Learning rate: 0.00001**

The model was then evaluated using flower images it had not seen during training.

---

## 📊 Classification Results

The highest overall classification accuracy achieved during experimentation was:

### **70% Accuracy**

| Flower Type | Correct Predictions |
|---|---:|
| 🌹 Roses | 5 / 10 |
| 🌷 Tulips | 7 / 10 |
| 🪷 Water Lilies | 9 / 10 |

The results showed that the model performed particularly well at recognizing water lilies while having more difficulty distinguishing roses.

---

# 🎨 AI Flower Generation

In addition to classification, Flower-AI explores **generative deep learning** through a Generative Adversarial Network.

The generation system consists of two neural networks that are trained against one another:

### Generator

The Generator learns to transform random noise into synthetic flower images.

It receives a **300-dimensional random noise vector** and passes it through a fully connected neural network.

The final output contains:

`49,152 values = 128 × 128 × 3`

These values are reshaped to produce a **128 × 128 RGB image**.

### Discriminator

The Discriminator acts as a binary classifier.

It receives either:

- A real flower image from the dataset
- An image produced by the Generator

and learns to determine whether that image is **real or generated**.

The Discriminator contains a hidden layer of **1,000 neurons** and produces a single real/fake prediction.

---

## 🔄 GAN Training Process

The Generator and Discriminator improve by competing against each other.

```text
Real Flower
     │
     ▼
Discriminator ──────► Real / Fake
     ▲
     │
Generated Flower
     ▲
     │
 Generator
     ▲
     │
Random Noise

