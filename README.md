# Landmark-Classification-Tagging-for-Social-Media

This project focuses on building a landmark classifier using Convolutional Neural Networks (CNNs). The goal is to automatically predict the location of an image based on any landmarks depicted in it. The solution address the challenge faced by photo sharing and storage services in organizing and tagging images without location metadata.

## Project Structure

The project is divided into three main parts, each implemented in a separate Jupyter notebook:
1. `cnn_from_scratch.ipynb`: Create a CNN from scratch
    - Visualize the dataset
    - Process data for training
    - Build and train a CNN
    - Export the best model using TorchScript

2. `transfer_learning.ipynb`: Use transfer learning
    - Investigate pre-trained models
    - Train and test a transfer-learned network
    - Export the best transfer learning solution

3. `app.ipynb`: Deploy the best model in an app
    - Create a simple application for landmark classification
    - Test the model and reflect on its strengths and weaknesses.

## Acknowledgements
This project is part of the Udacity Deep Learning Nanodegree program.

