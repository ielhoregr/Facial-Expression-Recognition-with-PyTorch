# Emotion Detection with EfficientNet

This project utilizes the EfficientNet architecture for emotion detection. It classifies facial expressions into seven categories: angry, disgust, fear, happy, neutral, sad, and surprise. The model is trained using the Facial Expression Dataset.

## Table of Contents

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Results](#results)
- [Additional Functionality](#additional-functionality)

## Overview

The project employs the EfficientNet architecture, a state-of-the-art convolutional neural network, for emotion detection from facial images. It utilizes PyTorch for implementation. The dataset consists of two parts: a training set and a validation set, both containing images labeled with their corresponding emotion categories.

## Dependencies

The project relies on the following libraries:

- NumPy
- Matplotlib
- PyTorch
- OpenCV (cv2)
- timm (for EfficientNet)
- tqdm (for progress tracking)

Ensure you have these libraries installed in your Python environment before running the project.

## Usage

### Training

To train the model:

1. Set the appropriate paths for training and validation image folders.
2. Define hyperparameters such as learning rate, batch size, and number of epochs.
3. Instantiate the model and optimizer.
4. Train the model using the provided training and validation loaders.

### Inference

After training, you can perform inference on images or videos.

#### Image Inference

1. Preprocess the input image using `preprocess_image()` function.
2. Load the trained model.
3. Utilize `predict_emotion()` function to predict the emotion class.
4. Visualize the prediction using `view_classify()` function.

#### Video Inference

1. Load the trained model.
2. Utilize OpenCV to capture frames from the video.
3. Detect faces in the frames using Haar cascade classifier.
4. Preprocess the detected faces and pass them through the model for emotion prediction.
5. Aggregate predictions over a certain number of frames to provide a more stable prediction.

## Results

The model achieves satisfactory accuracy on the validation set. Best weights are saved based on validation loss, ensuring the model's performance is preserved.

## Additional Functionality

Apart from training and inference on single images, the project also includes functionality for real-time emotion detection in videos. It continuously processes frames from the video feed, detects faces, and predicts emotions, providing live feedback on the emotions displayed in the video.
