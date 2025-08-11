# Temporal Pose Transformer (TPT)

A lightweight, sequence-based deep learning model for real-time human activity recognition from pose keypoints. This project contains the validated Keras/TensorFlow implementation of the TPT architecture.

## Project Goal

The goal of this model is to achieve high-accuracy activity recognition (>84%) with a very small footprint (<0.10M parameters), making it suitable for deployment on edge devices like a Raspberry Pi.

## Current Status

The model architecture has been implemented and successfully validated. The core components are tested for creation, training on dummy data, saving, and reloading.

### Requirements

To run the validation script, you will need the following libraries:
- tensorflow
- numpy
- h5py

You can install them via pip:
`pip install tensorflow numpy h5py`

### How to Validate the Model

To confirm the code is working correctly, run the validation script from the terminal:

`python validate_model.py`

You should see a series of checks that end with "✅ ALL CHECKS PASSED!".