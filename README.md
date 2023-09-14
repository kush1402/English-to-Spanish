# English to Spanish Translation with Neural Machine Translation

## Overview

This project implements a Neural Machine Translation (NMT) model to translate English sentences into Spanish using the Keras library. The model is based on LSTM (Long Short-Term Memory) networks and attention mechanisms for better translation performance.

## Requirements

Make sure you have the following libraries and tools installed:

- Python 3.x
- pandas
- numpy
- matplotlib
- scikit-learn
- Keras
- TensorFlow (compatible with Keras)
- gradio (for the interface)

## Usage

1. **Data Preprocessing**: The code starts by preprocessing the data from the `spa.txt` file. The dataset is cleaned and transformed to prepare it for training.

2. **Model Training**: The NMT model is trained with the preprocessed data. The training process can take some time, depending on your hardware and the dataset size.

3. **Model Evaluation**: The model's performance is evaluated on a test dataset.

4. **Inference**: You can use the trained model to translate English sentences into Spanish.

5. **Interactive Translation Interface**: An interactive translation interface is provided using Gradio. You can run the interface and input English sentences to get their Spanish translations.

## Model Architecture

The NMT model architecture consists of three LSTM layers for the encoder and one LSTM layer for the decoder. An attention mechanism is used to improve translation quality.

## Model Weights

The model weights after training for 12 epochs are provided in the repository (`nmt_weights_12epoch.h5`). You can load these weights to skip the training step.

## Acknowledgments

- The dataset used for training comes from [spa.txt](Notebook\spa.txt).
- This project was inspired by various NMT tutorials and resources in the field of machine translation.

