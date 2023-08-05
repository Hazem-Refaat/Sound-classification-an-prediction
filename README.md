# Urbansound8k-Sound-Classfication-And-Prediction

Note: Please update the file paths in the code to match your PC's directory structure before running the scripts. Make sure to change the paths in the data loading section and any other sections where file paths are referenced.

# Sound Classification using Convolutional Neural Network (CNN)

This project demonstrates the use of a Convolutional Neural Network (CNN) for sound classification. The goal of the model is to predict and classify different sounds based on their audio features.

## Dataset

The dataset used in this project is the UrbanSound8K dataset, which can be downloaded from [here](https://urbansounddataset.weebly.com/urbansound8k.html). It consists of 8,732 labeled sound excerpts from various urban environments, categorized into 10 different sound classes.

## Model Architecture

The CNN model for sound classification consists of the following layers:

1. Convolutional Layer: Applies convolutional filters to extract local features from the input audio spectrogram.
2. Max Pooling Layer: Performs downsampling by selecting the maximum value within a region of the feature map.
3. Flatten Layer: Flattens the output from the previous layer into a 1-dimensional vector.
4. Dense Layers: Fully connected layers that learn higher-level representations and make predictions.
5. Output Layer: Uses the softmax activation function to produce class probabilities for multi-class classification.

The model is trained using the Adam optimizer and categorical cross-entropy loss function.

## Usage

1. Install the required dependencies.
2. Download the UrbanSound8K dataset and extract the audio files into a folder.
3. Adjust the file paths in the code according to your local directory structure.
4. Run the code to preprocess the data, build the CNN model, train it on the dataset, and evaluate its performance.
5. Use the trained model to predict and classify new sound samples.

## Results

The model achieved an accuracy of 97.07% on the training set and 93.36% on the validation set. These results indicate that the CNN model is capable of accurately classifying different sounds based on their audio features.
