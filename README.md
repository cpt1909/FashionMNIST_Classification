# Fashion MNIST Classification

This project demonstrates a deep learning approach to classifying images of clothing from the Fashion MNIST dataset. The dataset consists of grayscale images of 10 categories of clothing, such as T-shirts, trousers, and sneakers. The project uses a convolutional neural network (CNN) implemented in TensorFlow/Keras to achieve efficient and accurate classification.

## Features

- **Dataset Handling**: Loads and preprocesses the Fashion MNIST dataset, including normalization and reshaping of image data for input into the CNN.
- **Model Architecture**: Defines a CNN with layers such as Conv2D, MaxPooling2D, Flatten, and Dense for effective feature extraction and classification.
- **Model Training**: Trains the model on the training dataset, with options for saving and loading pre-trained models.
- **Evaluation**: Evaluates the model on test data to compute accuracy and loss, providing insights into performance.
- **Visualization**: Includes tools for visualizing sample images from the dataset and their predicted labels.

## Project Structure

- **Dataset Loading and Visualization**
  - Load the Fashion MNIST dataset using Keras.
  - Visualize sample images from the dataset to understand the data distribution.

- **Data Preprocessing**
  - Normalize pixel values to the range [0, 1].
  - Reshape the data to include a channel dimension for compatibility with CNNs.

- **Model Definition and Training**
  - Define a sequential CNN model with the following layers:
    - Convolutional and pooling layers for feature extraction.
    - Fully connected layers for classification.
  - Train the model using the Adam optimizer and categorical cross-entropy loss.
  - Save the trained model to a file for future use.

- **Model Evaluation and Testing**
  - Evaluate the trained model on the test dataset to calculate accuracy and loss.
  - Test the model on individual samples to verify predictions.

## Requirements

- Python 3.7+
- TensorFlow 2.0+
- Matplotlib
- NumPy

Install the required libraries using:
```bash
pip install tensorflow matplotlib numpy
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/cpt1909/FashionMNIST_Classification.git
   cd fashion-mnist-classification
   ```

2. Run the Jupyter Notebook or Python script to train and test the model.

3. To test individual predictions, modify the code to input specific indices from the test dataset.

## Results

The model achieves an accuracy of **90.89%** on the test dataset, demonstrating its ability to classify Fashion MNIST images effectively.

## Acknowledgments

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- TensorFlow/Keras for deep learning support.