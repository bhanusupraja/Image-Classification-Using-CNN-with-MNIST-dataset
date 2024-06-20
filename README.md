# Image Classification Using CNN with MNIST Dataset

This repository contains Python code for training a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. The CNN is implemented using TensorFlow and Keras, and the trained model achieves high accuracy in recognizing digits.

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

- Python (3.x recommended)
- TensorFlow (2.x recommended)
- NumPy
- Matplotlib

### Installation

 Clone the repository:
   ```bash
   git clone https://github.com/bhanusupraja/Image-Classification-Using-CNN-with-MNIST-dataset.git
   cd Image-Classification-Using-CNN-with-MNIST-dataset
   ```

### Dataset

The MNIST dataset is used for training and testing the CNN model. It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). Each image is grayscale with a resolution of 28x28 pixels.

### Model Architecture

The CNN model architecture used for this project is as follows:

1. **Input Layer**: Accepts images of size (28, 28, 1).
2. **Convolutional Layers**: Three convolutional layers with ReLU activation.
   - 32 filters of size (3, 3)
   - 64 filters of size (3, 3)
   - 128 filters of size (3, 3)
3. **Max Pooling Layers**: Two max pooling layers with pool size (2, 2).
4. **Dropout Layers**: Applied after each max pooling layer to prevent overfitting (dropout rate = 0.5).
5. **Flatten Layer**: Flattens the 2D output from the convolutional layers to 1D.
6. **Dense Layer**: Output layer with 10 units and softmax activation for multi-class classification (digits 0-9).

### Training and Evaluation

The model is trained for 5 epochs using the Adam optimizer with a sparse categorical cross-entropy loss function. During training, both training and validation loss/accuracy are monitored and recorded using a custom callback (`TrainingHistory`).

After training, the model is evaluated on the test set to measure its performance on unseen data. Test accuracy and loss are printed to assess the model's effectiveness.

### Usage

You can run the main script `MnistClassificationUsingCNN.ipynb` to train and evaluate the CNN model:

The script will train the model, evaluate its performance on the test set, and plot the training/validation loss and accuracy graphs using Matplotlib.

### Example Prediction

After training, the model can predict digits from the test set. An example prediction is shown where a sample image from the test set is input into the trained model, and the actual and predicted labels are displayed using Matplotlib.

## Results

The trained CNN model achieves an accuracy of over 99% on the test set, demonstrating its capability to accurately classify handwritten digits from the MNIST dataset.
![image](https://github.com/bhanusupraja/Image-Classification-Using-CNN-with-MNIST-dataset/assets/48727565/80c3fceb-9836-44ec-ad47-d10cb57a19cd)


## Authors

- Bhanu Supraja(https://github.com/bhanusupraja)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the MNIST dataset and CNN architecture tutorials.

---

This README file provides a comprehensive overview of the project, including how to get started, the dataset used, model architecture, training and evaluation process, usage instructions, example predictions, results, author information, licensing details, and acknowledgments. Adjust the links and details to fit your specific project and preferences.
