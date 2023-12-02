# Simple convolutional neural network to perform classification

In this Project, a **Convolutional Neural Network (CNN)** is implemented for image classification using the **CIFAR-10 dataset**. 

The process begins with environment setup and dataset preparation, where the CIFAR-10 dataset, consisting of 60,000 color images across 10 classes, is downloaded and preprocessed. 
Then we created a CNN model. The CNN model architecture includes convolutional and max-pooling layers, followed by fully connected layers, with parameters customizable for flexibility. 
The model is trained for 20 epochs using the Adam optimizer and sparse categorical crossentropy loss function, and its performance is evaluated on the testing dataset, including metrics like accuracy, confusion matrix, precision, and recall. 

Additionally, the project explores the impact of different learning rates on training and validation loss and involves transfer learning by fine-tuning state-of-the-art pre-trained models such as ResNet and GoogLeNet on the CIFAR-10 dataset. The results are then compared with those of the custom CNN model, followed by a discussion on the trade-offs, advantages, and limitations of using a custom model versus a pre-trained one.

## The CIFAR-10 dataset

![image](https://github.com/Pattern-Recog/Simple-convolutional-neural-network-to-perform-classification/assets/68577937/a74c280f-6613-4462-81ff-539b14ab18ac)


The CIFAR-10 dataset is a widely used benchmark in the field of computer vision and machine learning. It consists of 60,000 color images, each with dimensions of 32x32 pixels, distributed across 10 distinct classes. The dataset is designed for the task of object recognition and image classification, making it suitable for training and evaluating various machine learning models.

The ten classes in the CIFAR-10 dataset represent common objects and animals, including airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks. Each class contains 6,000 images, with a balanced distribution to ensure a diverse and representative set of examples for training and testing.

Researchers and practitioners often use CIFAR-10 as a standard dataset for developing and comparing image classification algorithms. Its relatively small size facilitates quicker model training and experimentation. The dataset's popularity is attributed to its ability to challenge models with diverse visual patterns and complexities, making it a valuable resource for assessing the robustness and generalization capabilities of machine learning models in the context of image recognition tasks.

## Data pre-processing

### For PyTorch:
1. Data Transformation:
  The transforms.Compose function is used to define a sequence of transformations.
  transforms.ToTensor() converts the images into PyTorch tensors.
  transforms.Normalize standardizes the image pixel values to have a mean of 0.5 and a standard deviation of 0.5 for each color channel (RGB).

2. Load Training and Testing Datasets:
  The CIFAR-10 dataset is loaded using torchvision.datasets.CIFAR10.
  The root parameter specifies the directory to save/download the dataset.
  train=True indicates the training dataset, and train=False indicates the testing dataset.
  The defined transformations are applied to both the training and testing datasets.

3. Dataset Splitting:
  The torch.utils.data.random_split function is employed to split the training dataset into training, validation, and test sets.
  The sizes of the sets are determined based on the specified ratios (60% training, 20% validation, 20% testing).

5. Data Loaders:
  torch.utils.data.DataLoader is used to create data loaders for the training, validation, and testing sets.
  Batch size is set to 64, and shuffle=True is applied to the training loader for randomization during training.
  num_workers=2 indicates the number of subprocesses used for data loading.

### For Keras:

1. Load CIFAR-10 Dataset:
The CIFAR-10 dataset is loaded using keras.datasets.cifar10.
Training and testing images along with their labels are stored in (train_images, train_labels) and (test_images, test_labels).
_

The data pre-processing tasks, particularly normalization and dataset splitting, are crucial for ensuring that the model is trained on properly scaled inputs and evaluated on distinct subsets of data. Normalization helps in stabilizing training by ensuring that the input features are within a similar numerical range, while dataset splitting enables the model to generalize well by having separate sets for training, validation, and testing. Additionally, the use of data loaders enhances the efficiency of feeding batches of data during the training process.





