# Simple convolutional neural network to perform classification

In this Project, a **Convolutional Neural Network (CNN)** is implemented for image classification using the **CIFAR-10 dataset**. 

The process begins with environment setup and dataset preparation, where the CIFAR-10 dataset, consisting of 60,000 color images across 10 classes, is downloaded and preprocessed. 
Then we created a CNN model. The CNN model architecture includes convolutional and max-pooling layers, followed by fully connected layers, with parameters customizable for flexibility. 
The model is trained for 20 epochs using the Adam optimizer and sparse categorical crossentropy loss function, and its performance is evaluated on the testing dataset, including metrics like accuracy, confusion matrix, precision, and recall. 

Additionally, the project explores the impact of different learning rates on training and validation loss and involves transfer learning by fine-tuning state-of-the-art pre-trained models such as ResNet and GoogLeNet on the CIFAR-10 dataset. The results are then compared with those of the custom CNN model, followed by a discussion on the trade-offs, advantages, and limitations of using a custom model versus a pre-trained one.

## The CIFAR-10 dataset

![image](https://github.com/Pattern-Recog/Simple-convolutional-neural-network-to-perform-classification/assets/68577937/a74c280f-6613-4462-81ff-539b14ab18ac)


The CIFAR-10 [https://www.cs.toronto.edu/~kriz/cifar.html] dataset is a widely used benchmark in the field of computer vision and machine learning. It consists of 60,000 color images, each with dimensions of 32x32 pixels, distributed across 10 distinct classes. The dataset is designed for the task of object recognition and image classification, making it suitable for training and evaluating various machine learning models.

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

## CNN Architecture Overview
![image](https://github.com/Pattern-Recog/Simple-convolutional-neural-network-to-perform-classification/assets/68577937/61069055-bb59-4e04-bb32-0ed4e42775b3)

1. **Convolutional Layers:**
  The CNN begins with three convolutional layers (conv1, conv2, conv3), each followed by batch normalization (bn1, bn2, bn3) and Rectified Linear Unit (ReLU) activation (relu1, relu2, relu3).
  Each convolutional layer applies a 5x5 kernel with padding and a stride of 1 to capture spatial features in the input images.
  The number of output channels for the convolutional layers gradually increases from x1 (32) to x2 (64) to x3 (128), controlling the complexity and abstraction of learned features.

2. **Max Pooling Layers:**
  After each convolutional layer, a max-pooling layer (maxpool1, maxpool2, maxpool3) with a 2x2 kernel and a stride of 2 is applied, reducing the spatial dimensions of the feature maps and providing translation invariance.

3. **Fully Connected Layers:**
  The flattened output from the last max-pooling layer is fed into a fully connected layer (fc1), followed by ReLU activation (relu4) and dropout (dropout) with a rate of d (0.5).
  The output size of this fully connected layer is controlled by the parameter x4 (64).

4. **Output Layer:**
  The final layer (fc2) produces the output logits, which are passed through a softmax activation (softmax) to obtain class probabilities for the 10 classes in the CIFAR-10 dataset.


### Convolutional Layers:

  **1. Convolutional Layer 1:**
  - Input: 3 channels (RGB)
  - Output Channels: 32
  - Kernel Size: 5x5
  - Stride: 1, Padding: 2
  - Batch Normalization and **_ReLU Activation_**
    
  **2. Max Pooling Layer 1:**
  - Kernel Size: 2x2
  - Stride: 2
  
  **3. Convolutional Layer 2:**
  - Input Channels: 32
  - Output Channels: 64
  - Kernel Size: 5x5
  - Stride: 1, Padding: 2
  - Batch Normalization and **_ReLU Activation_**
  
  **4. Max Pooling Layer 2:**
  - Kernel Size: 2x2
  - Stride: 2
  
  **5. Convolutional Layer 3:**
  - Input Channels: 64
  - Output Channels: 128
  - Kernel Size: 5x5
  - Stride: 1, Padding: 2
  - Batch Normalization and **_ReLU Activation_**
  
  **6. Max Pooling Layer 3:**
  - Kernel Size: 2x2
  - Stride: 2

### Fully Connected Layers:

  **1. Flatten Layer:**
  - Flattens the output from the last convolutional layer.
  
  **2. Fully Connected Layer 1:**
  - Input Features: 2048
  - Output Features: 64
  - **_ReLU Activation_**, Dropout with probability 0.5

     
### Output Layer:

  **1. Fully Connected Layer 2 (Output Layer):**
  - Input Features: 64
  - Output Features: 10 (for 10 classes)
  - **_SoftMax Activation_** for multiclass classification

### Model Parameters:
- Convolutional Layer 1: 2464 parameters
- Convolutional Layer 2: 51328 parameters
- Convolutional Layer 3: 205056 parameters
- Fully Connected Layer 1: 131136 parameters
- Output Layer: 650 parameters
  
The total number of learnable parameters in the entire CNN model is **_390,634_**. This summary provides an overview of the architectural choices and the parameterization of each layer in the SimpleCNN model.


## Model Training

### Training Configuration:

- Loss Function: CrossEntropyLoss used for multi-class classification.

- Optimizer: Adam optimizer employed with a learning rate of 0.00025.

- Number of Epochs: The model is trained for 20 epochs.

- Loss Tracking: Average training and validation losses are computed and stored for each epoch.

### Interpretation:
- The architecture is characterized by progressively increasing convolutional channels, promoting feature extraction.
- Batch normalization enhances training stability, while dropout mitigates overfitting in fully connected layers.
- Adam optimizer is chosen for its adaptive learning rate capabilities.
- The model is trained over 20 epochs to optimize parameters and minimize the cross-entropy loss.
- Training and validation losses are tracked for each epoch, providing insights into model convergence and generalization.

![image](https://github.com/Pattern-Recog/Simple-convolutional-neural-network-to-perform-classification/assets/68577937/8238031e-bc2c-4043-9b31-3c6647fc734a)

![image](https://github.com/Pattern-Recog/Simple-convolutional-neural-network-to-perform-classification/assets/68577937/71c8ba8e-e6e6-425c-8825-5bc97787fd3d)


## Model Evaluation

### Model Accuracy:
The model accuracy is a metric indicating the proportion of correctly classified instances out of the total instances in the test dataset. In this case, the test accuracy is calculated using the following formula:

![image](https://github.com/Pattern-Recog/Simple-convolutional-neural-network-to-perform-classification/assets/68577937/5d268a2c-f233-48b0-b8a3-c1bbf834f59f)
Here, the accuracy is reported as 70.78%, suggesting that the model correctly classified approximately 70.78% of the samples in the test set.

### Confusion Matrix:
The confusion matrix is a table used for evaluating the performance of a classification model. It displays the counts of true positive, true negative, false positive, and false negative predictions. The matrix is structured as follows:

![image](https://github.com/Pattern-Recog/Simple-convolutional-neural-network-to-perform-classification/assets/68577937/093c3b38-0b2e-4d94-8e1f-e51f16dceb29)

- **True Positive (TP):** Instances that are actually positive and predicted as positive.
- **False Positive (FP):** Instances that are actually negative but predicted as positive.
- **False Negative (FN):** Instances that are actually positive but predicted as negative.
- **True Negative (TN):** Instances that are actually negative and predicted as negative.

The heatmap visualization in the code provides a clear overview of how well the model is performing across different classes.

![image](https://github.com/Pattern-Recog/Simple-convolutional-neural-network-to-perform-classification/assets/68577937/f6d0a17d-1195-4887-9370-b922a36c6335)


## With state-of-the-art networks
- The primary goal is to leverage the knowledge encoded in pre-trained models (developed on large datasets like ImageNet) and adapt them to perform well on the CIFAR-10 dataset, a task known as transfer learning.

- Each pre-trained model's classifier is modified to have the appropriate number of output units for CIFAR-10 (10 classes). The modified models are then fine-tuned on the CIFAR-10 dataset using the provided training and testing data splits.

- Training and validation loss values are recorded for each epoch to monitor the training progress and potential overfitting.

- After fine-tuning, the models are evaluated on the testing dataset to assess their performance.





