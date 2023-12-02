# Simple convolutional neural network to perform classification

In this Project, a **Convolutional Neural Network (CNN)** is implemented for image classification using the **CIFAR-10 dataset**. 

The process begins with environment setup and dataset preparation, where the CIFAR-10 dataset, consisting of 60,000 color images across 10 classes, is downloaded and preprocessed. 
Then we created a CNN model. The CNN model architecture includes convolutional and max-pooling layers, followed by fully connected layers, with parameters customizable for flexibility. 
The model is trained for 20 epochs using the Adam optimizer and sparse categorical crossentropy loss function, and its performance is evaluated on the testing dataset, including metrics like accuracy, confusion matrix, precision, and recall. 

Additionally, the project explores the impact of different learning rates on training and validation loss and involves transfer learning by fine-tuning state-of-the-art pre-trained models such as ResNet and GoogLeNet on the CIFAR-10 dataset. The results are then compared with those of the custom CNN model, followed by a discussion on the trade-offs, advantages, and limitations of using a custom model versus a pre-trained one.
