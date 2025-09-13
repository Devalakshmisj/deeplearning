      exp 1
TensorFlow Basics and Tensor Operations
1.Problem Statement
The objective of this experiment is to explore the basics of TensorFlow, including the creation and manipulation of tensors. The tasks include creating constants, identifying tensor properties, generating random tensors, and performing basic tensor operations. This foundational knowledge is essential for building and training deep learning models.

2.Deep Learning Methods & Description
This experiment focuses on fundamental TensorFlow operations, which are the building blocks of deep learning:
Tensor Creation
Using tf.constant() to create scalar, vector, matrix, and tensor objects.
Tensor Properties
Finding the shape (dimensions), rank (number of dimensions), and size (total elements) of tensors.
Random Tensor Generation
Creating tensors with random values within specific ranges and shapes.
Tensor Operations
Performing basic mathematical operations such as addition, multiplication, and reshaping.

3.Methods Applied to Solve the Problem
Created a scalar, vector, matrix, and tensor using tf.constant().
Computed and displayed the shape, rank, and size for each tensor.
Generated two tensors with random values between 0 and 1 of shape [5, 300].
Used TensorFlow operations to manipulate and analyze tensors, such as reshaping and performing arithmetic.
Verified the results by printing intermediate outputs.

4.Results
Successfully created and manipulated tensors of various dimensions.
Verified the shape, rank, and size of all tensors created in Q1 and Q2.
Generated random tensors as specified in Q3, ensuring values were within the correct range.
Demonstrated basic arithmetic and reshaping operations in TensorFlow.

5.Conclusion
This experiment provided hands-on practice with TensorFlow basics, an essential prerequisite for implementing deep learning architectures. Understanding tensors, their properties, and operations ensures a solid foundation for model building, training, and optimization.


     Experiment 2 – Outlier Detection using Z-score and IQR

1.Problem Statement
The goal of this experiment is to identify outliers in a dataset using statistical methods. Outliers can significantly affect the performance of data analysis and machine learning models. This experiment applies two common techniques — Z-score and Interquartile Range (IQR) — to detect outliers in a given dataset.

2.Deep Learning Methods & Description
Although this experiment focuses on statistical methods rather than deep learning, it is an important preprocessing step before applying deep learning models. The methods are:
Z-score Method
Calculates how many standard deviations a data point is from the mean. If the absolute value of the Z-score is above a certain threshold, the point is considered an outlier.
IQR Method
Uses the first quartile (Q1) and third quartile (Q3) to measure the spread of the middle 50% of data. Any point lying outside the range [Q1 - 1.5IQR, Q3 + 1.5IQR] is flagged as an outlier.

3.Methods Applied to Solve the Problem
In this experiment:
A dataset of 9 values was analyzed.
Z-score method was applied with a threshold of 1 to find points far from the mean.
IQR method was applied to find points beyond the whiskers in a boxplot.
Both methods identified potential outliers for removal.

4.Results
Detected Outliers (Z-score method): Values with |Z| > 1 were identified.
Detected Outliers (IQR method): Values greater than Q3 + 1.5IQR or less than Q1 - 1.5IQR were flagged.
Boxplot Visualization: Clearly shows the distribution and highlights the outliers.

5.Conclusion
Outlier detection is crucial for improving the quality of datasets before applying machine learning or deep learning algorithms. In this experiment, both Z-score and IQR successfully identified the extreme values. Removing these values can lead to better model accuracy and reliability.

     Experiment 3  CIFAR-10 Classification with Fully Connected Neural Network


1.Problem Statement
The goal of this project is to classify images from the CIFAR-10 dataset into ten different categories using a fully connected (Dense) neural network architecture. CIFAR-10 is a widely used benchmark dataset in computer vision, consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

2.Deep Learning Methods Used
Dense (Fully Connected) Neural Network:
The model uses a sequential architecture with multiple dense layers to learn representations from image data.

Layers Included:
Flatten layer to convert image matrices into 1D arrays.
Two Dense layers with 128 and 68 neurons respectively, activated by ReLU.
Output Dense layer with 10 neurons (one for each class), activated by softmax.

3.Method Description
Preprocessing:

Input images are normalized to the range.
Labels are one-hot encoded for multi-class classification.
Model Compilation:
Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy

Training:
10 epochs
Batch size: 32
20% of the data is used for validation during training.

4.Results
Test Accuracy:
The final test accuracy (on the unseen test data) is printed after evaluation.

Learning Curves:
Training and validation loss and accuracy are plotted to visualize performance over epochs.

5.Conclusion
This project demonstrates how a simple fully connected neural network can be applied to the CIFAR-10 image classification problem. Although convolutional neural networks (CNNs) are more commonly used for image classification, this model achieves reasonable performance and illustrates the workflow from data preparation to evaluation using Keras and TensorFlow.

       Experiment 4 CIFAR-10 Image Classification with Deep Learning

1.Problem Statement Image classification is a fundamental task in computer vision where the goal is to categorize images into predefined classes.
This project focuses on classifying the CIFAR-10 dataset, which contains 60,000 images across 10 categories such as airplanes, cars, birds, cats, and more.

2.Deep Learning Methods & Description

Fully Connected Neural Network (Dense Layers)

The network uses multiple dense layers to process flattened image pixel data.

Dropout layers are applied to prevent overfitting.

Different initializers and regularizers can be tested for optimization.

Activation Function (ReLU)

Introduces non-linearity to help the network learn complex patterns.

Dropout Regularization

Randomly disables neurons during training to reduce overfitting.

3.DL Methods Applied to Solve the Problem

Data Preprocessing

Normalized image pixel values to the range [0, 1].
Converted labels to one-hot encoded vectors.
Model Architecture

Input: Flattened 32×32×3 images.
Dense layer with 512 neurons + ReLU activation.
Dropout (0.3) for regularization.
Dense layer with 256 neurons + ReLU activation.
Output layer with 10 neurons (softmax) for classification.
Training

Dataset: CIFAR-10 training set.
Loss: Categorical cross-entropy.
Optimizer: Adam.
Evaluation on CIFAR-10 test set.


4.Results
Achieved a good accuracy test set.
Successfully classified images into 10 categories.
Visualized sample predictions to confirm model learning.


5.Conclusion This project demonstrates a simple yet effective approach for CIFAR-10 classification using dense neural networks.
Although convolutional neural networks (CNNs) could improve performance, this architecture shows how dense layers can still capture patterns in image data after flattening. 

       Experiment 5
1.Problem Statement
The task is to classify images from the MNIST dataset (handwritten digits 0–9) using deep learning techniques. The goal is to train a neural network that can accurately predict the digit shown in an image.

2.Deep Learning Methods
The project applies Convolutional Neural Networks (CNNs), which are widely used for image classification tasks because they automatically extract spatial features like edges, shapes, and textures from images.

3.Description of Methods
Data Preprocessing
MNIST dataset is loaded, containing 28x28 grayscale digit images.
Pixel values are normalized to improve training efficiency.

CNN Architecture
Convolutional layers extract local image features.
Pooling layers reduce spatial dimensions while retaining important information.
Fully connected (dense) layers interpret these features for classification.
Softmax activation is used in the final layer to output probabilities for each digit (0–9).

Training Process
The dataset is split into training and test sets.
Model is trained using backpropagation with an optimizer (e.g., Adam).
Loss function used is categorical cross-entropy.
Methods Applied to Solve the Problem
Convolutional Neural Network (CNN): Used to extract features and classify digits.
Dropout Regularization: Prevents overfitting by randomly dropping neurons during training.
Activation Functions (ReLU & Softmax): Provide non-linearity and probability outputs.

4.Result
The model achieves high accuracy (typically above 98%) on the MNIST test dataset, showing that CNNs are effective for digit recognition tasks.

5.Conclusion
This project demonstrates the application of CNNs for handwritten digit classification. The results confirm that deep learning methods can achieve excellent performance on image recognition problems, and similar techniques can be extended to more complex datasets in real-world applications.
