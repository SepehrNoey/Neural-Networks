## Neural Networks implementation from Scratch
This project separates to six parts. At the beginning, we used PyTorch to implement a classifier for MNIST handwritten digits, and afterwards, we implemented a framework from scratch to train and use neural networks for inference. Details of each task are as follows: 

### Task 1 & 2: Handwritten Digits Recognition using Fully Connected Neural Networks and CNNs
- **ReLU** activation function
- Stochastic Gradient Descent(**SGD**) optimizer
- **Cross Entropy** Loss Function
- Dataset: **MNIST** handwritten digits
- Validation data accuracy 97%

Sample images from MNIST:

![](https://github.com/SepehrNoey/Neural-Networks-Framework-from-Scratch/blob/main/MNIST-Samples.jpg)

Train and validation accuracy of trained CNN:

![](https://github.com/SepehrNoey/Neural-Networks-Framework-from-Scratch/blob/main/loss_mnist_cnn.png)

### Task 3: Brain Tumor Recognition
- Model: **ResNet50**
- Optimizer: **Adam**
- Loss Function: **Cross Entropy** 

Dataset sample:

![](https://github.com/SepehrNoey/Neural-Networks-Framework-from-Scratch/blob/main/brain_sample_data.png)

Training loss plot:

![](https://github.com/SepehrNoey/Neural-Networks-Framework-from-Scratch/blob/main/resnet50_loss_function.png)

### Task 4 to 5: Implementing Framework for Neural Networks Implementation
- **Tensor Class**: Included mathematical operations and ability to save gradients
- Implemented layer: Linear Layer
- Implemented Optimizers: **SGD**, **Momentum**, **Adam**, and **RMSprop**
- Implemented Loss Functions: **Mean Squared Error**, **Categorical Cross Entropy**
- Implemented Activation Functions: **ReLU**, **LeakyReLU**, **Tanh**, and **Sigmoid**

### Task 6: Using Implemented Framework for MNIST Handwritten Digit Recognition
In this part, we use our implemented 
#### How to use:
For testing this part, you can easily run `mnist-task.py` which uses the implemented framework to train a simple neural network with two linear layers:

- Used two linear layers with hidden neuron numbers of `100`
- Used Adam optimizer with learning rate of `0.1` with batch size of `100`
- Used `MSE` loss function
