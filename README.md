# ZOO: Zeroth Order Optimization Based Adversarial Black Box Attack (PyTorch)
This repository contains the PyTorch implementation of Zeroth Order Optimization Based Adversarial Black Box Attack(https://arxiv.org/abs/1708.03999) using MNIST and CIFAR10 dataset. This is the exact replica as far possible of the ZOO Attack (https://github.com/IBM/ZOO-Attack) which was originally implemented in Tensorflow. The results match almost as same as the paper evaluation results for MNIST and CIFAR10 for both targeted and untargeted attack all with 100% success rate on the 7 layer CNNs model trained on MNIST with 99.5% val accuracy and on CIFAR10 with 80% val accuracy as done in the original paper work. Both ZOO_Adam and ZOO_Newton methods of Coordinate Descent Solvers are implemented.

**Note: This doesn't contain implementation of importance sampling, hierarchical attack, and dimentional reduction right now (as its mainly needed for large image sized dataset like ImageNet). For larger dataset google colab viewer [link](https://colab.research.google.com/drive/1_rDu0LNAI9kg490rV9PMxb55Wwbj13aD?usp=sharing) [NOT TESTED]**

## Setup and train models
The code is tested with Python 3.7.6 and PyTorch 1.6.0. The following packages are required:
```
python pip install --upgrade pip
pip install torch==1.6.0 torchsummary==1.5.1 torchvision==0.7.0
pip install numpy matplotlib 
```
To prepare model and datasets of MNIST and CIFAR10
```
python setup_mnist_model.py
python setup_cifar10_model.py
```
## Run attacks
To run the attacks run the 
```
python zoo_l2_attack_black.py
```
Both untargeted and targeted attack are accessible via above code all the changes (comment/uncomment) for transition from ZOO_Adam/ZOO_Newton or CIFAR10/MNIST are from line 259-262, 270/271, 274-277 and for visualization of example generated, line 307/329. For more details go through the code zoo_l2_attack_black.py and the paper https://arxiv.org/abs/1708.03999

## Sample Results
#### ZOO_Adam 
##### Untargeted on CIFAR10
![](/sample_results/adam_untargeted_cifar10.png)
##### Untargeted on CIFAR10
![](/sample_results/adam_untargeted_mnist.png)
#### ZOO_Newton
##### Targeted on MNIST
![](/sample_results/newton_targeted_mnist.png)


