# ZOO_Attack_PyTorch
This repository contains the PyTorch implementation of Zeroth Order Black Box Attack using MNIST and CIFAR10 dataset. This is the exact replica as far possible of the ZOO Attack (https://github.com/IBM/ZOO-Attack) which was implemented in Tensorflow and Keras. The results match almost as same as the paper evaluation results for MNIST and CIFAR10 for both targeted and untargeted attack all with 100% success rate on the 7 layer CNNs model trained on MNIST with 99.5% val accuracy and on CIFAR10 with 80% val accuracy as done in the original paper work. Both ZOO_Adam and ZOO_Newton methods of Coordinate Descent Solvers are implemented.

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
Both untargeted and targeted attack are accessible via above code all the changes (comment/uncomment) for transition from ZOO_Adam/ZOO_Newton or CIFAR10/MNIST are from line 259-262, 270/271, 274-277 and for visualtion line 307/329. For more details go through the code zoo_l2_attack_black.py and the paper https://arxiv.org/abs/1708.03999

## Sample Results
#### ZOO_Adam 
##### Untargeted on CIFAR10
![](/sample_results/adam_untargeted_cifar10.png)
##### Untargeted on CIFAR10
![](/sample_results/adam_untargeted_mnist.png)
#### ZOO_Newton
##### Targeted on MNIST
![](/sample_results/newton_targeted_mnist.png)
