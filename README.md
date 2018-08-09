# Resnet: Classifying CIFAR-10 images 

An easy-to-follow TensorFlow implementation of Resnet model for classifying CIFAR-10 images. Can be useful for those interested in reproducing the results presented in the paper, or understanding how to build Resnet model in TensorFlow.

### Usage
##### Clone the Repo
```
git clone git@github.com:chao-ji/tf-resnet-cifar10.git
```
##### Download and untar CIFAR-10 dataset
```
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xvzf cifar-10-binary.tar.gz
```
##### Train the Resnet Classifier
To train the Resnet model using default settings, simply run
```
python run_trainer.py \
  --path=cifar-10-batches-bin
```
To change the number of layers in the Resnet (for example, to 110), specify `--num_layers=110`. To degenerate the Resnet model to a *Plain network*, specify `--shortcut_connections=False`. To see a full list of arguments, run
```
python run_trainer.py --help
```
##### Evaluate a Trained Resnet Classifier
To evaluate the trained model on the evaluation set (10,000 images), run
```
  python run_evaluator.py \
    --path=cifar-10-batches-bin \
    --ckpt_path=/PATH/TO/CKPT \
    --num_layers=110
```
Note that you need to specify the path to the checkpoint file containing trained weights via `--ckpt_path`.
### Results
We train three Resnets with the number of layers being 20, 56, and 110, and we evaluate their accuracy on the evaluation set (10,000 images). As we can see, deeper versions of Resnet model achieves better accuracy compared with the shallower version, while the trend is opposite for the Plain network that comes with no residual connections.


##### Residual Net
<img src="files/resnet.png" width="500">

##### Plain Net
<img src="files/plain.png" width="500">


### Build Resnet Model
##### Initial Conv layer
The Resnet model backbone starts with an initial conv layer with no bias:
<img src="files/init_conv.png" width="400">

##### Blocks and Units
Then it stacks up three similarly structured *blocks*, which gradually halves the sizes of height and width, and doubles the depth dimension. Each block has multiple repeating *units*, each of which contains two conv layers:
<img src="files/unit.png" width="500">

Note the conv layers in the repeating units are **preactivated** -- the batch-norm layer and nonlinearity *precedes* the conv layers as opposed to following them, which is used in *Resnet v2*. <sup>[1](#myfootnote2)</sup>

##### Shortcut Connection
At the joints between two neighboring blocks, the sizes of height and width are halved while the depth is doubled. To ensure the shapes are compatible, we use the **Identity Shortcut** option described in the paper -- in the shortcut branch, we average-pool the incoming feature map with `stride=2, kernel_size=2, pad='SAME'`, and then zero-pad the depth dimension, so that the two ends of shortcut connection have the same shape. 

<img src="files/shortcut.png" width="500">

##### Global Pooling and Projection Layer
The backbone ends with a *global average pooling*, which reduces the spatial dimension from `8x8` to `1x1`, and generates the prediction logits via a projection layer (implemented as an affine transformation).

<img src="files/final.png" width="450">

The full diagram of a 20-layer Resnet can be found [here](./files/ResNet20_CIFAR10.pdf)

### References:
  1. <a name="myfootnote1">Resnet V1</a>, Deep Residual Learning for Image Recognition, He *et al.*
  2. <a name="myfootnote2">Resnet V2</a>, Identity Mappings in Deep Residual Networks, He *et al.*
