# PyTorch Network Slimming

This repo implements the following paper in [PyTorch](https://pytorch.org): 

[**Learning Efficient Convolutional Networks Through Network Slimming**](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)

Different with most other popular network slimming repos, this implementation enables training & pruning models in hand with a few lines of new codes. Writing codes specifically for pruning is not required. 

BN layers are automatically identified by 1) parse the traced graph by [TorchScript](https://pytorch.org/docs/stable/jit.html), and 2) identify **prunable BN layers** which only have Convolution(groups=1)/Linear in preceding & succeeding layers. Example of a **prunable BN**:

            conv/linear --> ... --> BN --> ... --> conv/linear
                                     |
                                    ...
                                     | --> relu --> ... --> conv/linear
                                    ...
                                     | --> ... --> maxpool --> ... --> conv/linear
For more details please refer to source code. 

It is supposed to support user defined models with Convolution(groups=1)/Linear and BN layers. The package is tested with the [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) examples in this repo, and an in-house [Conv3d](https://pytorch.org/docs/stable/nn.html#conv3d) based model for video classification. 

## Requirements

Python >= 3.6
torch >= 1.0.0
torchvision >= 0.3.0

## Usage

1. Import from [netslim](./netslim) in your training script
   
     ```python
   from netslim import update_bn
   ```
   
2. Insert the following code between loss.backward() and optimizer.step(). The following is an example:

   *before*

   ```python
   ...
   loss.backward()
   optimizer.step()
   ...
   ```

   *after*

      ```python
   ...
   loss.backward()
   update_bn(model)  # or update_bn(model, s), specify s to control sparsity on BN
   optimizer.step()
   ...
      ```

   <font size=2> \* ***update_bn*** puts L1 regularization on all BNs. Sparsity on prunable BNs only is also supported for networks with complex connections, such as ResNet. Check examples for more details. </font>

3. Prune the model after training

   ```python
   from netslim import prune
   # For example, input_shape for CIFAR is (3, 32, 32)
   pruned_model = prune(model, input_shape, prune_ratio=0.5)
   ```

4. Fine-tune & export model

5. Load the pruned model and have fun

   ```
   from netslim import load_pruned_model
   model = MyModel()
   weights = torch.load("/path/to/pruned-weights.pth")
   pruned_model = load_pruned_model(model, weights)
   ...
   ```

## Run CIFAR-100 examples

### ResNet-18

   ```shell
./experiment-resnet.sh
   ```

### VGG-11

   ```shell
./experiment-vgg11.sh
./experiment-vgg11s.sh % simplified VGG-11 by replacing classifier with a Linear
   ```

### DenseNet-121

   ```shell
./experiment-densenet.sh
   ```

<font size=2> ***-all-bn*** refers to L1 sparsity on all BN layers </font>

## Results on CIFAR-100

### Params (M)

|                           | Original | PR=0.3 | PR=0.5 | PR=0.7 |
| :------------------------ | :------: | :----: | :----: | :----: |
| ResNet-18                 |          |        |        |        |
| ResNet-18 (L1 on all BNs) |          |        |        |        |
| VGG-11                    |          |        |        |        |
| simplified VGG-11         |          |        |        |        |
| DenseNet-63               |          |        |        |        |

### GFLOPS

|                           | Original | PR=0.3 | PR=0.5 | PR=0.7 |
| :------------------------ | :------: | :----: | :----: | :----: |
| ResNet-18                 |          |        |        |        |
| ResNet-18 (L1 on all BNs) |          |        |        |        |
| VGG-11                    |          |        |        |        |
| simplified VGG-11         |          |        |        |        |
| DenseNet-63               |          |        |        |        |

### Accuracy (%)

|                           | Original | L1 on BN | PR=0.3 | PR=0.5 | PR=0.7 |
| :------------------------ | :------: | :------: | :----: | :----: | :----: |
| ResNet-18                 |          |          |        |        |        |
| ResNet-18 (L1 on all BNs) |          |          |        |        |        |
| VGG-11                    |          |          |        |        |        |
| simplified VGG-11         |          |          |        |        |        |
| DenseNet-63               |          |          |        |        |        |

<font size=2>\* **TFS**: Train-from-scratch as proposed in Liu's later paper on ICLR 2019 [**Rethinking the Value of Network Pruning**](https://openreview.net/forum?id=rJlnB3C5Ym). </font>

<font size=2> \* **PR**: Prune Ratio </font>

## Acknowledgement

The implementation of ***udpate_bn*** is referred to [pytorch-slimming](https://github.com/foolwood/pytorch-slimming).

