# torch-lego
Build pytorch modules using yaml description files. Yaml can be used to keep track of model architecture using tools such as Mlflow.
This library supports convolution (1d, 2d and 3d) modules, transpose convolution (1d,2d,3d) modules and linear modules. Average and max pooling layers are also supported as well as fully connected layers. Batch norm, dropout and all built-in PyTorch activations are available. See the examples in the test directory for more detail.

# Example
## Deep convolutional classifier
The example given below is a simple MNIST classifier. LeakyReLU are used as activations. The negative slope value is not specfied so the default PyTorch value will be used. The same applies for all non-specified parameters. Max pooling is used and the input size is automatically computed using parameters from the preceeding convolution layer.

```yaml
architecture:
  encoder:
    - conv:
        in_channels: 1
        out_channels: 36
        kernel_size: [3,3]
        stride: 1
        padding: 2
      mpool:
        kernel_size: [2,2]
      act:
        type: 'LeakyReLU'
        params:
    - conv:
        in_channels: 36
        out_channels: 100
        kernel_size: [3,3]
        stride: 1
        padding: 2
      mpool:
        kernel_size: [ 2,2 ]
      act:
        type: 'LeakyReLU'
        params:
      drpt: 0.5
  classifier:
    - in_features: 6400
    - out_features: 200
      act:
        type: 'ReLU'
        params:
      drpt: 0.4
    - out_features: 100
      act:
        type: 'ReLU'
        params:
      drpt: 0.4
    - out_features: 10
      act:
        type:
        params:
      drpt: 0.2
```


# Installation and test
This is a python package that can be installed using pip after cloning the repository:
```bash
pip install -e torch_lego
```
All tests can be run by executing the *all.sh* script in the *tests* directory.
