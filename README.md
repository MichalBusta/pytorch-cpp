# Pytorch-C++

```Pytorch-C++``` is a simple C++ 11 library which provides a [Pytorch](http://pytorch.org/)-like
interface for building neural networks and inference (so far only forward pass is supported). The library
respects the semantics of ```torch.nn``` module of PyTorch. Models from [pytorch/vision](https://github.com/pytorch/vision)
are supported and can be [easily converted](convert_weights.ipynb).

The library heavily relies on an amazing [ATen](https://github.com/zdevito/ATen) library and was inspired by
[cunnproduction](https://github.com/szagoruyko/cunnproduction).

The structure of the project and CMake will be changed in a future, as it is not optimal now.

## Table of contents

<a href="#use-cases">Use-cases</a><br>
<a href='#some-examples'>Examples</a><br>
<a href='#implemented-layers'>Implemented layers</a><br>
<a href='#implemented-models'>Implemented models</a><br>
<a href='#demos'>Demos</a><br>
<a href='#installation'>Installation</a><br>
<a href='#about'>About</a><br>
<a href='#contributors'>Contributors</a><br>


## Use-cases

The library can be used in cases where you want to integrate your trained ```Pytorch```
networks into an existing C++ stack and you don't want to convert your weights to other libraries
like ```Caffe/Caffe2/Tensorflow```. The library respects the semantics of the ```Pytorch``` and uses
the same underlying C library to perform all the operations.

You can achieve more low-level control over your memory. For example,
you can use a memory that was already allocated on GPU. This way you can accept memory from other
application on GPU and avoid expensive transfer to CPU. See [this example](examples/read_allocated_gpu_memory.cpp).

Conversion from other image types like OpenCV's ```mat``` to ```Tensor``` can be easily performed and all the post-processing
can be done using numpy-like optimized operations, thanks to [ATen](https://github.com/zdevito/ATen) library.
See examples [here](examples/opencv_realtime_webcam_human_segmentation.cpp).


## Some examples

### Inference

```c++
auto net = torch::resnet50_imagenet();

net->load_weights("../resnet50_imagenet.h5");

# Transfer network to GPU
net->cuda();

# Generate a dummy tensor on GPU of type float
Tensor dummy_input = CUDA(kFloat).ones({1, 3, 224, 224});

# Perform inference
auto result = net->forward(dummy_input);

map<string, Tensor> dict;

# Get the result of the inference back to CPU
dict["main"] = result.toBackend(Backend::CPU);

# Save the result of the inference in the HDF5 file
torch::save("resnet50_output.h5", dict);
```

### Display network's architecture

```c++

auto net = torch::resnet50_imagenet();

net->load_weights("../resnet50_imagenet.h5");

cout << net->tostring() << endl;

```

Output:

```
ResNet (
 (conv1)  Conv2d( in_channels=3 out_channels=64 kernel_size=(7, 7) stride=(2, 2) padding=(3, 3) dilation=(1, 1) groups=1 bias=0 )
 (bn1)  BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
 (relu)  ReLU
 (maxpool)  MaxPool2d( kernel_size=(3, 3) stride=(2, 2) padding=(1, 1) )
 (layer1)  Sequential (
  (0)   Bottleneck (
   (conv1)    Conv2d( in_channels=64 out_channels=64 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
   (bn1)    BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
   (conv2)    Conv2d( in_channels=64 out_channels=64 kernel_size=(3, 3) stride=(1, 1) padding=(1, 1) dilation=(1, 1) groups=1 bias=0 )
   (bn2)    BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
   (conv3)    Conv2d( in_channels=64 out_channels=256 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
   (bn3)    BatchNorm2d( num_features=256 eps=0.000010 momentum=0.100000 )
   (downsample)    Sequential (
    (0)     Conv2d( in_channels=64 out_channels=256 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
    (1)     BatchNorm2d( num_features=256 eps=0.000010 momentum=0.100000 )
   )

  )

  (1)   Bottleneck (
   (conv1)    Conv2d( in_channels=256 out_channels=64 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
   (bn1)    BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
   (conv2)    Conv2d( in_channels=64 out_channels=64 kernel_size=(3, 3) stride=(1, 1) padding=(1, 1) dilation=(1, 1) groups=1 bias=0 )
   (bn2)    BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
   (conv3)    Conv2d( in_channels=256 out_channels=256 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
   (bn3)    BatchNorm2d( num_features=256 eps=0.000010 momentum=0.100000 )
  )

  (2)   Bottleneck (
   (conv1)    Conv2d( in_channels=256 out_channels=64 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
   (bn1)    BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
   (conv2)    Conv2d( in_channels=64 out_channels=64 kernel_size=(3, 3) stride=(1, 1) padding=(1, 1) dilation=(1, 1) groups=1 bias=0 )
   (bn2)    BatchNorm2d( num_features=64 eps=0.000010 momentum=0.100000 )
   (conv3)    Conv2d( in_channels=256 out_channels=256 kernel_size=(1, 1) stride=(1, 1) padding=(0, 0) dilation=(1, 1) groups=1 bias=0 )
   (bn3)    BatchNorm2d( num_features=256 eps=0.000010 momentum=0.100000 )
  )

 )

 /*  .... */

 (avgpool)  AvgPool2d( kernel_size=(7, 7) stride=(1, 1) padding=(0, 0) )
 (fc)  nn.Linear( in_features=2048 out_features=1000 bias=1 )
)
```

### Inspect a Tensor


```c++
auto net = torch::resnet50_imagenet();

net->load_weights("../resnet50_imagenet.h5");
net->cuda();

Tensor dummy_input = CUDA(kFloat).ones({1, 3, 224, 224});

auto result = net->forward(dummy_input);

cout << result << endl;
```


```
Columns 1 to 10-0.3081  0.0798 -1.1900 -1.4837 -0.5136  0.3683 -2.1639 -0.8705 -1.8812 -0.1608

Columns 11 to 20 0.2168 -0.9283 -1.2954 -1.0791 -1.4445 -0.8946 -0.0959 -1.3099 -1.2062 -1.2327

Columns 21 to 30-1.0658  0.9427  0.5739 -0.2746 -1.0189 -0.3583 -0.1826  0.2785  0.2209 -0.3340

Columns 31 to 40-1.9800 -0.5552 -1.0804 -0.8056 -0.0005 -1.8402 -0.7979 -1.4823  1.3657 -0.8970

/*  .... */

Columns 961 to 970-0.0557 -0.7405 -0.5501 -1.7207 -0.7043 -1.0925  1.5812 -0.1215  0.8915  0.9794

Columns 971 to 980-1.1422 -0.1235 -0.5999 -2.1338 -0.0775 -0.8374 -0.2350 -0.0104 -0.0416 -1.0296

Columns 981 to 990-0.2914 -0.2242 -0.8063 -0.7818 -0.2714  0.0002 -1.2355  0.1238  0.0183 -0.6904

Columns 991 to 1000 0.5216 -1.8008 -1.7826 -1.2970 -1.6565 -1.3306 -0.6564 -1.6531  0.1178  0.2436
[ CUDAFloatTensor{1,1000} ]
```

### Create a network


```c++
auto new_net = std::make_shared<torch::Sequential>();
new_net->add(std::make_shared<torch::Conv2d>(3, 10, 3, 3));
new_net->add(std::make_shared<torch::BatchNorm2d>(10));
new_net->add(std::make_shared<torch::ReLU>());
new_net->add(std::make_shared<torch::Linear>(10, 3));
```
## Implemented layers

So far, these layers are available which respect the Pytorch's layers semantics which
can be found [here](http://pytorch.org/docs/0.1.12/nn.html#convolution-layers).


- [x] nn.Sequential
- [x] nn.Conv2d
- [x] nn.MaxPool2d
- [x] nn.AvgPool2d
- [x] nn.ReLU
- [x] nn.Linear
- [x] nn.SoftMax
- [x] nn.BatchNorm2d
- [ ] nn.Dropout2d
- [ ] nn.DataParallel
- [ ] nn.AdaptiveMaxPool2d
- [ ] nn.Sigmoid
and others.

## Implemented models

Some convered models are provided for ease of access. Other models can be [easily converted](convert_weights.ipynb).

### Imagenet models

All models were converted from [pytorch/vision](https://github.com/pytorch/vision) and checked for
correctness.

- [x] Resnet-18
- [x] Resnet-34
- [x] [Resnet-50](https://www.dropbox.com/s/bukezzx17dr8qdd/resnet50_imagenet.h5?dl=0)
- [x] Resnet-101
- [x] Resnet-150
- [x] Resnet-152
- [ ] All VGG models
- [ ] All Densenet models
- [ ] All Inception models
- [ ] All squeezenet models
- [ ] Alexnet

### Segmentation PASCAL VOC 

All models were converted from [this repository](https://github.com/warmspringwinds/dense-ai) and checked for
correctness.

- [x] Resnet-18-8S
- [x] [Resnet-34-8S](https://www.dropbox.com/s/104my8hr5zm6l7d/resnet34_fcn_pascal.h5?dl=0)
- [ ] Resnet-50-8S
- [ ] Resnet-101-8S
- [ ] Resnet-152-8S
- [x] FCN-32s
- [ ] FCN-16s
- [ ] FCN-8s

## Demos

We created a couple of [demos](examples) where we grab frames using opencv and classify
or segment them.

Here you can see and example of real-time segmentation:


![Alt text](examples/segmentation_demo_preview.gif?raw=true "Title")

## Installation
Test with CMake 3.8. Get error on CMake 3.9 as HDF5 find_hdf5 has changed.

### ATen
[ATen](https://github.com/zdevito/ATen) is a C++ 11 library that wraps a powerfull C Tensor library with
implementation of numpy-like operations (CPU/CUDA/SPARSE/CUDA-SPARSE backends).
Follow these steps to install it:  

#### Linux
1. Make sure you have [dependencies](https://github.com/zdevito/ATen#installation) of ```ATen``` installed.
2. ```git clone --recursive https://github.com/warmspringwinds/pytorch-cpp```
2. ```cd pytorch-cpp/ATen;mkdir build;cd build;cmake-gui .. ``` and specify ```CUDA_TOOLKIT_ROOT_DIR```.
3. ```make``` or better ```make -j7``` (replace ```7``` with a number of cores that you have).
4. ```cd ../../``` -- returns you back to the root directory (necessary for the next step).

#### Windows
Only tested to build with CUDA for pytorch-cpp. Test on your own with CPU only build

1. ```git clone https://github.com/jackyko1991/ATen```
2. CMake configure and generate
2. Build and Install

### HDF5
#### Linux
We use ```HDF5``` to be able to [easily convert](convert_weights.ipynb) weigths between ```Pytorch``` and ```Pytorch-C++```.

1. ```wget https://support.hdfgroup.org/ftp/HDF5/current18/src/CMake-hdf5-1.8.19.tar.gz; tar xvzf CMake-hdf5-1.8.19.tar.gz```
2. ```cd CMake-hdf5-1.8.19; ./build-unix.sh```
2. ```cd ../``` -- return back.

#### Windows
1. Download ```HDF5``` source code from office [HDF5 site](https://support.hdfgroup.org/HDF5/release/cmakebuild518.html)
2. CMake configure, BUILD_SHARED_LIBS is recommended, example and test are not necessary
2. Generate, build
3. Install (Optional but highly recoommended)

### OpenCV
####Linux and Windows (build from source)
[OpenCV 3](https://github.com/opencv/opencv.git) is more preferred as CUDA runtime bug found in OpenCV v2.4.13

#### Ubuntu (Binary)
We need ```OpenCV``` for a couple of examples which grab frames from a web camera.
It is not a dependency and can be removed if necessary.
This was tested on ```Ubuntu-16``` and might need some changes on a different system.

```sudo apt-get install libopencv-dev python-opencv```

#### Windows (Binary)
https://opencv.org/releases.html

### Pytorch-C++
```Pytorch-C++``` is a library on top of ```ATen``` that provides a [Pytorch](http://pytorch.org/)-like
interface for building neural networks and inference (so far only forward pass is supported)
inspired by [cunnproduction](https://github.com/szagoruyko/cunnproduction) library. To install it, follow
these steps:

#### Linux
1. ```mkdir build; cd build; cmake-gui ..``` and specify ```CUDA_TOOLKIT_ROOT_DIR```.
2. ```make```
2. ```cd ../``` -- return back

#### Windows
1. CMake and specify ```OpenCV```, ```HDF5``` dependencies
2. Generate and build (Currently only support release build, since ```ATen``` libraries does not build the Debug and Release separately)

### Problems with the build

It was noticed that if you have anaconda installed and your ```PATH``` variable is modified to include
its folder, it can lead to failed buid (caused by the fact that anaconda uses different version of ```gcc```).
To solve this problem, remove the path to anaconda from ```PATH``` for the time of the build.

If you face any problems or some steps are not clear, please open an issue. Note: every time you enter the ```cmake-gui```
press ```configure``` first, then specify your ```CUDA``` path and then press ```generate```, after that you can build.


## About

If you used the code for your research, please, cite the paper:

    @article{pakhomov2017deep,
      title={Deep Residual Learning for Instrument Segmentation in Robotic Surgery},
      author={Pakhomov, Daniil and Premachandran, Vittal and Allan, Max and Azizian, Mahdi and Navab, Nassir},
      journal={arXiv preprint arXiv:1703.08580},
      year={2017}
    }

During implementation, some preliminary experiments and notes were reported:
- [Converting Image Classification network into FCN](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/10/30/image-classification-and-segmentation-using-tensorflow-and-tf-slim/)
- [Performing upsampling using transposed convolution](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/)
- [Conditional Random Fields for Refining of Segmentation and Coarseness of FCN-32s model segmentations](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/)
- [TF-records usage](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)

## Contributors

- Daniil Pakhomov
