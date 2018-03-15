#ifndef PYTORCH_H
#define PYTORCH_H

#include "ATen/ATen.h"
#include "ATen/Type.h"

#include <sstream>
#include <map>
#include "H5Cpp.h"

#include <opencv2/opencv.hpp>

#define TENSOR_DEFAULT_TYPE CUDA(kFloat)

using namespace at;


using std::map;
using std::string;
using std::vector;
using std::pair;
using std::shared_ptr;
using std::make_shared;
using std::cout;
using std::endl;
using std::tie;

using namespace cv;

namespace torch
{
	//IO
	map<string, Tensor> load(string hdf5_filename);
	void save(string hdf5_filename, map<string, Tensor> dict_to_write);
	vector<string> get_hdf5_file_keys(string hdf5_filename);
	void inspect_checkpoint(string hdf5_filename);

	 // TODO: explicit usage of Opencv's mat -- maybe try to hide it
	 // so that opencv won't be necessary for the main code

	 // Just put it in the opencv example file instead of the main library
	Tensor convert_opencv_mat_image_to_tensor(Mat input_mat);

	class Module
	{
	public:


		// Sequential module needs the counter
		// as names of submodules are not provided
		// sometimes.
		int submodule_counter;
		string module_name = "Module";

		Module();
		~Module();

		// We will use pointer to other modules a lot
		// This is done to automatically handle deallocation of created
		// module objects
		typedef shared_ptr<Module> Ptr;
		virtual Tensor forward(Tensor input);

		// This function gets overwritten
		// for the leafnodes like Conv2d, AvgPool2d and so on
		virtual string tostring(int indentation_level = 0);

		// vector<pair<string, Ptr>> because we want to emulate
		// the ordered dict this way, meaning that elements
		// are stored in the same order they were added

		// Like in Pytorch each module stores the modules that it uses
		vector<pair<string, Ptr>> modules;

		// And parameters that are explicitly used by the current module
		map<string, Tensor> parameters;

		// Plus buffers which are meant to store running mean and var for batchnorm layers
		map<string, Tensor> buffers;

		// We store parameter related to gradient computation here and other
		// tensors so far
		// TODO: some members of grads are not related to gradient computation
		//       and were put there temporary -- put them in a more relevant container.
		map<string, Tensor> grads;

		// A function to add another modules inside current module
		// Acts as Pytorch's Module.add_module() function
		void add_module(string module_name, Module::Ptr module);

		// Sometimes, when modules are being added, not all of them
		// have weights, like RELU. In this case the weights can be
		// numerated out of order. For example:
		// net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2))
		// net.state_dict().keys()
		// output: ['0.weight', '0.bias', '2.weight', '2.bias']

		// Equivalent behaviour will be seen with the add() function
		// described below: if relu is added, the counter for weights will
		// be increased.
		void add(Module::Ptr module);

		map<string, Tensor> state_dict(map<string, Tensor> & destination, string prefix = "");
		template<typename Func>	void apply(Func closure);
		void cuda();
		void cpu();
		void save_weights(string hdf5_filename);
		void load_weights(string hdf5_filename);
	};

	class Sequential : public Module
	{
	public:
		Sequential();
		~Sequential();
		// Forward for sequential block makes forward pass
		// for each submodule and passed it to the next one
		Tensor forward(Tensor input);
		Module::Ptr get(int i) const;
	};

	class ReLU : public Module
	{
	public:
		ReLU();
		~ReLU();

		Tensor forward(Tensor input);
		string tostring(int indentation_level = 0);
	};

	class Conv2d : public Module
	{
	public:
		int in_channels;
		int out_channels;
		int kernel_width;
		int kernel_height;
		int stride_width;
		int stride_height;
		int dilation_width;
		int dilation_height;
		int padding_width;
		int padding_height;
		int groups;
		int bias;
		bool dilated;

		Conv2d(
			int in_channels,
			int out_channels,
			int kernel_width,
			int kernel_height,
			int stride_width = 1,
			int stride_height = 1,
			int padding_width = 0,
			int padding_height = 0,
			int dilation_width = 1,
			int dilation_height = 1,
			int groups = 1,
			int bias = true); 
		~Conv2d();
		
		string tostring(int indentation_level = 0);
		Tensor forward(Tensor input);

	};

	class BatchNorm2d : public Module
	{
	public:
		int num_features;
		bool affine;
		bool training;
		double momentum;
		double eps;

		BatchNorm2d(
			int num_features,
			double eps = 1e-5,
			double momentum = 0.1,
			bool affine = true,
			bool training = false);
		~BatchNorm2d();

		string tostring(int indentation_level = 0);
		Tensor forward(Tensor input);
	};

	class MaxPool2d : public Module
	{
	public: 
		Tensor indices;

		bool ceil_mode;
		int kernel_width;
		int kernel_height;
		int stride_width;
		int stride_height;
		int padding_width;
		int padding_height;

		MaxPool2d(
			int kernel_width,
			int kernel_height,
			int stride_width = 1,
			int stride_height = 1,
			int padding_width = 0,
			int padding_height = 0,
			bool ceil_mode = false);
		~MaxPool2d();
		string tostring(int indentation_level = 0);
		Tensor forward(Tensor input);
	};

	class AvgPool2d: public Module
	{
	public:
		bool ceil_mode;
		bool count_include_pad;
		int kernel_width;
		int kernel_height;
		int stride_width;
		int stride_height;
		int padding_width;
		int padding_height;

		AvgPool2d(
			int kernel_width,
			int kernel_height,
			int stride_width=1,
			int stride_height=1,
			int padding_width=0,
			int padding_height=0,
			bool ceil_mode=false,
			bool count_include_pad=true);
		~AvgPool2d();
		Tensor forward(Tensor input);
		string tostring(int indentation_level = 0);

	};

	class Linear : public Module
	{
	public:
		int in_features;
		int out_features;
		bool bias;

		Linear(
			int in_features,
			int out_features,
			bool bias = true);
		~Linear();

		string tostring(int indentation_level = 0);
		Tensor forward(Tensor input);
	};

	class BasicBlock : public Module
	{
	public:
		static const int expansion = 1;

		int stride;
		Module::Ptr conv1;
		Module::Ptr bn1;
		Module::Ptr relu;
		Module::Ptr conv2;
		Module::Ptr bn2;
		Module::Ptr downsample;

		BasicBlock(int inplanes, int planes, int stride = 1, int dilation = 1, Module::Ptr downsample = nullptr);
		~BasicBlock();
		Tensor forward(Tensor input);
	};

	class Bottleneck : public Module
	{
	public:
		static const int expansion = 4;

		int stride;
		Module::Ptr conv1;
		Module::Ptr bn1;
		Module::Ptr relu;
		Module::Ptr conv2;
		Module::Ptr bn2;
		Module::Ptr conv3;
		Module::Ptr bn3;
		Module::Ptr downsample;

		Bottleneck(int inplanes, int planes, int stride = 1, int dilation = 1, Module::Ptr downsample = nullptr);
		~Bottleneck();

		Tensor forward(Tensor input);
	};

	Module::Ptr resnet_base_conv7x7();

	template< class BlockType>
	class ResNet : public Module
	{
	public:
		int output_stride;
		int in_planes;

		// Helper variables to help track
		// dilation factor and output stride
		int current_stride;
		int current_dilation;

		// Variables realted to the type of architecture.
		// Image Segmentation models don't have average pool
		// layer and Linear layers are converted to 1x1 convolution
		bool fully_conv;
		bool remove_avg_pool;

		Module::Ptr conv1;
		Module::Ptr bn1;
		Module::Ptr relu;
		Module::Ptr maxpool;
		Module::Ptr layer1;
		Module::Ptr layer2;
		Module::Ptr layer3;
		Module::Ptr layer4;
		Module::Ptr avgpool;
		Module::Ptr fc;

		ResNet(
			IntList layers,
			int num_classes = 1000,
			bool fully_conv = false,
			bool remove_avg_pool = false,
			int output_stride = 32);
		~ResNet();
		Tensor forward(Tensor input);
		Module::Ptr make_layer(int planes, int blocks, int stride);
	};

	#include "ResNet.hxx"

	// TODO: move this thing out in a separate logical unit: models/resnet

	// Helper functions for a 3 by 3 convolution without bias
	// Which is used in every resnet architecture.
	Tensor compute_full_padding_for_dilated_conv(Tensor kernel_size, int dilation = 1);
	Module::Ptr conv3x3(int in_planes, int out_planes, int stride = 1, int dilation = 1);
	Module::Ptr resnet_conv1x1(int in_planes, int planes);
	Tensor preprocess_batch(Tensor input_batch);
	Tensor convert_image_to_batch(Tensor input_img);

	//network architecture
	Module::Ptr resnet18(int num_classes, bool fully_conv, int output_stride, bool remove_avg_pool);
	Module::Ptr resnet34(int num_classes, bool fully_conv, int output_stride, bool remove_avg_pool);
	Module::Ptr resnet50(int num_classes, bool fully_conv, int output_stride, bool remove_avg_pool);
	Module::Ptr resnet101(int num_classes, bool fully_conv, int output_stride, bool remove_avg_pool);
	Module::Ptr resnet152(int num_classes, bool fully_conv, int output_stride, bool remove_avg_pool);

	// This one is just to build architecture, we can create functions to actually load
	// pretrained models like in pytorch
	class Resnet18_8s : public Module
	{
	public:
		int num_classes;
		Module::Ptr resnet18_8s;

		Resnet18_8s(int num_classes = 21);
		~Resnet18_8s();

		Tensor forward(Tensor input);
	};

	class Resnet34_8s : public Module
	{
	public:
		int num_classes;
		Module::Ptr resnet34_8s;

		Resnet34_8s(int num_classes = 21);
		~Resnet34_8s();

		Tensor forward(Tensor input);
	};

	// Maybe add new options like add_softmax?
	// imagenet
	Module::Ptr resnet18_imagenet();
	Module::Ptr resnet34_imagenet();
	Module::Ptr resnet50_imagenet();
	Module::Ptr resnet101_imagenet();
	Module::Ptr resnet152_imagenet();

	// Pascal VOC
	Module::Ptr resnet18_8s_pascal_voc();
	Module::Ptr resnet34_8s_pascal_voc();
}

#endif // !PYTORCH_H
