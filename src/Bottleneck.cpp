#include "pytorch.h"

// Make a standart value
torch::Bottleneck::Bottleneck(int inplanes, int planes, int stride, int dilation, Module::Ptr downsample)
{
	conv1 = resnet_conv1x1(inplanes, planes);
	bn1 = std::make_shared<BatchNorm2d>(planes);
	relu = std::make_shared<ReLU>();
       
	conv2 = conv3x3(planes, planes, stride, dilation);
	bn2 = std::make_shared<BatchNorm2d>(planes);

	conv3 = resnet_conv1x1(inplanes, planes * Bottleneck::expansion);
	bn3 = std::make_shared<BatchNorm2d>(planes * Bottleneck::expansion);

	// Avoiding ambiguitiy -- this is why we are using 'this' keyword.
	this->downsample = downsample;

	stride = stride;

	add_module("conv1", conv1);
	add_module("bn1", bn1);
	add_module("conv2", conv2);
	add_module("bn2", bn2);
	add_module("conv3", conv3);
	add_module("bn3", bn3);
       

	if( downsample != nullptr )
	{

		add_module("downsample", downsample);
	}

	module_name = "Bottleneck";

};

torch::Bottleneck::~Bottleneck()
{

};

Tensor torch::Bottleneck::forward(Tensor input)
{
	Tensor residual = input;
	Tensor out;

	out = conv1->forward(input);
	out = bn1->forward(out);
	out = relu->forward(out);
       
	out = conv2->forward(out);
	out = bn2->forward(out);
	out = relu->forward(out);

	out = conv3->forward(out);
	out = bn3->forward(out);


	if(downsample != nullptr)
	{
		residual = downsample->forward(input);
	}

	out += residual;
	out = relu->forward(out);

	return out;
}
