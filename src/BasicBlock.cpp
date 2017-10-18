#include "pytorch.h"

// Make a standart value
torch::BasicBlock::BasicBlock(int inplanes, int planes, int stride, int dilation, Module::Ptr downsample)
{
	conv1 = conv3x3(inplanes, planes, stride, dilation);
	bn1 = std::make_shared<BatchNorm2d>(planes);
	relu = std::make_shared<ReLU>();
	conv2 = conv3x3(planes, planes, 1, dilation);
	bn2 = std::make_shared<BatchNorm2d>(planes);

	// This doesn't work
	// downsample = downsample because
	// the argument gets assigned instead of a class member,
	// Should probably change the name of the member and argument
	// to be different
	this->downsample = downsample;

	stride = stride;

	add_module("conv1", conv1);
	add_module("bn1", bn1);
	add_module("conv2", conv2);
	add_module("bn2", bn2);

	if( downsample != nullptr )
	{

		add_module("downsample", downsample);
	}

	module_name = "BasicBlock";
};

torch::BasicBlock::~BasicBlock()
{

};

Tensor torch::BasicBlock::forward(Tensor input)
{
	// This is done in case we don't have the
	// downsample module
	Tensor residual = input;
	Tensor out;

	out = conv1->forward(input);
	out = bn1->forward(out);
	out = relu->forward(out);
	out = conv2->forward(out);
	out = bn2->forward(out);

	if(downsample != nullptr)
	{
     
		residual = downsample->forward(input);
	}

	out += residual;
	out = relu->forward(out);

	return out;
}