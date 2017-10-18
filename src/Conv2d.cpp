#include "pytorch.h"

torch::Conv2d::Conv2d(
	int in_channels,
	int out_channels,
	int kernel_width,
	int kernel_height,
	int stride_width,
	int stride_height,
	int padding_width,
	int padding_height,
	int dilation_width,
	int dilation_height,
	int groups,
	int bias) :
	in_channels(in_channels),
	out_channels(out_channels),
	kernel_width(kernel_width),
	kernel_height(kernel_height),
	stride_width(stride_width),
	stride_height(stride_height),
	padding_width(padding_width),
	padding_height(padding_height),
	dilation_width(dilation_width),
	dilation_height(dilation_height),
	groups(groups),
	bias(bias)
{
	// Register "wight" as a parameter in order to be able to
	// restore it from a file later on
	parameters["weight"] = TENSOR_DEFAULT_TYPE.zeros({ out_channels,
		in_channels,
		kernel_width,
		kernel_height });

	// Check if we need bias for our convolution
	if (bias)
	{
		parameters["bias"] = TENSOR_DEFAULT_TYPE.zeros({ out_channels });
	}
	else
	{
		// Doesn't work with TENSOR_DEFAULT_TYPE.tensor();,
		// This is why we use Tensor()
		parameters["bias"] = Tensor();
	}

	// These variables are not needed for forward inferece,
	// but we need them in order to call an underlying C
	// function. Later they will be used for backward pass

	grads["finput"] = TENSOR_DEFAULT_TYPE.tensor();
	grads["fgradInput"] = TENSOR_DEFAULT_TYPE.tensor();

	// These variables depend on # of groups, so far only
	// one group is supported. Needs to be changed to tensor_list
	// in order to support multiple groups.
	grads["ones"] = TENSOR_DEFAULT_TYPE.tensor();
	grads["columns"] = TENSOR_DEFAULT_TYPE.tensor();

	// There are separate functions for dilated and non-dilated convolutions
	dilated = false;

	if ((dilation_width > 1) || (dilation_height > 1))
	{
		dilated = true;
	}

};

torch::Conv2d::~Conv2d()
{

};


string torch::Conv2d::tostring(int indentation_level)
{
	std::stringstream string_stream;

	string indentation = string(indentation_level, ' ');

	string_stream << indentation << "Conv2d( "
		<< "in_channels=" << std::to_string(in_channels) << " "
		<< "out_channels=" << std::to_string(out_channels) << " "
		<< "kernel_size=(" << std::to_string(kernel_width) << ", " << std::to_string(kernel_height) << ") "
		<< "stride=(" << std::to_string(stride_width) << ", " << std::to_string(stride_height) << ") "
		<< "padding=(" << std::to_string(padding_width) << ", " << std::to_string(padding_height) << ") "
		<< "dilation=(" << std::to_string(dilation_width) << ", " << std::to_string(dilation_height) << ") "
		<< "groups=" << std::to_string(groups) << " "
		<< "bias=" << std::to_string(bias) << " )";

	return string_stream.str();
};

Tensor torch::Conv2d::forward(Tensor input)
{
	Tensor output = input.type().tensor();

	if (dilated)
	{
		SpatialDilatedConvolution_updateOutput(
			input,
			output,
			parameters["weight"],
			parameters["bias"],
			grads["columns"],
			grads["ones"],
			kernel_width,
			kernel_height,
			stride_width,
			stride_height,
			padding_width,
			padding_height,
			dilation_width,
			dilation_height);
	}
	else
	{
		SpatialConvolutionMM_updateOutput(
			input,
			output,
			parameters["weight"],
			parameters["bias"],
			grads["finput"],
			grads["fgradInput"],
			kernel_width,
			kernel_height,
			stride_width,
			stride_height,
			padding_width,
			padding_height);
	}
	return output;
};