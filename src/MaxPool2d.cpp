#include "pytorch.h"

torch::MaxPool2d::MaxPool2d(
	int kernel_width,
	int kernel_height,
	int stride_width,
	int stride_height,
	int padding_width,
	int padding_height,
	bool ceil_mode) :
	kernel_width(kernel_width),
	kernel_height(kernel_height),
	stride_width(stride_width),
	stride_height(stride_height),
	padding_width(padding_width),
	padding_height(padding_height),
	ceil_mode(ceil_mode)
{
	// TODO: so far this one is hardcoded.
	// Change to make it gpu or cpu depending
	// on the network placement
	grads["indices"] = CPU(kLong).tensor();
};

torch::MaxPool2d::~MaxPool2d()
{

};

Tensor torch::MaxPool2d::forward(Tensor input)
{
	Tensor output = input.type().tensor();

	SpatialMaxPooling_updateOutput(input,
		output,
		grads["indices"],
		kernel_width,
		kernel_width,
		stride_width,
		stride_height,
		padding_width,
		padding_height,
		ceil_mode);

	return output;
};

string torch::MaxPool2d::tostring(int indentation_level)
{
	std::stringstream string_stream;

	string indentation = string(indentation_level, ' ');

	string_stream << indentation
		<< "MaxPool2d( "
		<< "kernel_size=(" << std::to_string(kernel_width) << ", " << std::to_string(kernel_height) << ") "
		<< "stride=(" << std::to_string(stride_width) << ", " << std::to_string(stride_height) << ") "
		<< "padding=(" << std::to_string(padding_width) << ", " << std::to_string(padding_height) << ") )";

	return string_stream.str();
};