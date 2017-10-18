#include "pytorch.h"

torch::AvgPool2d::AvgPool2d(int kernel_width,
        int kernel_height,
        int stride_width,
        int stride_height,
        int padding_width,
        int padding_height,
        bool ceil_mode,
        bool count_include_pad) :
        kernel_width(kernel_width),
        kernel_height(kernel_height),
        stride_width(stride_width),
        stride_height(stride_height),
        padding_width(padding_width),
        padding_height(padding_height),
        ceil_mode(ceil_mode),
        count_include_pad(count_include_pad)
{ 

};

torch::AvgPool2d::~AvgPool2d()
{
		
};

Tensor torch::AvgPool2d::forward(Tensor input)
{
	Tensor output = input.type().tensor();

	SpatialAveragePooling_updateOutput(
		input,
		output,
		kernel_width,
		kernel_height,
		stride_width,
		stride_height,
		padding_width,
		padding_height,
		ceil_mode,
		count_include_pad);

	return output; 
};

string torch::AvgPool2d::tostring(int indentation_level)
{
	std::stringstream string_stream;

	string indentation = string(indentation_level, ' ');

	string_stream << indentation
					<< "AvgPool2d( "
					<< "kernel_size=(" << std::to_string(kernel_width) << ", " << std::to_string(kernel_height) << ") "
					<< "stride=(" << std::to_string(stride_width) << ", " << std::to_string(stride_height) << ") "
					<< "padding=(" << std::to_string(padding_width) << ", " << std::to_string(padding_height) << ") )"; 

	return string_stream.str();
};